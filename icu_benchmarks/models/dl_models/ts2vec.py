import logging
from pathlib import Path

import gin
import torch
import torchmetrics
import torch.nn.functional as F
from torch import Tensor
from torch import nn as nn

from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import DLPredictionWrapper, DLWrapper


def _coerce_time_mask(mask: Tensor | None, encoded: Tensor) -> Tensor:
    if mask is None:
        return torch.ones(encoded.shape[:2], device=encoded.device, dtype=torch.bool)
    if mask.ndim != 2:
        raise ValueError(f"Expected time mask with shape [B, T], got {tuple(mask.shape)}")
    if tuple(mask.shape) != tuple(encoded.shape[:2]):
        raise ValueError(f"Mask shape {tuple(mask.shape)} does not match encoded shape {tuple(encoded.shape[:2])}")
    return mask.to(device=encoded.device, dtype=torch.bool)


def masked_mean_pool(encoded: Tensor, mask: Tensor | None = None) -> Tensor:
    """Pool encoder outputs over valid timesteps only."""

    time_mask = _coerce_time_mask(mask, encoded)
    weights = time_mask.unsqueeze(-1).to(encoded.dtype)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (encoded * weights).sum(dim=1) / denom


def prepare_ts2vec_inputs(x: Tensor, use_observation_mask: bool = True) -> Tensor:
    """Convert partially observed inputs into encoder-ready features.

    Missing values remain meaningful by concatenating a binary observed-value mask
    to zero-filled raw measurements.
    """

    observed = torch.isfinite(x)
    values = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if not use_observation_mask:
        return values
    return torch.cat([values, observed.to(values.dtype)], dim=-1)


class SamePadConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        receptive_field = (kernel_size - 1) * dilation + 1
        padding = receptive_field // 2
        self.trim = 1 if receptive_field % 2 == 0 else 0
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        if self.trim:
            out = out[..., :-self.trim]
        return out


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels: int, hidden_dim: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            SamePadConv1d(channels, hidden_dim, kernel_size=kernel_size, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            SamePadConv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.project = nn.Identity() if channels == hidden_dim else nn.Conv1d(channels, hidden_dim, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x) + self.project(x)


class DilatedConvEncoder(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        kernel_size: int = 3,
    ):
        super().__init__()
        blocks = []
        current_dim = in_features
        for idx in range(max(1, layers)):
            blocks.append(
                ResidualDilatedBlock(
                    channels=current_dim,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2**idx,
                    dropout=dropout,
                )
            )
            current_dim = hidden_dim
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        # [B, T, F] -> [B, F, T]
        out = x.transpose(1, 2)
        for block in self.blocks:
            out = block(out)
        return out.transpose(1, 2)


@gin.configurable
class TS2Vec(DLWrapper):
    """Temporal TS2Vec-style encoder for self-supervised pretraining."""

    _supported_run_modes = [RunMode.pretrain]

    def __init__(
        self,
        input_size,
        hidden_dim: int = 128,
        projection_dim: int = 128,
        encoder_layers: int = 4,
        dropout: float = 0.1,
        aug_noise_std: float = 0.05,
        temperature: float = 0.07,
        kernel_size: int = 3,
        min_crop_ratio: float = 0.5,
        use_observation_mask: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, input_size=input_size, **kwargs)
        in_features = input_size[2]
        encoder_in_features = in_features * 2 if use_observation_mask else in_features
        self.input_feature_dim = in_features
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.encoder_layers = encoder_layers
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.aug_noise_std = aug_noise_std
        self.temperature = temperature
        self.min_crop_ratio = min_crop_ratio
        self.use_observation_mask = use_observation_mask

        self.encoder = DilatedConvEncoder(
            in_features=encoder_in_features,
            hidden_dim=hidden_dim,
            layers=encoder_layers,
            dropout=dropout,
            kernel_size=kernel_size,
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ts2vec_config"] = {
            "backbone": "dilated_conv_v1",
            "hidden_dim": int(self.hidden_dim),
            "projection_dim": int(self.projection_dim),
            "encoder_layers": int(self.encoder_layers),
            "kernel_size": int(self.kernel_size),
            "pooling": "masked_mean",
            "input_features": int(self.input_feature_dim),
            "use_observation_mask": bool(self.use_observation_mask),
        }
        return super().on_save_checkpoint(checkpoint)

    def set_metrics(self):
        return {}

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(prepare_ts2vec_inputs(x, use_observation_mask=self.use_observation_mask))

    def _augment(self, x: Tensor) -> Tensor:
        if self.aug_noise_std <= 0:
            return x
        noise = torch.randn_like(x) * self.aug_noise_std
        return torch.where(torch.isfinite(x), x + noise, x)

    def _sample_crops(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        valid_counts = mask.long().sum(dim=1)
        min_valid = int(valid_counts.min().item())
        if min_valid <= 2:
            return x, x

        min_crop_len = max(2, int(min_valid * self.min_crop_ratio))
        crop_len = int(torch.randint(min_crop_len, min_valid + 1, (1,), device=x.device).item())

        view_a: list[Tensor] = []
        view_b: list[Tensor] = []
        for sample, valid_len in zip(x, valid_counts.tolist()):
            valid_len = int(valid_len)
            if valid_len <= crop_len:
                left_a = 0
                left_b = 0
            else:
                left_a = int(torch.randint(0, valid_len - crop_len + 1, (1,), device=x.device).item())
                left_b = int(torch.randint(0, valid_len - crop_len + 1, (1,), device=x.device).item())
            view_a.append(sample[left_a : left_a + crop_len])
            view_b.append(sample[left_b : left_b + crop_len])

        return torch.stack(view_a, dim=0), torch.stack(view_b, dim=0)

    def _timestamp_contrastive_loss(self, z1: Tensor, z2: Tensor) -> Tensor:
        # Contrast across the batch independently for each timestamp.
        p1 = F.normalize(self.projector(z1), dim=-1)
        p2 = F.normalize(self.projector(z2), dim=-1)
        logits = torch.einsum("btd,ctd->tbc", p1, p2) / self.temperature
        labels = torch.arange(logits.shape[1], device=logits.device).expand(logits.shape[0], -1)
        loss_a = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        loss_b = F.cross_entropy(logits.transpose(1, 2).reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return 0.5 * (loss_a + loss_b)

    def _hierarchical_contrastive_loss(self, z1: Tensor, z2: Tensor) -> Tensor:
        total = z1.new_tensor(0.0)
        levels = 0
        cur1, cur2 = z1, z2
        while cur1.shape[1] > 1:
            total = total + self._timestamp_contrastive_loss(cur1, cur2)
            levels += 1
            cur1 = F.max_pool1d(cur1.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
            cur2 = F.max_pool1d(cur2.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
        total = total + self._timestamp_contrastive_loss(cur1, cur2)
        levels += 1
        return total / max(levels, 1)

    def _contrastive_loss(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        time_mask = _coerce_time_mask(mask, x)
        view_a, view_b = self._sample_crops(self._augment(x), time_mask)
        z1 = self.encode(view_a)
        z2 = self.encode(view_b)
        return self._hierarchical_contrastive_loss(z1, z2)

    def forward(self, x: Tensor) -> Tensor:
        return self.encode(x)

    def encode_stay(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        return masked_mean_pool(self.encode(x), mask)

    def step_fn(self, batch, step_prefix=""):
        mask = None
        if isinstance(batch, (list, tuple)):
            x = batch[0]
            if len(batch) > 1:
                mask = batch[1]
        else:
            x = batch
        x = x.float().to(self.device)
        if mask is not None:
            mask = mask.to(self.device).bool()
        loss = self._contrastive_loss(x, mask)
        self.log(f"{step_prefix}/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss


@gin.configurable
class TS2VecProbe(DLPredictionWrapper):
    """Downstream head/probe model using a pretrained TS2Vec encoder."""

    _supported_run_modes = [RunMode.classification, RunMode.regression]

    def __init__(
        self,
        input_size,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
        encoder_layers: int = 4,
        kernel_size: int = 3,
        use_observation_mask: bool = True,
        pretrained_encoder_path: str | None = None,
        freeze_encoder: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, input_size=input_size, num_classes=num_classes, **kwargs)
        in_features = input_size[2]
        encoder_in_features = in_features * 2 if use_observation_mask else in_features
        self.use_observation_mask = use_observation_mask
        self.encoder = DilatedConvEncoder(
            in_features=encoder_in_features,
            hidden_dim=hidden_dim,
            layers=encoder_layers,
            dropout=dropout,
            kernel_size=kernel_size,
        )
        self.logit = nn.Linear(hidden_dim, num_classes)
        self.pretrained_encoder_path = pretrained_encoder_path
        self.freeze_encoder = freeze_encoder
        if self.pretrained_encoder_path:
            self.load_pretrained_encoder(self.pretrained_encoder_path)
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def load_pretrained_encoder(self, checkpoint_path: str):
        if not Path(checkpoint_path).is_file():
            logging.warning(
                "Pretrained encoder checkpoint not found: %s — skipping encoder weight loading "
                "(this is expected when loading a fully-trained probe via --eval --source-dir).",
                checkpoint_path,
            )
            return
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        encoder_state = {k.replace("encoder.", "", 1): v for k, v in state_dict.items() if k.startswith("encoder.")}
        if encoder_state:
            current_state = self.encoder.state_dict()
            compatible_state = {
                key: value
                for key, value in encoder_state.items()
                if key in current_state and current_state[key].shape == value.shape
            }
            self.encoder.load_state_dict(compatible_state, strict=False)

    @staticmethod
    def _move_data(data, device: torch.device):
        if isinstance(data, list):
            for i in range(len(data)):
                data[i] = data[i].float().to(device)
            return data
        return data.float().to(device)

    @staticmethod
    def _last_valid_targets(labels: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor]:
        if labels.ndim == 1:
            keep = torch.isfinite(labels)
            return labels, keep
        if mask is None:
            mask = torch.ones_like(labels, dtype=torch.bool)
        mask = mask.bool()
        valid_counts = mask.long().sum(dim=1)
        keep = valid_counts > 0
        last_idx = valid_counts.clamp_min(1) - 1
        targets = labels.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        return targets, keep

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        encoded = self.encoder(prepare_ts2vec_inputs(x, use_observation_mask=self.use_observation_mask))
        pooled = masked_mean_pool(encoded, mask)
        return self.logit(pooled)

    def step_fn(self, element, step_prefix=""):
        if len(element) == 2:
            data, labels = element[0], element[1].to(self.device)
            data = self._move_data(data, self.device)
            mask = None
        elif len(element) == 3:
            data, labels, mask = element[0], element[1].to(self.device), element[2].to(self.device).bool()
            data = self._move_data(data, self.device)
        else:
            raise Exception("Loader should return either (data, label) or (data, label, mask)")

        out = self(data, mask=mask)

        if len(out) == 2 and isinstance(out, tuple):
            out, aux_loss = out
        else:
            aux_loss = 0

        target, keep = self._last_valid_targets(labels, mask)
        prediction = out[keep].to(self.device)
        target = target[keep].to(self.device)

        if prediction.shape[-1] > 1 and self.run_mode == RunMode.classification:
            class_weight = self.loss_weights.to(self.device) if isinstance(self.loss_weights, Tensor) else None
            loss = self.loss(prediction, target.long(), weight=class_weight) + aux_loss
        elif self.run_mode == RunMode.regression:
            loss = self.loss(prediction[:, 0], target.float()) + aux_loss
        else:
            raise ValueError(f"Run mode {self.run_mode} not yet supported. Please implement it.")

        transformed_output = self.output_transform((prediction, target))
        for key, value in self.metrics[step_prefix].items():
            if isinstance(value, torchmetrics.Metric):
                if key == "Binary_Fairness":
                    feature_names = key.feature_helper(self.trainer)
                    value.update(transformed_output[0], transformed_output[1], data, feature_names)
                else:
                    value.update(transformed_output[0], transformed_output[1])
            else:
                value.update(transformed_output)
        self.log(f"{step_prefix}/loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss
