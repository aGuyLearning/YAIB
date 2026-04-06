import logging
from pathlib import Path

import gin
import torch
import torchmetrics
from pypots.nn.modules.ts2vec import TS2VecEncoder
from pypots.representation.ts2vec.core import _TS2Vec
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
    encoded = encoded.masked_fill(~time_mask.unsqueeze(-1), 0.0)
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


@gin.configurable
class TS2Vec(DLWrapper):
    """Temporal TS2Vec-style encoder for self-supervised pretraining."""

    _supported_run_modes = [RunMode.pretrain]

    def __init__(
        self,
        input_size,
        hidden_dim: int = 128,
        encoder_layers: int = 4,
        dropout: float = 0.1,
        kernel_size: int = 3,
        max_crop_len: int | None = None,
        loss_alpha: float = 0.5,
        temporal_unit: int = 0,
        mask_mode: str = "binomial",
        use_observation_mask: bool = True,
        *args,
        **kwargs,
    ):
        kwargs.pop("loss", None)
        super().__init__(*args, input_size=input_size, loss=None, **kwargs)
        if max_crop_len is not None and max_crop_len < 2:
            raise ValueError("max_crop_len must be >= 2 when set")
        if not 0 <= loss_alpha <= 1:
            raise ValueError("loss_alpha must be between 0 and 1")
        if temporal_unit < 0:
            raise ValueError("temporal_unit must be >= 0")
        if loss_alpha != 0.5:
            logging.warning("PyPOTS _TS2Vec uses loss alpha 0.5 internally; ignoring TS2Vec.loss_alpha=%s.", loss_alpha)
        valid_mask_modes = {"binomial", "continuous", "all_true", "all_false", "mask_last"}
        if mask_mode not in valid_mask_modes:
            raise ValueError(f"mask_mode must be one of {sorted(valid_mask_modes)}, got {mask_mode!r}")
        in_features = input_size[2]
        encoder_in_features = in_features * 2 if use_observation_mask else in_features
        self.input_feature_dim = in_features
        self.encoder_input_feature_dim = encoder_in_features
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim
        self.encoder_layers = encoder_layers
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.max_crop_len = max_crop_len
        self.loss_alpha = loss_alpha
        self.temporal_unit = temporal_unit
        self.mask_mode = mask_mode
        self.use_observation_mask = use_observation_mask

        self.pypots_core = _TS2Vec(
            n_steps=input_size[1],
            n_features=encoder_in_features,
            n_pred_features=hidden_dim,
            d_hidden=hidden_dim,
            n_layers=encoder_layers,
            mask_mode=mask_mode,
            temporal_unit=temporal_unit,
        )

    @property
    def encoder(self):
        return self.pypots_core.encoder

    @encoder.setter
    def encoder(self, value):
        self.pypots_core.encoder = value

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ts2vec_config"] = {
            "backend": "pypots_core",
            "backbone": "pypots_ts2vec",
            "hidden_dim": int(self.hidden_dim),
            "output_dim": int(self.output_dim),
            "encoder_layers": int(self.encoder_layers),
            "kernel_size": int(self.kernel_size),
            "pooling": "masked_mean",
            "input_features": int(self.input_feature_dim),
            "encoder_input_features": int(self.encoder_input_feature_dim),
            "use_observation_mask": bool(self.use_observation_mask),
            "max_crop_len": None if self.max_crop_len is None else int(self.max_crop_len),
            "loss": "pypots_hierarchical_contrastive",
            "loss_alpha": 0.5,
            "temporal_unit": int(self.temporal_unit),
            "mask_mode": self.mask_mode,
        }
        return super().on_save_checkpoint(checkpoint)

    def set_metrics(self):
        return {}

    def _window_for_pypots_core(self, x: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor, int]:
        if mask is None:
            time_mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
        else:
            time_mask = _coerce_time_mask(mask, x)
        valid_counts = time_mask.long().sum(dim=1)
        min_valid = int(valid_counts.min().item())
        if min_valid < 2:
            window_len = max(min_valid, 1)
            return x[:, :window_len], time_mask[:, :window_len], window_len

        window_len = min_valid if self.max_crop_len is None else min(min_valid, int(self.max_crop_len))
        windowed: list[Tensor] = []
        windowed_masks: list[Tensor] = []
        for sample, sample_mask, valid_len in zip(x, time_mask, valid_counts.tolist()):
            valid_len = int(valid_len)
            if valid_len <= window_len:
                offset = 0
            else:
                offset = int(torch.randint(0, valid_len - window_len + 1, (1,), device=x.device).item())
            windowed.append(sample[offset : offset + window_len])
            windowed_masks.append(sample_mask[offset : offset + window_len])
        return torch.stack(windowed, dim=0), torch.stack(windowed_masks, dim=0), window_len

    def _prepare_pypots_inputs(self, x: Tensor, mask: Tensor | None = None, *, crop_to_valid_window: bool = False) -> dict:
        if crop_to_valid_window:
            x, mask, _ = self._window_for_pypots_core(x, mask)
        elif mask is not None:
            mask = _coerce_time_mask(mask, x)

        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), torch.nan)
        prepared = prepare_ts2vec_inputs(x, use_observation_mask=self.use_observation_mask)
        if mask is not None and not self.use_observation_mask:
            prepared = prepared.masked_fill(~mask.unsqueeze(-1), torch.nan)
        return {"X": prepared}

    def _zero_loss(self) -> Tensor:
        loss = next(self.parameters()).sum() * 0.0
        for param in list(self.parameters())[1:]:
            loss = loss + param.sum() * 0.0
        return loss

    def encode(self, x: Tensor, mask_mode: str | None = None, mask: Tensor | None = None) -> Tensor:
        prepared = self._prepare_pypots_inputs(x, mask)["X"]
        effective_mask = "all_true" if mask_mode is None else mask_mode
        try:
            return self.encoder(prepared, mask=effective_mask)
        except TypeError:
            return self.encoder(prepared)

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
        inputs = self._prepare_pypots_inputs(x, mask, crop_to_valid_window=True)
        if inputs["X"].shape[1] < 2:
            loss = self._zero_loss()
        else:
            results = self.pypots_core(inputs, calc_criterion=True)
            loss = results.get("loss", results.get("metric")).sum()
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
        self.encoder = TS2VecEncoder(
            n_features=encoder_in_features,
            n_pred_features=hidden_dim,
            d_hidden=hidden_dim,
            n_layers=encoder_layers,
            mask_mode="binomial",
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
        encoder_state = {
            k.replace("pypots_core.encoder.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("pypots_core.encoder.")
        }
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
        if mask is not None:
            x = x.masked_fill(~mask.to(device=x.device, dtype=torch.bool).unsqueeze(-1), torch.nan)
        prepared = prepare_ts2vec_inputs(x, use_observation_mask=self.use_observation_mask)
        if mask is not None and not self.use_observation_mask:
            prepared = prepared.masked_fill(~mask.to(device=x.device, dtype=torch.bool).unsqueeze(-1), torch.nan)
        try:
            encoded = self.encoder(prepared, mask="all_true")
        except TypeError:
            encoded = self.encoder(prepared)
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
