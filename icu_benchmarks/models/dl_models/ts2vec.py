import logging
from pathlib import Path

import gin
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn as nn

from icu_benchmarks.constants import RunMode
from icu_benchmarks.models.wrappers import DLPredictionWrapper, DLWrapper


@gin.configurable
class TS2Vec(DLWrapper):
    """Lightweight TS2Vec-style encoder for self-supervised pretraining."""

    _supported_run_modes = [RunMode.pretrain]

    def __init__(
        self,
        input_size,
        hidden_dim: int = 128,
        projection_dim: int = 64,
        encoder_layers: int = 3,
        dropout: float = 0.1,
        aug_noise_std: float = 0.05,
        temperature: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, input_size=input_size, **kwargs)
        in_features = input_size[2]
        self.aug_noise_std = aug_noise_std
        self.temperature = temperature

        layers = [nn.Linear(in_features, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(max(0, encoder_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        self.encoder = nn.Sequential(*layers)
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def set_metrics(self):
        return {}

    def encode(self, x: Tensor) -> Tensor:
        # x: [B, T, F] -> [B, T, H]
        return self.encoder(x)

    def _augment(self, x: Tensor) -> Tensor:
        noise = torch.randn_like(x) * self.aug_noise_std
        return x + noise

    def _contrastive_loss(self, z1: Tensor, z2: Tensor) -> Tensor:
        # pooled instance representations: [B, D]
        p1 = F.normalize(self.projector(z1.mean(dim=1)), dim=-1)
        p2 = F.normalize(self.projector(z2.mean(dim=1)), dim=-1)
        logits = torch.matmul(p1, p2.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_a + loss_b)

    def forward(self, x: Tensor) -> Tensor:
        return self.encode(x)

    def step_fn(self, batch, step_prefix=""):
        # Support tensor-only batches and tuple/list batches.
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        x = x.float().to(self.device)
        z1 = self.encode(self._augment(x))
        z2 = self.encode(self._augment(x))
        loss = self._contrastive_loss(z1, z2)
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
        encoder_layers: int = 3,
        pretrained_encoder_path: str | None = None,
        freeze_encoder: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, input_size=input_size, num_classes=num_classes, **kwargs)
        in_features = input_size[2]
        layers = [nn.Linear(in_features, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(max(0, encoder_layers - 1)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        self.encoder = nn.Sequential(*layers)
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

    def forward(self, x: Tensor) -> Tensor:
        encoded = self.encoder(x)
        return self.logit(encoded)
