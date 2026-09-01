from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionGuidanceTransformer(nn.Module):
    """Condition-only variational Transformer derived from TMDM guidance."""

    def __init__(
        self,
        condition_channels: int,
        target_channels: int = 4,
        seq_len: int = 24,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        pv_channel_idx: int = 3,
        use_pv_year_head: bool = True,
        pv_year_film_scale: float = 0.2,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pv_channel_idx = pv_channel_idx
        self.pv_year_film_scale = pv_year_film_scale
        self.input_projection = nn.Linear(condition_channels, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.z_mean = nn.Linear(hidden_dim, hidden_dim)
        self.z_logvar = nn.Linear(hidden_dim, hidden_dim)
        self.z_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.output_head = nn.Linear(hidden_dim, target_channels)
        if use_pv_year_head:
            self.pv_year_film = nn.Linear(1, 2 * hidden_dim)
            self.pv_head = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.pv_year_film.weight)
            nn.init.zeros_(self.pv_year_film.bias)
        else:
            self.pv_year_film = None
            self.pv_head = None

    def forward(
        self, conditions: torch.Tensor, pv_year: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if conditions.ndim != 3 or conditions.shape[-1] != self.seq_len:
            raise ValueError("conditions must be [B,C,24]")
        tokens = self.input_projection(conditions.transpose(1, 2)) + self.position
        tokens = self.encoder(tokens)
        mean = self.z_mean(tokens)
        logvar = self.z_logvar(tokens).clamp(-12.0, 12.0)
        if self.training:
            latent = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        else:
            latent = mean
        latent = self.z_out(latent)
        output = self.output_head(latent)
        if self.pv_year_film is not None:
            year = pv_year.transpose(1, 2).to(latent.dtype)
            gamma_raw, beta = self.pv_year_film(year).chunk(2, dim=-1)
            gamma = 1.0 + self.pv_year_film_scale * torch.tanh(gamma_raw)
            pv = self.pv_head(gamma * latent + beta)
            output = output.clone()
            output[:, :, self.pv_channel_idx : self.pv_channel_idx + 1] = pv
        kl = -0.5 * torch.mean(1.0 + logvar - mean.square() - logvar.exp())
        return output, kl


class ConditionalLinear(nn.Module):
    """TMDM timestep-modulated linear layer."""

    def __init__(self, num_in: int, num_out: int, num_steps: int) -> None:
        super().__init__()
        self.linear = nn.Linear(num_in, num_out)
        self.step_embedding = nn.Embedding(num_steps, num_out)
        nn.init.uniform_(self.step_embedding.weight)

    def forward(self, values: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        gamma = self.step_embedding(timesteps).unsqueeze(1)
        return gamma * self.linear(values)


class TMDMNoiseModel(nn.Module):
    """Official TMDM-style MLP conditioned on the Transformer point forecast."""

    def __init__(
        self,
        target_channels: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_steps: int = 100,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        layers: list[ConditionalLinear] = [
            ConditionalLinear(2 * target_channels, hidden_dim, num_steps)
        ]
        layers.extend(
            ConditionalLinear(hidden_dim, hidden_dim, num_steps)
            for _ in range(num_layers - 1)
        )
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_dim, target_channels)

    def forward(
        self, noisy_target: torch.Tensor, point_forecast: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        hidden = torch.cat([noisy_target, point_forecast], dim=-1)
        for layer in self.layers:
            hidden = F.softplus(layer(hidden, timesteps))
        return self.output(hidden)


def make_beta_schedule(
    schedule: str, num_timesteps: int, start: float, end: float
) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(start, end, num_timesteps)
    if schedule == "cosine":
        offset = 0.008
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
        alpha_bar = torch.cos(((steps / num_timesteps + offset) / (1 + offset)) * math.pi / 2).square()
        alpha_bar = alpha_bar / alpha_bar[0]
        return (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999).float()
    raise ValueError(f"Unknown beta schedule {schedule!r}")


def _extract(values: torch.Tensor, timesteps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    selected = values.gather(0, timesteps.to(values.device))
    return selected.reshape(timesteps.shape[0], *([1] * (reference.ndim - 1)))


class TMDMDiffusion(nn.Module):
    """Shifted-mean diffusion used by TMDM, adapted to four joint channels."""

    def __init__(
        self,
        target_channels: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 3,
        timesteps: int = 100,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()
        self.num_timesteps = int(timesteps)
        betas = make_beta_schedule(beta_schedule, self.num_timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar_sqrt", alpha_bar.sqrt())
        self.register_buffer("one_minus_alpha_bar_sqrt", (1.0 - alpha_bar).sqrt())
        self.noise_model = TMDMNoiseModel(
            target_channels=target_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_steps=self.num_timesteps,
        )

    def q_sample(
        self,
        target: torch.Tensor,
        point_forecast: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha = _extract(self.alpha_bar_sqrt, timesteps, target)
        sqrt_one_minus = _extract(self.one_minus_alpha_bar_sqrt, timesteps, target)
        return (
            sqrt_alpha * target
            + (1.0 - sqrt_alpha) * point_forecast
            + sqrt_one_minus * noise
        )

    def training_loss(
        self, target: torch.Tensor, point_forecast: torch.Tensor
    ) -> torch.Tensor:
        batch_size = target.shape[0]
        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=target.device)
        noise = torch.randn_like(target)
        noisy = self.q_sample(target, point_forecast, timesteps, noise)
        predicted = self.noise_model(noisy, point_forecast, timesteps)
        return F.mse_loss(predicted, noise)

    def _reverse_step(
        self,
        current: torch.Tensor,
        point_forecast: torch.Tensor,
        timestep: int,
    ) -> torch.Tensor:
        batch_size = current.shape[0]
        t = torch.full((batch_size,), timestep, dtype=torch.long, device=current.device)
        alpha_t = _extract(self.alphas, t, current)
        sqrt_one_minus_t = _extract(self.one_minus_alpha_bar_sqrt, t, current)
        sqrt_alpha_bar_t = (1.0 - sqrt_one_minus_t.square()).sqrt()
        predicted_noise = self.noise_model(current, point_forecast, t)
        y0 = (
            current
            - (1.0 - sqrt_alpha_bar_t) * point_forecast
            - predicted_noise * sqrt_one_minus_t
        ) / sqrt_alpha_bar_t.clamp_min(1e-8)
        if timestep == 0:
            return y0

        previous_t = torch.full(
            (batch_size,), timestep - 1, dtype=torch.long, device=current.device
        )
        sqrt_one_minus_previous = _extract(
            self.one_minus_alpha_bar_sqrt, previous_t, current
        )
        sqrt_alpha_bar_previous = (1.0 - sqrt_one_minus_previous.square()).sqrt()
        denominator = sqrt_one_minus_t.square().clamp_min(1e-8)
        gamma0 = (1.0 - alpha_t) * sqrt_alpha_bar_previous / denominator
        gamma1 = sqrt_one_minus_previous.square() * alpha_t.sqrt() / denominator
        gamma2 = 1.0 + (sqrt_alpha_bar_t - 1.0) * (
            alpha_t.sqrt() + sqrt_alpha_bar_previous
        ) / denominator
        mean = gamma0 * y0 + gamma1 * current + gamma2 * point_forecast
        variance = (
            sqrt_one_minus_previous.square() / denominator * (1.0 - alpha_t)
        ).clamp_min(0.0)
        return mean + variance.sqrt() * torch.randn_like(current)

    @torch.no_grad()
    def sample(self, point_forecast: torch.Tensor) -> torch.Tensor:
        current = point_forecast + torch.randn_like(point_forecast)
        for timestep in reversed(range(self.num_timesteps)):
            current = self._reverse_step(current, point_forecast, timestep)
        return current


def build_guidance_model(
    config: dict[str, Any], condition_channels: int
) -> ConditionGuidanceTransformer:
    model = config["guidance_model"]
    return ConditionGuidanceTransformer(
        condition_channels=condition_channels,
        target_channels=4,
        seq_len=int(config["data"]["seq_len"]),
        hidden_dim=int(model["hidden_dim"]),
        num_layers=int(model["num_layers"]),
        num_heads=int(model["num_heads"]),
        dropout=float(model["dropout"]),
        use_pv_year_head=bool(model.get("use_pv_year_head", True)),
        pv_year_film_scale=float(model.get("pv_year_film_scale", 0.2)),
    )


def build_diffusion(config: dict[str, Any]) -> TMDMDiffusion:
    model = config["diffusion_model"]
    diffusion = config["diffusion"]
    return TMDMDiffusion(
        target_channels=4,
        hidden_dim=int(model["hidden_dim"]),
        num_layers=int(model["num_layers"]),
        timesteps=int(diffusion["timesteps"]),
        beta_schedule=str(diffusion.get("beta_schedule", "linear")),
        beta_start=float(diffusion.get("beta_start", 1e-4)),
        beta_end=float(diffusion.get("beta_end", 2e-2)),
    )


def checkpoint_payload(
    config: dict[str, Any],
    guidance_model: nn.Module,
    diffusion: nn.Module | None,
    epoch: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": 1,
        "method": "TMDM-condition-only-baseline2",
        "epoch": epoch,
        "guidance_model": deepcopy(guidance_model.state_dict()),
        "config": deepcopy(config),
        "metrics": deepcopy(metrics),
    }
    if diffusion is not None:
        payload["diffusion_model"] = deepcopy(diffusion.state_dict())
    return payload

