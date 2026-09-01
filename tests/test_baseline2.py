from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline2.data import (
    BASE_WEATHER_COLUMNS,
    PV_WEATHER_COLUMNS,
    HEEWConditionDataset,
)
from baseline2.models import ConditionGuidanceTransformer, TMDMDiffusion


def _write_tiny_heew(root: Path) -> tuple[Path, Path]:
    timestamps = pd.DatetimeIndex(
        [
            pd.Timestamp(year=year, month=1, day=1, hour=hour)
            for year in range(2014, 2023)
            for hour in range(24)
        ]
    )
    base = {
        "Year": timestamps.year,
        "Month": timestamps.month,
        "Day": timestamps.day,
        "Hour": timestamps.hour,
    }
    time = np.arange(len(timestamps), dtype=np.float32)
    energy = pd.DataFrame(
        {
            **base,
            "Electricity": 100.0 + time,
            "Heat": 10.0 + 0.1 * time,
            "Cooling": 50.0 + 0.5 * time,
            "PV": np.maximum(0.0, np.sin((timestamps.hour - 6) * np.pi / 12)) * 20.0,
        }
    )
    weather_values = {
        column: time * (index + 1) / 100.0
        for index, column in enumerate(BASE_WEATHER_COLUMNS + PV_WEATHER_COLUMNS)
    }
    weather = pd.DataFrame({**base, **weather_values})
    weather["PV_IS_DAYLIGHT"] = ((timestamps.hour >= 7) & (timestamps.hour <= 18)).astype(np.float32)
    energy_path = root / "energy.csv"
    weather_path = root / "weather.csv"
    energy.to_csv(energy_path, index=False)
    weather.to_csv(weather_path, index=False)
    return energy_path, weather_path


def test_fixed_split_and_no_history(tmp_path: Path) -> None:
    energy_path, weather_path = _write_tiny_heew(tmp_path)
    train = HEEWConditionDataset(energy_path, weather_path, "train")
    validation = HEEWConditionDataset(energy_path, weather_path, "val")
    test = HEEWConditionDataset(energy_path, weather_path, "test")
    assert (len(train), len(validation), len(test)) == (7, 1, 1)
    conditions, pv_year, target, date = train[0]
    assert conditions.shape == (18, 24)
    assert pv_year.shape == (1, 24)
    assert target.shape == (4, 24)
    assert date == "2014-01-01"
    assert len(train[0]) == 4
    assert torch.allclose(train.targets.mean(dim=(0, 2)), torch.zeros(4), atol=1e-5)


def test_model_and_sampling_shapes() -> None:
    guidance = ConditionGuidanceTransformer(
        condition_channels=18,
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    conditions = torch.randn(2, 18, 24)
    pv_year = torch.randn(2, 1, 24)
    prediction, kl = guidance(conditions, pv_year)
    assert prediction.shape == (2, 24, 4)
    assert kl.ndim == 0
    diffusion = TMDMDiffusion(
        hidden_dim=32,
        num_layers=2,
        timesteps=4,
    )
    loss = diffusion.training_loss(torch.randn_like(prediction), prediction)
    assert torch.isfinite(loss)
    diffusion.eval()
    generated = diffusion.sample(prediction.detach())
    assert generated.shape == prediction.shape
    assert torch.isfinite(generated).all()

