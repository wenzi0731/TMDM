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
from baseline2.metrics import compute_precision_recall, summarize


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


def test_channel_metrics_include_normalized_distribution_scores() -> None:
    random = np.random.default_rng(7)
    target = random.normal(size=(8, 4, 24))
    samples = target[:, None] + random.normal(scale=0.2, size=(8, 6, 4, 24))
    labels = ["Electricity", "Heat", "Cooling", "PV"]
    metrics = summarize(
        samples,
        target,
        labels,
        target_mean=np.array([10.0, 20.0, 30.0, 40.0]),
        target_std=np.array([2.0, 4.0, 5.0, 8.0]),
        precision_recall_k=2,
    )
    for label in labels:
        for metric in ("RMSE_Z", "MAE_Z", "Precision_Z", "Recall_Z", "CR", "IW"):
            assert np.isfinite(metrics[f"{label}_{metric}"])
        assert 0.0 <= metrics[f"{label}_Precision_Z"] <= 1.0
        assert 0.0 <= metrics[f"{label}_Recall_Z"] <= 1.0
        assert 0.0 <= metrics[f"{label}_CR"] <= 1.0
        assert metrics[f"{label}_IW"] >= 0.0


def test_precision_recall_is_one_for_identical_sets() -> None:
    trajectories = np.arange(8 * 24, dtype=np.float64).reshape(8, 24)
    precision, recall = compute_precision_recall(trajectories, trajectories, k=2)
    assert precision == 1.0
    assert recall == 1.0
