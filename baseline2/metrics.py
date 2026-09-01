from __future__ import annotations

from pathlib import Path

import numpy as np


def crps_map(samples: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pointwise ensemble CRPS for samples [N,S,C,T] and targets [N,C,T]."""
    values = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if values.ndim != 4 or truth.shape != (
        values.shape[0],
        values.shape[2],
        values.shape[3],
    ):
        raise ValueError("Expected samples [N,S,C,T] and target [N,C,T].")
    term1 = np.mean(np.abs(values - truth[:, None]), axis=1)
    ordered = np.sort(values, axis=1)
    ensemble_size = values.shape[1]
    coefficients = (2 * np.arange(ensemble_size) - ensemble_size + 1).reshape(
        1, ensemble_size, 1, 1
    )
    mean_pairwise = (
        2.0 * np.sum(coefficients * ordered, axis=1) / ensemble_size**2
    )
    return term1 - 0.5 * mean_pairwise


def summarize(
    samples: np.ndarray, target: np.ndarray, labels: list[str]
) -> dict[str, float]:
    ensemble_mean = samples.mean(axis=1)
    error = ensemble_mean - target
    scores = crps_map(samples, target)
    result: dict[str, float] = {}
    for channel, label in enumerate(labels):
        channel_target = target[:, channel]
        channel_crps = scores[:, channel]
        lower = np.quantile(samples[:, :, channel], 0.05, axis=1)
        upper = np.quantile(samples[:, :, channel], 0.95, axis=1)
        result[f"{label}_RMSE"] = float(
            np.sqrt(np.mean(error[:, channel] ** 2))
        )
        result[f"{label}_MAE"] = float(np.mean(np.abs(error[:, channel])))
        result[f"{label}_CRPS"] = float(np.mean(channel_crps))
        result[f"{label}_nCRPS"] = float(
            np.sum(channel_crps) / (np.sum(np.abs(channel_target)) + 1e-8)
        )
        result[f"{label}_Coverage90"] = float(
            np.mean((channel_target >= lower) & (channel_target <= upper))
        )
        result[f"{label}_IntervalWidth90"] = float(np.mean(upper - lower))
    result["mean_nCRPS"] = float(
        np.sum(scores) / (np.sum(np.abs(target)) + 1e-8)
    )
    return result


def compute_global_pearson_matrix(total_data: np.ndarray) -> np.ndarray:
    values = np.asarray(total_data, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"Expected [N,C,T], got {values.shape}.")
    flattened = values.transpose(0, 2, 1).reshape(-1, values.shape[1])
    return np.corrcoef(flattened, rowvar=False)


def _save_heatmap(
    matrix: np.ndarray,
    title: str,
    save_path: str | Path,
    labels: list[str],
    cmap: str,
    vmin: float,
    vmax: float,
) -> Path:
    import matplotlib.pyplot as plt

    destination = Path(save_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axis)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(destination, dpi=300)
    plt.close(figure)
    return destination


def save_global_pearson_plot(
    total_real: np.ndarray, save_path: str | Path, labels: list[str]
) -> Path:
    return _save_heatmap(
        compute_global_pearson_matrix(total_real),
        "Global Pearson Correlation",
        save_path,
        labels,
        "coolwarm",
        -1.0,
        1.0,
    )


def save_global_pearson_comparison(
    total_real: np.ndarray,
    total_generated: np.ndarray,
    output_dir: str | Path,
    labels: list[str],
) -> dict[str, float]:
    real = np.asarray(total_real, dtype=np.float64)
    generated = np.asarray(total_generated, dtype=np.float64)
    if real.ndim != 3 or generated.ndim != 4:
        raise ValueError("Expected real [N,C,T] and generated [N,S,C,T].")
    real_correlation = compute_global_pearson_matrix(real)
    generated_correlation = compute_global_pearson_matrix(
        generated.reshape(-1, generated.shape[2], generated.shape[3])
    )
    difference = np.abs(generated_correlation - real_correlation)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for matrix, title, filename, cmap, vmin, vmax in (
        (real_correlation, "Real Global Pearson", "real_global_pearson.png", "coolwarm", -1.0, 1.0),
        (generated_correlation, "Generated Global Pearson", "generated_global_pearson.png", "coolwarm", -1.0, 1.0),
        (difference, "Pearson Absolute Error", "pearson_difference.png", "Reds", 0.0, 1.0),
    ):
        _save_heatmap(matrix, title, destination / filename, labels, cmap, vmin, vmax)
    return {
        "Pearson_MAE": float(difference.mean()),
        "Pearson_RMSE": float(np.sqrt(np.mean(difference**2))),
    }


def save_random_timeseries_plots(
    total_real: np.ndarray,
    total_generated: np.ndarray,
    dates: list[str],
    output_dir: str | Path,
    labels: list[str],
    seed: int = 42,
    n_plots: int = 50,
) -> list[Path]:
    import matplotlib.pyplot as plt

    real = np.asarray(total_real)
    generated = np.asarray(total_generated)
    if real.ndim != 3 or generated.ndim != 4:
        raise ValueError("Expected real [N,C,T] and generated [N,S,C,T].")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    selected = np.random.default_rng(seed).choice(
        real.shape[0], size=min(n_plots, real.shape[0]), replace=False
    )
    saved: list[Path] = []
    for index in selected:
        figure, axes = plt.subplots(len(labels), 1, figsize=(10, 3 * len(labels)), sharex=True)
        if len(labels) == 1:
            axes = [axes]
        for channel, label in enumerate(labels):
            for scenario in range(min(100, generated.shape[1])):
                axes[channel].plot(generated[index, scenario, channel], color="red", alpha=0.1, linewidth=1)
            axes[channel].plot(real[index, channel], color="black", linewidth=2, linestyle="--", label="Ground Truth")
            axes[channel].set_title(label)
            axes[channel].grid(True, alpha=0.3)
            axes[channel].legend()
        figure.suptitle(f"Generated scenarios - {dates[index]}")
        figure.tight_layout()
        save_path = destination / f"{dates[index]}.png"
        figure.savefig(save_path, dpi=200)
        plt.close(figure)
        saved.append(save_path)
    return saved

