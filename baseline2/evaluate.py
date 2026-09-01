from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from baseline2.data import HEEWConditionDataset
from baseline2.metrics import (
    save_global_pearson_comparison,
    save_global_pearson_plot,
    save_random_timeseries_plots,
    summarize,
)
from baseline2.models import build_diffusion, build_guidance_model
from baseline2.runtime import (
    choose_device,
    load_checkpoint,
    resolve_path,
    seed_everything,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TMDM-Cond Baseline 2.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--energy-path", default=None)
    parser.add_argument("--weather-path", default=None)
    parser.add_argument("--output-dir", default="evaluation_results/baseline2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-scenarios", type=int, default=None)
    parser.add_argument("--max-days", type=int, default=None)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    seed = int(config["run"]["seed"])
    seed_everything(seed)
    device = choose_device(args.device)
    data_cfg = config["data"]
    dataset = HEEWConditionDataset(
        resolve_path(args.energy_path or data_cfg["energy_path"]),
        resolve_path(args.weather_path or data_cfg["weather_path"]),
        "test",
        data_cfg["weather_feature_set"],
    )
    evaluation_set = dataset
    if args.max_days is not None:
        if args.max_days <= 0:
            parser.error("--max-days must be positive")
        evaluation_set = Subset(dataset, range(min(args.max_days, len(dataset))))
    loader = DataLoader(
        evaluation_set,
        batch_size=int(config["eval"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    guidance_model = build_guidance_model(config, dataset.condition_channels).to(device)
    guidance_model.load_state_dict(checkpoint["guidance_model"], strict=True)
    diffusion = build_diffusion(config).to(device)
    diffusion.load_state_dict(checkpoint["diffusion_model"], strict=True)
    guidance_model.eval()
    diffusion.eval()

    scenario_count = int(args.num_scenarios or config["eval"]["num_scenarios"])
    if scenario_count <= 0:
        parser.error("--num-scenarios must be positive")
    all_scenarios: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_points: list[torch.Tensor] = []
    all_dates: list[str] = []
    seed_everything(seed)
    with torch.no_grad():
        for conditions, pv_year, target, dates in loader:
            conditions = conditions.to(device)
            pv_year = pv_year.to(device)
            point, _kl = guidance_model(conditions, pv_year)
            batch_size = point.shape[0]
            tiled_point = point[:, None].expand(
                batch_size, scenario_count, 24, 4
            ).reshape(batch_size * scenario_count, 24, 4)
            generated = diffusion.sample(tiled_point).reshape(
                batch_size, scenario_count, 24, 4
            ).permute(0, 1, 3, 2)
            point_channels_first = point.permute(0, 2, 1)
            scenarios_phys = dataset.denormalize(generated).cpu()
            targets_phys = dataset.denormalize(target).cpu()
            points_phys = dataset.denormalize(point_channels_first).cpu()
            scenarios_phys[:, :, 3, :].clamp_(min=0.0)
            points_phys[:, 3, :].clamp_(min=0.0)
            all_scenarios.append(scenarios_phys)
            all_targets.append(targets_phys)
            all_points.append(points_phys)
            all_dates.extend(str(value) for value in dates)

    scenarios = torch.cat(all_scenarios).numpy()
    targets = torch.cat(all_targets).numpy()
    points = torch.cat(all_points).numpy()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "baseline2_scenarios.npz"
    np.savez_compressed(
        archive,
        scenarios=scenarios,
        targets=targets,
        deterministic=points,
        dates=np.asarray(all_dates),
        channel_names=np.asarray(dataset.target_cols),
        seed=seed,
        sampler="tmdm_ancestral",
        num_scenarios=scenario_count,
    )
    metrics = summarize(scenarios, targets, dataset.target_cols)
    random_timeseries_dir = output_dir / "random_timeseries_50"
    random_paths = save_random_timeseries_plots(
        targets,
        scenarios,
        all_dates,
        random_timeseries_dir,
        dataset.target_cols,
        seed,
        50,
    )
    pearson_dir = output_dir / "pearson"
    pearson_metrics = save_global_pearson_comparison(
        targets, scenarios, pearson_dir, dataset.target_cols
    )
    global_pearson = save_global_pearson_plot(
        targets, output_dir / "global_pearson.png", dataset.target_cols
    )
    global_metrics = output_dir / "global_metrics.csv"
    pd.DataFrame([{**metrics, **pearson_metrics}]).to_csv(global_metrics, index=False)
    summary = {
        "method": "TMDM-condition-only-baseline2",
        "checkpoint": str(resolve_path(args.checkpoint)),
        "split": "test-2022",
        "num_days": len(all_dates),
        "num_scenarios": scenario_count,
        "sampler": "tmdm_ancestral",
        "seed": seed,
        "archive": str(archive),
        **metrics,
        **pearson_metrics,
        "global_metrics": str(global_metrics),
        "global_pearson": str(global_pearson),
        "pearson_dir": str(pearson_dir),
        "random_timeseries_dir": str(random_timeseries_dir),
        "random_timeseries_count": len(random_paths),
    }
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

