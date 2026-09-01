from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from baseline2.data import HEEWConditionDataset
from baseline2.models import (
    TMDMDiffusion,
    build_diffusion,
    build_guidance_model,
    checkpoint_payload,
)
from baseline2.runtime import (
    choose_device,
    load_checkpoint,
    load_config,
    resolve_path,
    seed_everything,
    seed_worker,
)


def make_loader(
    dataset: HEEWConditionDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def guidance_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    kl_loss: torch.Tensor,
    k_z: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    mse = F.mse_loss(prediction, target)
    total = mse + k_z * kl_loss
    return total, {"mse": float(mse.detach()), "kl": float(kl_loss.detach())}


def joint_objective(
    diffusion: TMDMDiffusion,
    prediction: torch.Tensor,
    target: torch.Tensor,
    kl_loss: torch.Tensor,
    k_cond: float,
    k_z: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    diffusion_loss = diffusion.training_loss(target, prediction)
    gaussian_nll = 0.5 * (math.log(2.0 * math.pi) + (target - prediction).square()).mean()
    condition_loss = gaussian_nll + k_z * kl_loss
    total = diffusion_loss + k_cond * condition_loss
    return total, {
        "diffusion": float(diffusion_loss.detach()),
        "condition_nll": float(gaussian_nll.detach()),
        "kl": float(kl_loss.detach()),
    }


def _mean(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    return {
        key: sum(record[key] for record in records) / len(records)
        for key in records[0]
    }


def run_guidance_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    k_z: float,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip: float = 1.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    records: list[dict[str, float]] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for conditions, pv_year, target, _dates in loader:
            conditions = conditions.to(device)
            pv_year = pv_year.to(device)
            target = target.to(device).transpose(1, 2)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction, kl_loss = model(conditions, pv_year)
            loss, parts = guidance_objective(prediction, target, kl_loss, k_z)
            if training:
                loss.backward()
                clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
            records.append({"loss": float(loss.detach()), **parts})
    return _mean(records)


def run_joint_epoch(
    guidance_model: torch.nn.Module,
    diffusion: TMDMDiffusion,
    loader: DataLoader,
    device: torch.device,
    k_cond: float,
    k_z: float,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip: float = 1.0,
) -> dict[str, float]:
    training = optimizer is not None
    guidance_model.train(training)
    diffusion.train(training)
    records: list[dict[str, float]] = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for conditions, pv_year, target, _dates in loader:
            conditions = conditions.to(device)
            pv_year = pv_year.to(device)
            target = target.to(device).transpose(1, 2)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction, kl_loss = guidance_model(conditions, pv_year)
            loss, parts = joint_objective(
                diffusion, prediction, target, kl_loss, k_cond, k_z
            )
            if training:
                loss.backward()
                parameters: Iterable[torch.nn.Parameter] = list(
                    guidance_model.parameters()
                ) + list(diffusion.parameters())
                clip_grad_norm_(parameters, gradient_clip)
                optimizer.step()
            records.append({"loss": float(loss.detach()), **parts})
    return _mean(records)


def train_guidance(
    config: dict,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    run_dir: Path,
) -> Path:
    train_cfg = config["train"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["guidance_learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    patience = int(train_cfg["patience"])
    best = float("inf")
    stale = 0
    checkpoint_path = run_dir / "best_guidance.pt"
    for epoch in range(1, int(train_cfg["guidance_epochs"]) + 1):
        train_metrics = run_guidance_epoch(
            model,
            train_loader,
            device,
            float(train_cfg["k_z"]),
            optimizer,
            float(train_cfg["gradient_clip"]),
        )
        val_metrics = run_guidance_epoch(
            model, val_loader, device, float(train_cfg["k_z"])
        )
        scheduler.step(val_metrics["loss"])
        print(json.dumps({"stage": "guidance", "epoch": epoch, "train": train_metrics, "val": val_metrics}))
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            stale = 0
            torch.save(
                checkpoint_payload(config, model, None, epoch, val_metrics),
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= patience:
                break
    state = load_checkpoint(checkpoint_path)
    model.load_state_dict(state["guidance_model"], strict=True)
    return checkpoint_path


def train_joint(
    config: dict,
    guidance_model: torch.nn.Module,
    diffusion: TMDMDiffusion,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    run_dir: Path,
) -> Path:
    train_cfg = config["train"]
    optimizer = AdamW(
        list(guidance_model.parameters()) + list(diffusion.parameters()),
        lr=float(train_cfg["joint_learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    patience = int(train_cfg["patience"])
    best = float("inf")
    stale = 0
    checkpoint_path = run_dir / "best_baseline2.pt"
    for epoch in range(1, int(train_cfg["diffusion_epochs"]) + 1):
        train_metrics = run_joint_epoch(
            guidance_model,
            diffusion,
            train_loader,
            device,
            float(train_cfg["k_cond"]),
            float(train_cfg["k_z"]),
            optimizer,
            float(train_cfg["gradient_clip"]),
        )
        val_metrics = run_joint_epoch(
            guidance_model,
            diffusion,
            val_loader,
            device,
            float(train_cfg["k_cond"]),
            float(train_cfg["k_z"]),
        )
        scheduler.step(val_metrics["loss"])
        print(json.dumps({"stage": "joint", "epoch": epoch, "train": train_metrics, "val": val_metrics}))
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            stale = 0
            torch.save(
                checkpoint_payload(
                    config, guidance_model, diffusion, epoch, val_metrics
                ),
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= patience:
                break
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TMDM-Cond Baseline 2.")
    parser.add_argument("--config", default="baseline2/configs/heew.yaml")
    parser.add_argument("--stage", choices=["all", "guidance", "diffusion"], default="all")
    parser.add_argument("--guidance-checkpoint", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    config = load_config(args.config, args.overrides)
    seed = int(config["run"]["seed"])
    seed_everything(seed)
    device = choose_device(config["run"].get("device", "auto"))
    data_cfg = config["data"]
    train_dataset = HEEWConditionDataset(
        resolve_path(data_cfg["energy_path"]),
        resolve_path(data_cfg["weather_path"]),
        "train",
        data_cfg["weather_feature_set"],
    )
    val_dataset = HEEWConditionDataset(
        resolve_path(data_cfg["energy_path"]),
        resolve_path(data_cfg["weather_path"]),
        "val",
        data_cfg["weather_feature_set"],
    )
    train_loader = make_loader(
        train_dataset,
        int(config["train"]["batch_size"]),
        True,
        int(config["run"]["num_workers"]),
        seed,
    )
    val_loader = make_loader(
        val_dataset,
        int(config["train"]["batch_size"]),
        False,
        int(config["run"]["num_workers"]),
        seed,
    )
    run_dir = resolve_path(config["run"]["output_root"]) / f"{config['run']['name']}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    guidance_model = build_guidance_model(
        config, train_dataset.condition_channels
    ).to(device)

    if args.guidance_checkpoint:
        state = load_checkpoint(args.guidance_checkpoint)
        guidance_model.load_state_dict(state["guidance_model"], strict=True)
    elif args.stage in {"all", "guidance"}:
        train_guidance(
            config, guidance_model, train_loader, val_loader, device, run_dir
        )
    elif (run_dir / "best_guidance.pt").exists():
        state = load_checkpoint(run_dir / "best_guidance.pt")
        guidance_model.load_state_dict(state["guidance_model"], strict=True)
    else:
        raise FileNotFoundError(
            "Diffusion training requires --guidance-checkpoint or "
            f"{run_dir / 'best_guidance.pt'}."
        )

    checkpoint_path: Path | None = None
    if args.stage in {"all", "diffusion"}:
        diffusion = build_diffusion(config).to(device)
        checkpoint_path = train_joint(
            config,
            guidance_model,
            diffusion,
            train_loader,
            val_loader,
            device,
            run_dir,
        )
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint_path) if checkpoint_path else None,
                "device": str(device),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

