# TMDM — HEEW Condition-Only Baseline 2

This branch retains the original TMDM source under `TMDM/` and adds a clean
adaptation under `baseline2/` for comparison with the two-stage joint
source-load scenario generator.

> **Method label for papers:** `TMDM-Cond (Baseline 2)`
>
> This is a condition-only adaptation of TMDM, not an exact reproduction of
> the original history-based forecasting protocol.

The original method is *Transformer-Modulated Diffusion Models for
Probabilistic Multivariate Time Series Forecasting*. Its defining mechanism is
preserved here:

1. a variational Transformer produces a deterministic conditional forecast;
2. diffusion uses that forecast as both its conditioning signal and shifted
   terminal mean;
3. the timestep-modulated TMDM noise MLP jointly predicts all four channels;
4. the guidance Transformer and diffusion model are jointly fine-tuned after
   guidance pretraining.

## Why adaptation is required

Original TMDM is a forecasting model that receives historical target sequences.
The proposed `2-stages` method deliberately uses no historical energy. Giving
TMDM history would create a different information set, so `baseline2/` receives
only target-day weather, cyclical time features, and the PV-only year control.

## Fair-comparison contract

| Item | TMDM-Cond Baseline 2 |
|---|---|
| Targets | Electricity, Heat, Cooling, PV, jointly generated |
| Horizon | One complete 24-hour day |
| Train | 2014–2020 |
| Validation | 2021 |
| Test | 2022 |
| Normalization | Fitted on 2014–2020 only |
| Conditions | Same 10 weather + 8 cyclical time features |
| Historical energy | Not used |
| PV year control | PV guidance head only |
| Default scenarios | 100 |
| Diffusion | 100-step shifted-mean ancestral sampling |
| Seed | 42 |

Observed/reanalysis weather should be reported as an oracle-condition setting.
For operational day-ahead experiments, every method must receive the same
archived weather forecasts.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Place the cleaned files described in [Data/README.md](Data/README.md) under
`Data/`.

## Train

Pretrain the Transformer guidance and then jointly fine-tune guidance and
diffusion:

```bash
python -m baseline2.train --config baseline2/configs/heew.yaml --stage all
```

Configuration values can be overridden without editing the public YAML:

```bash
python -m baseline2.train \
  --config baseline2/configs/heew.yaml \
  --set train.guidance_epochs=2 \
  --set train.diffusion_epochs=2 \
  --set diffusion.timesteps=10
```

## Generate and evaluate scenarios

```bash
python -m baseline2.evaluate \
  --checkpoint experiments/baseline2/tmdm_condition_only_baseline2_seed42/best_baseline2.pt \
  --num-scenarios 100 \
  --output-dir evaluation_results/baseline2
```

The evaluator saves the same comparison-facing outputs as Baseline 1:

- `baseline2_scenarios.npz`, with scenarios shaped
  `[day, scenario, channel, hour]`;
- `metrics_summary.json` and `global_metrics.csv`;
- per-channel physical-space `RMSE`, `MAE`, `CRPS`, `nCRPS`, 90% coverage and interval width;
- per-channel normalized `RMSE_Z` and `MAE_Z`, using the 2014--2020 training mean/std;
- per-channel `Precision_Z` and `Recall_Z` (k-NN manifold scores, `k=5`, in the same z-score space);
- per-channel `CR` and `IW` for the central 95% interval; `CR` is coverage and `IW` remains in each channel's physical unit;
- `random_timeseries_50/`;
- `pearson/` plus `global_pearson.png`.

## Tests

```bash
pytest
```

Tests cover the fixed year split, train-only normalization, absence of historical
energy inputs, guidance shape, diffusion loss, and reverse-sampling shape.

## Attribution

Original TMDM repository and authors should be cited in any publication using
this adaptation. The upstream repository did not provide a license file; confirm
reuse and redistribution terms before publishing a derived release.
