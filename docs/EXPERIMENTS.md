# Experiment Framework Guide

This document describes how to run and reproduce experiments using the unified
runner and config system.

## 1. Quick Start

Run the full pipeline on the default dataset:

```
python src/runners/run_experiment.py --base configs/base.yaml --dataset configs/datasets/data.yaml --experiment configs/experiments/exp_default.yaml --resume
```

Use the smaller dataset:

```
python src/runners/run_experiment.py --base configs/base.yaml --dataset configs/datasets/data_100.yaml --experiment configs/experiments/exp_default.yaml --resume
```

## 2. Config Structure

Configs are merged in this order:

1) base.yaml
2) dataset config
3) experiment config

Later files override earlier values.

Top-level keys used:

- seed, deterministic_tf
- paths
- dataset
- steps
- experiment

### Example: base.yaml (abridged)

```
seed: 42
paths:
  artifacts_root: artifacts
  results_jsonl: results/results.jsonl
  results_csv: results/results.csv

dataset:
  id: data
  data_root: data/Data

steps:
  sgt:
    script: src/sgt_features.py
    alpha: 0.8
    max_dt_beats: 8
    token_mode: prog_pitch
    top_vocab: 64
  train:
    task: both
    n_splits: 5
```

### Dataset config

```
dataset:
  id: data_100
  data_root: data/Data_100
```

### Experiment config

```
experiment:
  name: default

steps:
  sgt:
    script: src/sgt_features.py
  train:
    task: both
```

## 3. Step Outputs and Caching

Each step output is stored under:

```
artifacts/{dataset_id}/{step}/{hash}/...
```

The hash is computed from:

- dataset config
- step config
- upstream dependency hashes (for dependent steps)

When `--resume` is used, steps with existing outputs are skipped.

## 4. Selective Rerun

Skip a step:

```
python src/runners/run_experiment.py --base configs/base.yaml --dataset configs/datasets/data.yaml --experiment configs/experiments/exp_default.yaml --skip embed,sgt
```

Force rerun:

```
python src/runners/run_experiment.py --base configs/base.yaml --dataset configs/datasets/data.yaml --experiment configs/experiments/exp_default.yaml --force sgt
```

## 5. Results Logging

The training step appends metrics to:

- results/results.jsonl
- results/results.csv

Each record includes:

- run_id
- dataset_id
- experiment name
- mode, task, seed
- mean and std of accuracy/F1

## 6. Notes for Reproducibility

- Always fix `seed` in configs.
- Keep dataset configs separate from experiment configs.
- If you change feature extraction parameters, the hash changes and caches are invalidated.
- For compute-heavy steps (e.g., embeddings), use `--resume` to avoid reruns.
