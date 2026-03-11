# SGT MIDI Classification Pipeline

## Project Overview
This repository is a research-oriented symbolic music classification pipeline built on MIDI data. The core workflow combines:
- MIDI2vec graph embeddings from MIDI event networks
- SGT (Sequence Graph Transform) temporal features
- MLP classifiers with k-fold cross-validation
- Optional late fusion between MIDI2vec and SGT features

The pipeline order, caching, and reproducibility rules are enforced by a unified runner.

---

## Core Pipeline (Strict Order)
Unless stated otherwise, the pipeline runs in this exact sequence:

1. Generate MIDI graph edgelists
2. Train MIDI2vec embeddings
3. Generate XMIDI labels
4. Build piece-level vectors
5. Generate SGT features
6. Train and evaluate MLP (k-fold)

These steps are wired in `src/runners/run_experiment.py` as:
`edgelist -> embed -> labels -> piece_vectors -> sgt -> train`.

---

## Unified Runner and Configs
Entry point:

```bash
python src/runners/run_experiment.py --base configs/base.yaml --experiment configs/experiments/exp_data_500.yaml --resume
```

If your experiment config does not define a dataset, add `--dataset`:

```bash
python src/runners/run_experiment.py --base configs/base.yaml --dataset configs/datasets/data.yaml --experiment configs/experiments/exp_data_500.yaml --resume
```

Config merge order (later overrides earlier):
1) `configs/base.yaml`
2) `configs/datasets/*.yaml`
3) `configs/experiments/*.yaml`

Runner controls:
- `--resume`: skip steps whose outputs already exist
- `--skip`: comma-separated step names to skip
- `--force`: comma-separated step names to rerun even if cached

Caching and outputs:
- Artifacts are stored under `artifacts/{dataset_id}/{step}/{hash}/...`
- Results append to `results/results.jsonl` and `results/results.csv`
- Logs are written to `logs/`

For more detail, see `docs/EXPERIMENTS.md`.

---

## Requirements
- Python 3.x
- Node.js (required for MIDI2vec edgelist generation)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Key Directories
- `configs/`: base, dataset, and experiment configs that define the full pipeline
- `src/`: core scripts and step implementations
- `external/`: third-party tools, including `external/midi2vec` (Node.js + Python)
- `artifacts/`: cached step outputs by dataset and hash
- `results/`: aggregated experiment metrics
- `data/`: raw MIDI datasets (XMIDI format; do not regenerate automatically)
- `docs/`: extended guides and notes
- `legacy/`: older scripts kept for reference

---

## Manual Step-by-Step (Without the Runner)
Use these only if you are not running `src/runners/run_experiment.py`.

1) Generate MIDI graph edgelists (Node.js)

```bash
node external/midi2vec/midi2edgelist/index.js -i data/Data
```

2) Train MIDI2vec embeddings

```bash
python external/midi2vec/edgelist2vec/embed.py -i external/midi2vec/edgelist -o artifacts/midi-embs/myset/embeddings.bin
```

3) Generate XMIDI labels

```bash
python src/make_XMIDI_labels.py
```

4) Build piece-level vectors

```bash
python src/piece_vectors.py
```

5) Generate SGT features

```bash
python src/sgt_features.py
```

Alternative SGT script with drum-aware tokens:

```bash
python src/sgt_features_drums.py
```

6) Train and evaluate MLP (k-fold)

```bash
python src/mlp_kfold.py
```

---

## Configuration Essentials
`configs/base.yaml` is the default source of truth and defines:
- Dataset defaults (`dataset.id`, `dataset.data_root`)
- All pipeline steps (`edgelist`, `embed`, `labels`, `piece_vectors`, `sgt`, `train`)
- Step parameters (e.g., `walk_length`, `token_mode`, `top_vocab`, `epochs`)

Experiment configs typically override only what changes per run
(experiment name, dataset id/path, and a few step parameters).

---

## Reproducibility Rules
- Fix random seeds whenever possible (`seed` in configs).
- Any preprocessing that depends on dataset statistics must be performed inside each fold to avoid leakage.
- Do not overwrite previous outputs unless explicitly requested.
