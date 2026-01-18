# Project Agent Guide (AGENTS.md)

## Project Overview
This project implements a symbolic music classification pipeline based on MIDI data.
The workflow combines:
- MIDI2vec (graph-based embeddings from MIDI event networks)
- SGT (Sequence Graph Transform) temporal features
- MLP classifiers with k-fold cross-validation
- Optional late fusion between MIDI2vec and SGT features

The codebase is research-oriented and used for reproducible experiments.

---

## Standard Execution Pipeline (IMPORTANT)
Unless explicitly stated otherwise, the **default execution order is strictly as follows**:

1. Generate MIDI graph edgelists (Node.js)

node external/midi2vec/midi2edgelist/index.js -i data/Data
Input: "data/Data/" or "data/Data_100/" (raw MIDI files, XMIDI format)

Output: external/midi2vec/edgelist/ (graph edge lists per MIDI)

2. Train MIDI2vec embeddings

python external/midi2vec/edgelist2vec/embed.py \
  -i external/midi2vec/edgelist \
  -o artifacts/midi-embs/myset/embeddings.bin
Input: external/midi2vec/edgelist/

Output: artifacts/midi-embs/myset/embeddings.bin

3. Generate XMIDI labels


python src/make_XMIDI_labels.py
Output: label files used by downstream classifiers

4. Export piece-level MIDI2vec vectors


python src/piece_vectors.py
Input: embeddings.bin + label files

Output: piece-level feature vectors (word2vec text)

5. Generate SGT features


python src/sgt_features.py
Input: raw MIDI data

Output: SGT temporal feature matrices

Note: use src/sgt_features_drums.py as an alternative SGT feature generator with drum-aware tokens.

6. MLP classification with k-fold cross-validation


python src/mlp_kfold.py
(or eval_mlp_kfold_fusion.py for fusion experiments)

## Experiment Framework (new)
Use the unified runner to manage configs, caching, and results logging:

python src/runners/run_experiment.py --base configs/base.yaml --dataset configs/datasets/data.yaml --experiment configs/experiments/exp_default.yaml --resume

Outputs and intermediate artifacts are stored under artifacts/{dataset_id}/...

## Directory Structure Guide
"data/Data/" or "data/Data_100/"
Raw MIDI dataset (XMIDI format).
⚠️ Large files. Do NOT modify or regenerate automatically.

external/midi2vec/
MIDI2vec source code and edgelist generation scripts.

artifacts/midi-embs/
Stored MIDI2vec embedding outputs (binary).

SGT-related scripts

src/sgt_features.py: generate temporal SGT features

piece_vectors.py: align embeddings to pieces

eval_mlp_kfold.py: MLP evaluation

eval_mlp_kfold_fusion.py: late fusion evaluation

## Reproducibility Rules
All experiments should use fixed random seeds when possible.

Any preprocessing that depends on dataset statistics (e.g. PCA, normalization,
vocabulary selection) must be performed inside each cross-validation fold
to avoid information leakage.

Outputs should not overwrite previous results unless explicitly requested.

## Restrictions for the Agent
❌ Do NOT modify raw datasets in data/

❌ Do NOT regenerate embeddings unless explicitly requested

❌ Do NOT delete or overwrite experiment outputs automatically

✅ Code refactoring, bug fixing, and modularization are allowed

✅ Suggest improvements, but ask before making large structural changes

## Communication Preference
Prefer minimal, correct, research-grade code

Explain changes in terms of experimental validity
