# mlp_kfold.py
# ------------------------------------------------------------------------------------
# A switchable evaluation script for XMIDI multi-task classification (Emotion + Genre)
# supporting:
#   - MIDI2vec only
#   - SGT only (with optional log1p + reducer)
#   - Early fusion (feature concatenation)
#   - Late fusion (probability-level fusion with automatic weight search on validation set)
#
# Key design goals:
#   1) No leakage: scaler/reducer are fit ONLY on inner-train inside each fold.
#   2) Robust validation: explicit stratified train/val split inside each outer fold.
#   3) Comparable outputs: per-fold metrics + overall summary + confusion matrices/reports.
#
# Recommended workflow:
#   - Start with EXPERIMENTS list below (emb_only / sgt_only / early_fusion / late_fusion)
#   - Then expand PCA_N_LIST / LOG1P_SGT_LIST / MONITORS to grid-search.
# ------------------------------------------------------------------------------------

import os
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers, callbacks, Model, Input

from lib.results import append_jsonl, append_csv


# ========= Paths =========
PIECE_VEC_BIN = r"artifacts/midi-embs/myset/piece_vectors.bin"
LABELS_CSV    = r"artifacts/midi-embs/myset/labels.csv"

SGT_FEATURES_NPY = r"artifacts/midi-embs/myset/sgt_features.npy"
SGT_IDS_NPY      = r"artifacts/midi-embs/myset/sgt_ids.npy"  # optional; required if SGT_FEATURES_NPY is a matrix
# =========================

# ========= CV / Training =========
N_SPLITS    = 5
EPOCHS      = 50
BATCH_SIZE  = 32
LR          = 1e-3
SEED        = 42
VAL_SPLIT   = 0.1   # inner split inside each fold (stratified)
# ================================

# ========= MLP Architecture =========
TRUNK_1      = 256
TRUNK_2      = 128
DROP_TRUNK   = 0.3
HEAD_HIDDEN  = 64
DROP_HEAD    = 0.2
# ================================

# ========= Task Selection =========
# "emotion" | "genre" | "both"
TASK = "both"
# ================================

# ========= Results Output =========
RESULTS_JSONL = None
RESULTS_CSV = None
RUN_ID = None
DATASET_ID = None
# ================================

# ========= Switches / Search Space =========
PRINT_CONFUSION_MATRICES = True
PRINT_CLASSIFICATION_REPORTS = True

# Reducer for SGT ("pca" or "svd"). SVD is often more stable for sparse-ish count features.
SGT_REDUCER = "pca"

# Grid lists (used only if you add multiple experiments below)
PCA_N_LIST = [32, 64]       # try smaller first; 32/64 often helps if SGT is noisy | e.g., [32, 64, 128]
LOG1P_SGT_LIST = [True]   # log1p can help long-tailed count features | e.g., [True, False]
MONITORS = ["val_loss", "val_genre_accuracy"]  # for early stopping / checkpoint selection | e.g., ["val_loss", "val_genre_accuracy", "val_emotion_accuracy"]

# If True, build experiments by grid search (can be a lot of runs).
AUTO_GRID = False
# Modes to include in grid (MIDI2vec_only will always be included).
GRID_MODES = ["sgt_only", "early_fusion", "late_fusion"]
GRID_REDUCER_N_LIST = PCA_N_LIST
GRID_LOG1P_LIST = LOG1P_SGT_LIST
GRID_MONITORS = MONITORS


# For late fusion: search weight w in p = w*p_emb + (1-w)*p_sgt
W_GRID = np.round(np.linspace(0.0, 1.0, 21), 2).tolist()
WEIGHT_METRIC = "macro_f1"  # "macro_f1" or "accuracy"
# ===========================================


@dataclass
class Experiment:
    name: str
    mode: str  # "emb_only" | "sgt_only" | "early_fusion" | "late_fusion"
    log1p_sgt: bool = True
    reducer: str = "pca"  # "pca" | "svd" | "none"
    reducer_n: Optional[int] = 64
    monitor: str = "val_loss"  # early stopping monitor
    patience: int = 5


# Default experiments (edit/extend as needed)
DEFAULT_EXPERIMENTS: List[Experiment] = [
    Experiment(name="MIDI2vec_only", mode="emb_only", monitor="val_loss"),
    Experiment(name="SGT_only", mode="sgt_only", log1p_sgt=True, reducer=SGT_REDUCER, reducer_n=64, monitor="val_loss"),
    Experiment(name="EarlyFusion", mode="early_fusion", log1p_sgt=True, reducer=SGT_REDUCER, reducer_n=64, monitor="val_loss"),
    Experiment(name="LateFusion_autoW", mode="late_fusion", log1p_sgt=True, reducer=SGT_REDUCER, reducer_n=64, monitor="val_loss"),
]

def build_experiments() -> List[Experiment]:
    if not AUTO_GRID:
        return DEFAULT_EXPERIMENTS

    exps: List[Experiment] = [Experiment(name="MIDI2vec_only", mode="emb_only", monitor="val_loss")]
    grid_monitors = GRID_MONITORS
    if TASK != "both":
        grid_monitors = ["val_loss", "val_accuracy"]
    for mode in GRID_MODES:
        for log1p in GRID_LOG1P_LIST:
            for n in GRID_REDUCER_N_LIST:
                for mon in grid_monitors:
                    name = f"{mode}_log1p{int(log1p)}_{SGT_REDUCER}{n}_{mon}"
                    exps.append(Experiment(
                        name=name,
                        mode=mode,
                        log1p_sgt=log1p,
                        reducer=SGT_REDUCER,
                        reducer_n=int(n),
                        monitor=mon,
                    ))
    return exps

def set_seed(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_monitor_mode(monitor: str) -> str:
    if "accuracy" in monitor.lower():
        return "max"
    elif "loss" in monitor.lower():
        return "min"
    else:
        raise ValueError(f"unable to infer monitor mode from: {monitor}")


def _norm_id(x: str) -> str:
    s = str(x).strip().replace("\\", "/").split("/")[-1]
    s = os.path.splitext(s)[0]
    return s.lower()


def load_piece_vectors(vec_path: str) -> Dict[str, np.ndarray]:
    kv = KeyedVectors.load_word2vec_format(vec_path, binary=False)
    out: Dict[str, np.ndarray] = {}
    for k in kv.key_to_index.keys():
        out[_norm_id(k)] = kv.get_vector(k)
    return out


def load_sgt_features(features_path: str, ids_path: Optional[str] = None) -> Dict[str, np.ndarray]:
    obj = np.load(features_path, allow_pickle=True)

    # Case A: dict saved with .item()
    if getattr(obj, "dtype", None) == object:
        maybe = obj.item()
        if isinstance(maybe, dict):
            out: Dict[str, np.ndarray] = {}
            for k, v in maybe.items():
                out[_norm_id(k)] = np.asarray(v)
            return out

    # Case B: matrix + ids
    if ids_path is None:
        raise ValueError("SGT_FEATURES_NPY is a matrix, but SGT_IDS_NPY is not provided.")

    feats = np.asarray(obj)
    ids = np.load(ids_path, allow_pickle=True)
    if len(ids) != feats.shape[0]:
        raise ValueError(f"SGT ids length {len(ids)} != features rows {feats.shape[0]}")

    out: Dict[str, np.ndarray] = {}
    for i, pid in enumerate(ids):
        out[_norm_id(pid)] = feats[i]
    return out


def build_multitask_mlp(input_dim: int, n_emotion: int, n_genre: int) -> Model:
    inp = Input(shape=(input_dim,), name="x")

    x = layers.Dense(TRUNK_1, activation="relu")(inp)
    x = layers.Dropout(DROP_TRUNK)(x)
    x = layers.Dense(TRUNK_2, activation="relu")(x)
    x = layers.Dropout(DROP_TRUNK)(x)

    emo = layers.Dense(HEAD_HIDDEN, activation="relu")(x)
    emo = layers.Dropout(DROP_HEAD)(emo)
    out_emo = layers.Dense(n_emotion, activation="softmax", name="emotion")(emo)

    gen = layers.Dense(HEAD_HIDDEN, activation="relu")(x)
    gen = layers.Dropout(DROP_HEAD)(gen)
    out_gen = layers.Dense(n_genre, activation="softmax", name="genre")(gen)

    model = Model(inputs=inp, outputs=[out_emo, out_gen], name="mtl_mlp")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss={"emotion": "sparse_categorical_crossentropy", "genre": "sparse_categorical_crossentropy"},
        metrics={"emotion": ["accuracy"], "genre": ["accuracy"]},
    )
    return model


def build_single_task_mlp(input_dim: int, n_classes: int, task: str) -> Model:
    if task not in ("emotion", "genre"):
        raise ValueError("task must be 'emotion' or 'genre'")

    inp = Input(shape=(input_dim,), name="x")

    x = layers.Dense(TRUNK_1, activation="relu")(inp)
    x = layers.Dropout(DROP_TRUNK)(x)
    x = layers.Dense(TRUNK_2, activation="relu")(x)
    x = layers.Dropout(DROP_TRUNK)(x)

    head = layers.Dense(HEAD_HIDDEN, activation="relu")(x)
    head = layers.Dropout(DROP_HEAD)(head)
    out = layers.Dense(n_classes, activation="softmax", name=task)(head)

    model = Model(inputs=inp, outputs=out, name=f"stl_mlp_{task}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _fit_transform_sgt(
    Xs_tr_raw: np.ndarray,
    Xs_val_raw: np.ndarray,
    Xs_te_raw: np.ndarray,
    log1p: bool,
    reducer: str,
    reducer_n: Optional[int],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # log1p first (good for long-tailed counts)
    if log1p:
        Xs_tr_raw = np.log1p(Xs_tr_raw)
        Xs_val_raw = np.log1p(Xs_val_raw)
        Xs_te_raw = np.log1p(Xs_te_raw)

    # scale
    sc = StandardScaler()
    Xs_tr = sc.fit_transform(Xs_tr_raw)
    Xs_val = sc.transform(Xs_val_raw)
    Xs_te = sc.transform(Xs_te_raw)

    # reduce
    if reducer is None or reducer == "none" or reducer_n is None:
        return Xs_tr, Xs_val, Xs_te

    n_comp = int(min(reducer_n, Xs_tr.shape[1]))
    if reducer == "pca":
        red = PCA(n_components=n_comp, random_state=seed)
        Xs_tr = red.fit_transform(Xs_tr)
        Xs_val = red.transform(Xs_val)
        Xs_te = red.transform(Xs_te)
        return Xs_tr, Xs_val, Xs_te

    if reducer == "svd":
        # TruncatedSVD is often more robust for sparse-ish features
        red = TruncatedSVD(n_components=n_comp, random_state=seed)
        Xs_tr = red.fit_transform(Xs_tr)
        Xs_val = red.transform(Xs_val)
        Xs_te = red.transform(Xs_te)
        return Xs_tr, Xs_val, Xs_te

    raise ValueError(f"Unknown reducer: {reducer}")


def _fit_transform_emb(
    Xe_tr_raw: np.ndarray,
    Xe_val_raw: np.ndarray,
    Xe_te_raw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sc = StandardScaler()
    Xe_tr = sc.fit_transform(Xe_tr_raw)
    Xe_val = sc.transform(Xe_val_raw)
    Xe_te = sc.transform(Xe_te_raw)
    return Xe_tr, Xe_val, Xe_te


def _metric_score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro"))
    raise ValueError(f"Unknown metric: {metric}")


def _best_weight(
    p_a: np.ndarray,
    p_b: np.ndarray,
    y_true: np.ndarray,
    metric: str,
    w_grid: List[float],
) -> Tuple[float, float]:
    best_w = 0.5
    best_s = -1.0
    for w in w_grid:
        p = w * p_a + (1.0 - w) * p_b
        y_pred = np.argmax(p, axis=1)
        s = _metric_score(y_true, y_pred, metric)
        if s > best_s:
            best_s = s
            best_w = w
    return best_w, best_s


def _train_one(
    X_tr: np.ndarray,
    yemo_tr: np.ndarray,
    ygen_tr: np.ndarray,
    X_val: np.ndarray,
    yemo_val: np.ndarray,
    ygen_val: np.ndarray,
    n_emotion: int,
    n_genre: int,
    monitor: str,
    patience: int,
) -> Model:
    model = build_multitask_mlp(input_dim=X_tr.shape[1], n_emotion=n_emotion, n_genre=n_genre)

    monitor_mode = get_monitor_mode(monitor)
    es = callbacks.EarlyStopping(
        monitor=monitor,
        mode=monitor_mode,
        patience=patience,
        restore_best_weights=True,
        verbose=0,
    )

    model.fit(
        X_tr,
        {"emotion": yemo_tr, "genre": ygen_tr},
        validation_data=(X_val, {"emotion": yemo_val, "genre": ygen_val}),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[es],
    )
    return model


def _train_one_single(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    task: str,
    monitor: str,
    patience: int,
) -> Model:
    model = build_single_task_mlp(input_dim=X_tr.shape[1], n_classes=n_classes, task=task)

    monitor_mode = get_monitor_mode(monitor)
    es = callbacks.EarlyStopping(
        monitor=monitor,
        mode=monitor_mode,
        patience=patience,
        restore_best_weights=True,
        verbose=0,
    )

    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[es],
    )
    return model


def _normalize_monitor(monitor: str, task: str) -> str:
    if task == "both":
        return monitor
    if monitor in ("val_loss", "val_accuracy"):
        return monitor
    if monitor in ("val_emotion_accuracy", "val_genre_accuracy"):
        return "val_accuracy"
    print(f"[WARN] monitor '{monitor}' incompatible with TASK='{task}', fallback to val_loss")
    return "val_loss"


def _write_results(record: Dict[str, Any]) -> None:
    if RESULTS_JSONL:
        append_jsonl(RESULTS_JSONL, record)
    if RESULTS_CSV:
        keys = list(record.keys())
        append_csv(RESULTS_CSV, record, keys)


def run_experiment(
    exp: Experiment,
    X_emb: np.ndarray,
    X_sgt: np.ndarray,
    y_emo: np.ndarray,
    y_gen: np.ndarray,
    y_pair: np.ndarray,
    le_emo: LabelEncoder,
    le_gen: LabelEncoder,
) -> None:
    if TASK not in ("emotion", "genre", "both"):
        raise ValueError("TASK must be 'emotion', 'genre', or 'both'")

    exp_monitor = _normalize_monitor(exp.monitor, TASK)

    print("\n" + "=" * 90)
    print(f"Experiment: {exp.name}")
    print(f"  mode={exp.mode} | task={TASK} | log1p_sgt={exp.log1p_sgt} | reducer={exp.reducer}({exp.reducer_n}) | monitor={exp_monitor}")
    print("=" * 90)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    if TASK == "both":
        emo_accs, emo_f1s = [], []
        gen_accs, gen_f1s = [], []
        ov_accs, ov_f1s = [], []

        emo_true_all, emo_pred_all = [], []
        gen_true_all, gen_pred_all = [], []
    else:
        task_accs, task_f1s = [], []
        task_true_all, task_pred_all = [], []

    fold = 0
    y_split = y_pair if TASK == "both" else (y_emo if TASK == "emotion" else y_gen)
    split_name = "y_pair" if TASK == "both" else ("y_emo" if TASK == "emotion" else "y_gen")
    print(f"[OK] TASK={TASK} | stratify_label={split_name}")

    for train_idx, test_idx in skf.split(X_emb, y_split):
        fold += 1

        # inner train/val split (stratified)
        y_split_tr = y_split[train_idx]
        try:
            tr_idx, val_idx = train_test_split(
                train_idx,
                test_size=VAL_SPLIT,
                random_state=SEED + fold,
                stratify=y_split_tr,
            )
        except ValueError:
            if TASK == "both":
                tr_idx, val_idx = train_test_split(
                    train_idx,
                    test_size=VAL_SPLIT,
                    random_state=SEED + fold,
                    stratify=y_emo[train_idx],
                )
            else:
                n_classes = len(np.unique(y_split_tr))
                val_size = int(round(VAL_SPLIT * len(train_idx)))
                val_size = max(n_classes, val_size)
                max_val = len(train_idx) - n_classes
                if val_size > max_val:
                    val_size = max_val
                if val_size < n_classes or val_size <= 0:
                    print("[WARN] cannot stratify single-task split; fallback to random split")
                    tr_idx, val_idx = train_test_split(
                        train_idx,
                        test_size=VAL_SPLIT,
                        random_state=SEED + fold,
                        stratify=None,
                    )
                else:
                    tr_idx, val_idx = train_test_split(
                        train_idx,
                        test_size=val_size,
                        random_state=SEED + fold,
                        stratify=y_split_tr,
                    )

        # slice labels
        yemo_tr, yemo_val, yemo_te = y_emo[tr_idx], y_emo[val_idx], y_emo[test_idx]
        ygen_tr, ygen_val, ygen_te = y_gen[tr_idx], y_gen[val_idx], y_gen[test_idx]

        # prepare inputs per mode
        # Emb pipeline
        Xe_tr, Xe_val, Xe_te = _fit_transform_emb(X_emb[tr_idx], X_emb[val_idx], X_emb[test_idx])

        # SGT pipeline (log1p + scaler + reducer)
        Xs_tr, Xs_val, Xs_te = _fit_transform_sgt(
            X_sgt[tr_idx], X_sgt[val_idx], X_sgt[test_idx],
            log1p=exp.log1p_sgt,
            reducer=exp.reducer,
            reducer_n=exp.reducer_n,
            seed=SEED,
        )

        n_emotion = len(le_emo.classes_)
        n_genre = len(le_gen.classes_)

        if exp.mode == "emb_only":
            if TASK == "both":
                model = _train_one(
                    Xe_tr, yemo_tr, ygen_tr,
                    Xe_val, yemo_val, ygen_val,
                    n_emotion, n_genre,
                    monitor=exp_monitor,
                    patience=exp.patience,
                )
                p_emo, p_gen = model.predict(Xe_te, verbose=0)
            else:
                task = TASK
                y_tr = yemo_tr if task == "emotion" else ygen_tr
                y_val = yemo_val if task == "emotion" else ygen_val
                n_cls = n_emotion if task == "emotion" else n_genre
                model = _train_one_single(
                    Xe_tr, y_tr, Xe_val, y_val, n_cls, task,
                    monitor=exp_monitor,
                    patience=exp.patience,
                )
                p_task = model.predict(Xe_te, verbose=0)

        elif exp.mode == "sgt_only":
            if TASK == "both":
                model = _train_one(
                    Xs_tr, yemo_tr, ygen_tr,
                    Xs_val, yemo_val, ygen_val,
                    n_emotion, n_genre,
                    monitor=exp_monitor,
                    patience=exp.patience,
                )
                p_emo, p_gen = model.predict(Xs_te, verbose=0)
            else:
                task = TASK
                y_tr = yemo_tr if task == "emotion" else ygen_tr
                y_val = yemo_val if task == "emotion" else ygen_val
                n_cls = n_emotion if task == "emotion" else n_genre
                model = _train_one_single(
                    Xs_tr, y_tr, Xs_val, y_val, n_cls, task,
                    monitor=exp_monitor,
                    patience=exp.patience,
                )
                p_task = model.predict(Xs_te, verbose=0)

        elif exp.mode == "early_fusion":
            X_tr = np.hstack([Xe_tr, Xs_tr])
            X_val = np.hstack([Xe_val, Xs_val])
            X_te = np.hstack([Xe_te, Xs_te])

            if TASK == "both":
                model = _train_one(
                    X_tr, yemo_tr, ygen_tr,
                    X_val, yemo_val, ygen_val,
                    n_emotion, n_genre,
                    monitor=exp_monitor,
                    patience=exp.patience,
                )
                p_emo, p_gen = model.predict(X_te, verbose=0)
            else:
                task = TASK
                y_tr = yemo_tr if task == "emotion" else ygen_tr
                y_val = yemo_val if task == "emotion" else ygen_val
                n_cls = n_emotion if task == "emotion" else n_genre
                model = _train_one_single(
                    X_tr, y_tr, X_val, y_val, n_cls, task,
                    monitor=exp_monitor,
                    patience=exp.patience,
                )
                p_task = model.predict(X_te, verbose=0)

        elif exp.mode == "late_fusion":
            # Train emb-only model
            if TASK == "both":
                model_emb = _train_one(
                    Xe_tr, yemo_tr, ygen_tr,
                    Xe_val, yemo_val, ygen_val,
                    n_emotion, n_genre,
                    monitor=exp_monitor,
                    patience=exp.patience,
                )
                p_emo_emb_val, p_gen_emb_val = model_emb.predict(Xe_val, verbose=0)
                p_emo_emb_te, p_gen_emb_te = model_emb.predict(Xe_te, verbose=0)

                # Train sgt-only model
                model_sgt = _train_one(
                    Xs_tr, yemo_tr, ygen_tr,
                    Xs_val, yemo_val, ygen_val,
                    n_emotion, n_genre,
                    monitor=exp_monitor,
                    patience=exp.patience,
                )
                p_emo_sgt_val, p_gen_sgt_val = model_sgt.predict(Xs_val, verbose=0)
                p_emo_sgt_te, p_gen_sgt_te = model_sgt.predict(Xs_te, verbose=0)

                # Search weights on validation set (separately for each head)
                w_emo, s_emo = _best_weight(p_emo_emb_val, p_emo_sgt_val, yemo_val, WEIGHT_METRIC, W_GRID)
                w_gen, s_gen = _best_weight(p_gen_emb_val, p_gen_sgt_val, ygen_val, WEIGHT_METRIC, W_GRID)

                # Fuse on test
                p_emo = w_emo * p_emo_emb_te + (1.0 - w_emo) * p_emo_sgt_te
                p_gen = w_gen * p_gen_emb_te + (1.0 - w_gen) * p_gen_sgt_te
            else:
                task = TASK
                y_tr = yemo_tr if task == "emotion" else ygen_tr
                y_val = yemo_val if task == "emotion" else ygen_val
                n_cls = n_emotion if task == "emotion" else n_genre

                model_emb = _train_one_single(
                    Xe_tr, y_tr, Xe_val, y_val, n_cls, task,
                    monitor=exp_monitor,
                    patience=exp.patience,
                )
                p_emb_val = model_emb.predict(Xe_val, verbose=0)
                p_emb_te = model_emb.predict(Xe_te, verbose=0)

                model_sgt = _train_one_single(
                    Xs_tr, y_tr, Xs_val, y_val, n_cls, task,
                    monitor=exp_monitor,
                    patience=exp.patience,
                )
                p_sgt_val = model_sgt.predict(Xs_val, verbose=0)
                p_sgt_te = model_sgt.predict(Xs_te, verbose=0)

                w_task, s_task = _best_weight(p_emb_val, p_sgt_val, y_val, WEIGHT_METRIC, W_GRID)
                p_task = w_task * p_emb_te + (1.0 - w_task) * p_sgt_te

        else:
            raise ValueError(f"Unknown mode: {exp.mode}")

        if TASK == "both":
            pred_emo = np.argmax(p_emo, axis=1)
            pred_gen = np.argmax(p_gen, axis=1)

            emo_acc = accuracy_score(yemo_te, pred_emo)
            emo_f1 = f1_score(yemo_te, pred_emo, average="macro")
            gen_acc = accuracy_score(ygen_te, pred_gen)
            gen_f1 = f1_score(ygen_te, pred_gen, average="macro")

            ov_acc = 0.5 * (emo_acc + gen_acc)
            ov_f1 = 0.5 * (emo_f1 + gen_f1)

            emo_accs.append(emo_acc); emo_f1s.append(emo_f1)
            gen_accs.append(gen_acc); gen_f1s.append(gen_f1)
            ov_accs.append(ov_acc); ov_f1s.append(ov_f1)

            emo_true_all.extend(yemo_te.tolist())
            emo_pred_all.extend(pred_emo.tolist())
            gen_true_all.extend(ygen_te.tolist())
            gen_pred_all.extend(pred_gen.tolist())

            if exp.mode == "late_fusion":
                print(
                    f"[Fold {fold:02d}] Emotion acc={emo_acc:.4f} F1={emo_f1:.4f} | "
                    f"Genre acc={gen_acc:.4f} F1={gen_f1:.4f} | "
                    f"Overall acc={ov_acc:.4f} F1={ov_f1:.4f} | "
                    f"w_emo={w_emo:.2f} w_gen={w_gen:.2f}"
                )
            else:
                print(
                    f"[Fold {fold:02d}] Emotion acc={emo_acc:.4f} F1={emo_f1:.4f} | "
                    f"Genre acc={gen_acc:.4f} F1={gen_f1:.4f} | "
                    f"Overall acc={ov_acc:.4f} F1={ov_f1:.4f}"
                )
        else:
            pred_task = np.argmax(p_task, axis=1)
            y_te = yemo_te if TASK == "emotion" else ygen_te

            task_acc = accuracy_score(y_te, pred_task)
            task_f1 = f1_score(y_te, pred_task, average="macro")

            task_accs.append(task_acc); task_f1s.append(task_f1)
            task_true_all.extend(y_te.tolist())
            task_pred_all.extend(pred_task.tolist())

            if exp.mode == "late_fusion":
                print(
                    f"[Fold {fold:02d}] {TASK} acc={task_acc:.4f} F1={task_f1:.4f} | "
                    f"w_{TASK}={w_task:.2f}"
                )
            else:
                print(
                    f"[Fold {fold:02d}] {TASK} acc={task_acc:.4f} F1={task_f1:.4f}"
                )

    def mean_std(a: List[float]) -> Tuple[float, float]:
        return float(np.mean(a)), float(np.std(a))

    title = "==== CV Results (" + exp.name + ") ===="
    print("\n" + title)

    if TASK == "both":
        emo_acc_m, emo_acc_s = mean_std(emo_accs)
        emo_f1_m, emo_f1_s = mean_std(emo_f1s)
        gen_acc_m, gen_acc_s = mean_std(gen_accs)
        gen_f1_m, gen_f1_s = mean_std(gen_f1s)
        ov_acc_m, ov_acc_s = mean_std(ov_accs)
        ov_f1_m, ov_f1_s = mean_std(ov_f1s)

        print(f"Emotion Accuracy : {emo_acc_m:.4f} ± {emo_acc_s:.4f}")
        print(f"Emotion Macro-F1 : {emo_f1_m:.4f} ± {emo_f1_s:.4f}")
        print(f"Genre   Accuracy : {gen_acc_m:.4f} ± {gen_acc_s:.4f}")
        print(f"Genre   Macro-F1 : {gen_f1_m:.4f} ± {gen_f1_s:.4f}")
        print(f"Overall Accuracy : {ov_acc_m:.4f} ± {ov_acc_s:.4f}")
        print(f"Overall Macro-F1 : {ov_f1_m:.4f} ± {ov_f1_s:.4f}")

        if PRINT_CONFUSION_MATRICES:
            emo_cm = confusion_matrix(emo_true_all, emo_pred_all, normalize="true")
            gen_cm = confusion_matrix(gen_true_all, gen_pred_all, normalize="true")

            print("\n==== Emotion Confusion Matrix (row-normalized) ====")
            print(np.array_str(emo_cm, precision=3, suppress_small=True))

            print("\n==== Genre Confusion Matrix (row-normalized) ====")
            print(np.array_str(gen_cm, precision=3, suppress_small=True))

        if PRINT_CLASSIFICATION_REPORTS:
            print("\n==== Emotion Classification Report ====")
            print(classification_report(emo_true_all, emo_pred_all, target_names=le_emo.classes_))
            print("\n==== Genre Classification Report ====")
            print(classification_report(gen_true_all, gen_pred_all, target_names=le_gen.classes_))

        record = {
            "run_id": RUN_ID,
            "dataset_id": DATASET_ID,
            "experiment": exp.name,
            "mode": exp.mode,
            "task": TASK,
            "seed": SEED,
            "emo_acc_mean": emo_acc_m,
            "emo_acc_std": emo_acc_s,
            "emo_f1_mean": emo_f1_m,
            "emo_f1_std": emo_f1_s,
            "gen_acc_mean": gen_acc_m,
            "gen_acc_std": gen_acc_s,
            "gen_f1_mean": gen_f1_m,
            "gen_f1_std": gen_f1_s,
            "overall_acc_mean": ov_acc_m,
            "overall_acc_std": ov_acc_s,
            "overall_f1_mean": ov_f1_m,
            "overall_f1_std": ov_f1_s,
        }
        _write_results(record)
    else:
        task_acc_m, task_acc_s = mean_std(task_accs)
        task_f1_m, task_f1_s = mean_std(task_f1s)
        task_name = "Emotion" if TASK == "emotion" else "Genre"
        target_names = le_emo.classes_ if TASK == "emotion" else le_gen.classes_

        print(f"{task_name} Accuracy : {task_acc_m:.4f} ± {task_acc_s:.4f}")
        print(f"{task_name} Macro-F1 : {task_f1_m:.4f} ± {task_f1_s:.4f}")

        if PRINT_CONFUSION_MATRICES:
            cm = confusion_matrix(task_true_all, task_pred_all, normalize="true")
            print(f"\n==== {task_name} Confusion Matrix (row-normalized) ====")
            print(np.array_str(cm, precision=3, suppress_small=True))

        if PRINT_CLASSIFICATION_REPORTS:
            print(f"\n==== {task_name} Classification Report ====")
            labels = list(range(len(target_names)))
            print(classification_report(task_true_all, task_pred_all, labels=labels, target_names=target_names))

        record = {
            "run_id": RUN_ID,
            "dataset_id": DATASET_ID,
            "experiment": exp.name,
            "mode": exp.mode,
            "task": TASK,
            "seed": SEED,
            "acc_mean": task_acc_m,
            "acc_std": task_acc_s,
            "f1_mean": task_f1_m,
            "f1_std": task_f1_s,
        }
        _write_results(record)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piece-vec", default=PIECE_VEC_BIN)
    ap.add_argument("--labels", default=LABELS_CSV)
    ap.add_argument("--sgt-feat", default=SGT_FEATURES_NPY)
    ap.add_argument("--sgt-ids", default=SGT_IDS_NPY)
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--n-splits", type=int, default=N_SPLITS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--val-split", type=float, default=VAL_SPLIT)
    ap.add_argument("--results-jsonl", default=None)
    ap.add_argument("--results-csv", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dataset-id", default=None)
    args = ap.parse_args()

    globals()["PIECE_VEC_BIN"] = args.piece_vec
    globals()["LABELS_CSV"] = args.labels
    globals()["SGT_FEATURES_NPY"] = args.sgt_feat
    globals()["SGT_IDS_NPY"] = args.sgt_ids
    globals()["TASK"] = args.task
    globals()["N_SPLITS"] = args.n_splits
    globals()["EPOCHS"] = args.epochs
    globals()["BATCH_SIZE"] = args.batch_size
    globals()["LR"] = args.lr
    globals()["SEED"] = args.seed
    globals()["VAL_SPLIT"] = args.val_split
    globals()["RESULTS_JSONL"] = args.results_jsonl
    globals()["RESULTS_CSV"] = args.results_csv
    globals()["RUN_ID"] = args.run_id
    globals()["DATASET_ID"] = args.dataset_id

    set_seed(SEED)

    # load labels
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(f"labels.csv not found: {LABELS_CSV}")
    df = pd.read_csv(LABELS_CSV)
    if not {"piece", "emotion", "genre"}.issubset(df.columns):
        raise ValueError("labels.csv must contain columns: piece, emotion, genre")

    # load embeddings + sgt
    if not os.path.exists(PIECE_VEC_BIN):
        raise FileNotFoundError(f"piece_vectors.bin not found: {PIECE_VEC_BIN}")
    if not os.path.exists(SGT_FEATURES_NPY):
        raise FileNotFoundError(f"sgt_features.npy not found: {SGT_FEATURES_NPY}")

    emb_dict = load_piece_vectors(PIECE_VEC_BIN)
    sgt_ids_path = SGT_IDS_NPY if SGT_IDS_NPY and os.path.exists(SGT_IDS_NPY) else None
    sgt_dict = load_sgt_features(SGT_FEATURES_NPY, sgt_ids_path)

    # align pieces: keep only those in BOTH
    ids: List[str] = []
    emo_text: List[str] = []
    gen_text: List[str] = []
    Xe_list: List[np.ndarray] = []
    Xs_list: List[np.ndarray] = []

    for _, r in df.iterrows():
        pid = _norm_id(r["piece"])
        if pid in emb_dict and pid in sgt_dict:
            ids.append(pid)
            emo_text.append(str(r["emotion"]))
            gen_text.append(str(r["genre"]))
            Xe_list.append(emb_dict[pid])
            Xs_list.append(sgt_dict[pid])

    X_emb = np.vstack(Xe_list).astype(np.float32)
    X_sgt = np.vstack(Xs_list).astype(np.float32)
    y_emo_text = np.array(emo_text, dtype=object)
    y_gen_text = np.array(gen_text, dtype=object)

    le_emo = LabelEncoder()
    le_gen = LabelEncoder()
    y_emo = le_emo.fit_transform(y_emo_text)
    y_gen = le_gen.fit_transform(y_gen_text)

    # pair stratification label
    pair_text = np.array([f"{e}__{g}" for e, g in zip(y_emo_text, y_gen_text)], dtype=object)
    le_pair = LabelEncoder()
    y_pair = le_pair.fit_transform(pair_text)

    print(f"[OK] samples: {X_emb.shape[0]}, d_emb: {X_emb.shape[1]}, d_sgt_raw: {X_sgt.shape[1]}")
    print(f"[OK] emotion_classes: {len(le_emo.classes_)}, genre_classes: {len(le_gen.classes_)}")
    print("[OK] SGT params: see feature generation script for exact settings")

    experiments = build_experiments()
    for exp in experiments:
        run_experiment(exp, X_emb, X_sgt, y_emo, y_gen, y_pair, le_emo, le_gen)


if __name__ == "__main__":
    main()
