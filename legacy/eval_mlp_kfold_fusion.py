# eval_mlp_kfold_fusion.py  (Multi-task Learning + Feature Fusion for XMIDI)
# ------------------------------------------------------------
# This script trains a multi-task MLP classifier on fused features:
#   - MIDI2vec piece embeddings (dense vectors)
#   - SGT (Sequence Graph Transform) piece-level features (high-dim -> optional PCA)
#   - Fused by concatenation: [emb | sgt_pca]
#
# Input:
#   - piece_vectors.bin : MIDI2vec embeddings (word2vec text format)
#   - labels.csv        : CSV with columns [piece, emotion, genre]
#   - sgt_features.npy  : either:
#       A) dict-like object {piece_id: feature_vector}
#       B) matrix (N,D) with separate ids in sgt_ids.npy
#
# Output:
#   - 10-fold CV metrics for emotion and genre (Accuracy, Macro-F1)
#   - Overall score (avg of the two tasks)
#   - Confusion matrices (row-normalized) and classification reports for both tasks
#
# Notes:
#   - To avoid information leakage, scaling/PCA is fitted inside each fold.
#   - StratifiedKFold uses a combined label "emotion__genre" for stable stratification.
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers, callbacks, Model, Input

# ========= Paths & Hyperparams =========
PIECE_VEC_BIN = r"artifacts/midi-embs/myset/piece_vectors.bin"
LABELS_CSV    = r"artifacts/midi-embs/myset/labels.csv"

# SGT feature files
SGT_FEATURES_NPY = r"artifacts/midi-embs/myset/sgt_features.npy"
SGT_IDS_NPY      = r"artifacts/midi-embs/myset/sgt_ids.npy"      # optional (required if SGT_FEATURES_NPY is a matrix)

PCA_N        = 32  # number of PCA components for SGT features; set None to disable PCA

N_SPLITS     = 10
EPOCHS       = 50
BATCH_SIZE   = 32
LR           = 1e-3
SEED         = 42
VAL_SPLIT    = 0.1  # explicit stratified split inside each fold (for early stopping)

# MLP architecture (same as original multi-task script)
TRUNK_1      = 256
TRUNK_2      = 128
DROP_TRUNK   = 0.3
HEAD_HIDDEN  = 64
DROP_HEAD    = 0.2
# ======================================


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _norm_id(x: str) -> str:
    """
    Robust ID normalization to align:
      - labels.csv 'piece' (often filename)
      - piece_vectors.bin keys
      - sgt_ids.npy entries
    Strategy: keep basename, lowercase, strip extension.
    """
    s = str(x).strip().replace("\\", "/").split("/")[-1]
    s = os.path.splitext(s)[0]
    return s.lower()


def build_multitask_mlp(input_dim: int, n_emotion: int, n_genre: int) -> Model:
    """Same multi-task MLP as eval_mlp_kfold.py."""
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

    model = Model(inputs=inp, outputs=[out_emo, out_gen], name="mtl_mlp_fusion")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss={
            "emotion": "sparse_categorical_crossentropy",
            "genre": "sparse_categorical_crossentropy",
        },
        metrics={
            "emotion": ["accuracy"],
            "genre": ["accuracy"],
        },
    )
    return model


def load_piece_vectors(vec_path: str) -> dict:
    """Load word2vec-format KeyedVectors and return {piece_id: embedding} with normalized keys."""
    kv = KeyedVectors.load_word2vec_format(vec_path, binary=False)
    out = {}
    for k in kv.key_to_index.keys():
        out[_norm_id(k)] = kv.get_vector(k)
    return out


def load_sgt_features(features_path: str, ids_path: str | None = None) -> dict:
    """
    Load SGT features from .npy and return a dict:
      piece_id -> feature vector

    Supports:
      1) features_path is a dict saved by np.save(..., allow_pickle=True)
      2) features_path is a matrix (N,D) and ids_path provides ids (N,)
    """
    obj = np.load(features_path, allow_pickle=True)

    # Case A: dict saved with .item()
    if obj.dtype == object:
        maybe = obj.item()
        if isinstance(maybe, dict):
            out = {}
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

    out = {}
    for i, pid in enumerate(ids):
        out[_norm_id(pid)] = feats[i]
    return out


def main():
    set_seed(SEED)

    # ------------------------------------------------------------
    # 1) Load labels + embeddings + SGT
    # ------------------------------------------------------------
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(f"labels.csv not found: {LABELS_CSV}")
    if not os.path.exists(PIECE_VEC_BIN):
        raise FileNotFoundError(f"piece_vectors.bin not found: {PIECE_VEC_BIN}")
    if not os.path.exists(SGT_FEATURES_NPY):
        raise FileNotFoundError(f"sgt_features.npy not found: {SGT_FEATURES_NPY}")

    df = pd.read_csv(LABELS_CSV)
    if not {"piece", "emotion", "genre"}.issubset(df.columns):
        raise ValueError("labels.csv must contain columns: piece, emotion, genre")

    emb_dict = load_piece_vectors(PIECE_VEC_BIN)
    sgt_dict = load_sgt_features(SGT_FEATURES_NPY, SGT_IDS_NPY if os.path.exists(SGT_IDS_NPY) else None)

    # Align rows by normalized id: keep only pieces existing in BOTH emb and sgt
    ids = []
    emo_text = []
    gen_text = []
    Xe_list = []
    Xs_list = []

    for _, r in df.iterrows():
        pid_raw = str(r["piece"])
        pid = _norm_id(pid_raw)
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

    n_emotion = len(le_emo.classes_)
    n_genre = len(le_gen.classes_)

    print(f"[OK] samples: {X_emb.shape[0]}, d_emb: {X_emb.shape[1]}, d_sgt_raw: {X_sgt.shape[1]}, classes: {n_emotion}/{n_genre}")

    # ------------------------------------------------------------
    # 2) Stratified K-Fold (pair stratification)
    # ------------------------------------------------------------
    pair_text = np.array([f"{e}__{g}" for e, g in zip(y_emo_text, y_gen_text)], dtype=object)
    le_pair = LabelEncoder()
    y_pair = le_pair.fit_transform(pair_text)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    emo_accs, emo_f1s = [], []
    gen_accs, gen_f1s = [], []
    overall_accs, overall_f1s = [], []

    # For global confusion matrix / report
    emo_true_all, emo_pred_all = [], []
    gen_true_all, gen_pred_all = [], []

    # ------------------------------------------------------------
    # 3) Cross-validation (scale/PCA inside each fold; no leakage)
    # ------------------------------------------------------------
    fold = 0
    for train_idx, test_idx in skf.split(X_emb, y_pair):
        fold += 1

        # ---- Explicit stratified train/val split inside fold ----
        # Avoid Keras 'validation_split' (tail slice; not stratified). This is critical when
        # samples are ordered (e.g., filenames) and can otherwise remove entire classes from training.
        y_pair_tr = y_pair[train_idx]
        try:
            tr_idx, val_idx = train_test_split(
                train_idx,
                test_size=VAL_SPLIT,
                random_state=SEED + fold,
                stratify=y_pair_tr,
            )
        except ValueError:
            # Fallback: if some (emotion,genre) pairs are too rare to stratify, stratify by emotion only.
            tr_idx, val_idx = train_test_split(
                train_idx,
                test_size=VAL_SPLIT,
                random_state=SEED + fold,
                stratify=y_emo[train_idx],
            )

        # ---- Slice raw features ----
        Xe_tr_raw = X_emb[tr_idx]
        Xe_val_raw = X_emb[val_idx]
        Xe_te_raw = X_emb[test_idx]

        Xs_tr_raw = X_sgt[tr_idx]
        Xs_val_raw = X_sgt[val_idx]
        Xs_te_raw = X_sgt[test_idx]

        yemo_tr, yemo_val, yemo_te = y_emo[tr_idx], y_emo[val_idx], y_emo[test_idx]
        ygen_tr, ygen_val, ygen_te = y_gen[tr_idx], y_gen[val_idx], y_gen[test_idx]

        # ---- Standardize embeddings (fit on inner-train only) ----
        sc_emb = StandardScaler()
        Xe_tr = sc_emb.fit_transform(Xe_tr_raw)
        Xe_val = sc_emb.transform(Xe_val_raw)
        Xe_te = sc_emb.transform(Xe_te_raw)

        # ---- Standardize SGT (fit on inner-train only) ----
        sc_sgt = StandardScaler()
        Xs_tr = sc_sgt.fit_transform(Xs_tr_raw)
        Xs_val = sc_sgt.transform(Xs_val_raw)
        Xs_te = sc_sgt.transform(Xs_te_raw)

        # ---- Optional PCA on SGT (fit on inner-train only) ----
        if PCA_N is not None:
            n_comp = min(int(PCA_N), Xs_tr.shape[1])
            pca = PCA(n_components=n_comp, random_state=SEED)
            Xs_tr = pca.fit_transform(Xs_tr)
            Xs_val = pca.transform(Xs_val)
            Xs_te = pca.transform(Xs_te)

        # ---- Feature fusion: concat (Xe | Xs) ----
        X_tr = np.hstack([Xe_tr, Xs_tr])
        X_val = np.hstack([Xe_val, Xs_val])
        X_te = np.hstack([Xe_te, Xs_te])

        # (Important) Match original script behavior: it only scales X once.
        # We already scaled Xe and Xs separately; we DO NOT re-scale fused X again.

        model = build_multitask_mlp(
            input_dim=X_tr.shape[1],
            n_emotion=n_emotion,
            n_genre=n_genre,
        )

        es = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=0,
        )

        history = model.fit(
            X_tr,
            {"emotion": yemo_tr, "genre": ygen_tr},
            validation_data=(X_val, {"emotion": yemo_val, "genre": ygen_val}),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=[es],
        )

        p_emo, p_gen = model.predict(X_te, verbose=0)
        pred_emo = np.argmax(p_emo, axis=1)
        pred_gen = np.argmax(p_gen, axis=1)

        emo_acc = accuracy_score(yemo_te, pred_emo)
        emo_f1  = f1_score(yemo_te, pred_emo, average="macro")

        gen_acc = accuracy_score(ygen_te, pred_gen)
        gen_f1  = f1_score(ygen_te, pred_gen, average="macro")

        overall_acc = 0.5 * (emo_acc + gen_acc)
        overall_f1  = 0.5 * (emo_f1 + gen_f1)

        emo_accs.append(emo_acc); emo_f1s.append(emo_f1)
        gen_accs.append(gen_acc); gen_f1s.append(gen_f1)
        overall_accs.append(overall_acc); overall_f1s.append(overall_f1)

        emo_true_all.extend(yemo_te.tolist())
        emo_pred_all.extend(pred_emo.tolist())
        gen_true_all.extend(ygen_te.tolist())
        gen_pred_all.extend(pred_gen.tolist())

        print(
            f"[Fold {fold:02d}] "
            f"Emotion acc={emo_acc:.4f} macro-F1={emo_f1:.4f} | "
            f"Genre acc={gen_acc:.4f} macro-F1={gen_f1:.4f} | "
            f"Overall acc={overall_acc:.4f} macro-F1={overall_f1:.4f} "
            f"(epochs_used={len(history.history['loss'])})"
        )

    # ------------------------------------------------------------
    # 4) Summary
    # ------------------------------------------------------------
    def mean_std(arr):
        return float(np.mean(arr)), float(np.std(arr))

    emo_acc_m, emo_acc_s = mean_std(emo_accs)
    emo_f1_m,  emo_f1_s  = mean_std(emo_f1s)

    gen_acc_m, gen_acc_s = mean_std(gen_accs)
    gen_f1_m,  gen_f1_s  = mean_std(gen_f1s)

    ov_acc_m,  ov_acc_s  = mean_std(overall_accs)
    ov_f1_m,   ov_f1_s   = mean_std(overall_f1s)

    print("\n==== CV Results (Multi-task + Fusion) ====")
    print(f"Emotion Accuracy : {emo_acc_m:.4f} ± {emo_acc_s:.4f}")
    print(f"Emotion Macro-F1 : {emo_f1_m:.4f} ± {emo_f1_s:.4f}")
    print(f"Genre   Accuracy : {gen_acc_m:.4f} ± {gen_acc_s:.4f}")
    print(f"Genre   Macro-F1 : {gen_f1_m:.4f} ± {gen_f1_s:.4f}")
    print(f"Overall Accuracy : {ov_acc_m:.4f} ± {ov_acc_s:.4f}")
    print(f"Overall Macro-F1 : {ov_f1_m:.4f} ± {ov_f1_s:.4f}")

    emo_cm = confusion_matrix(emo_true_all, emo_pred_all, normalize="true")
    gen_cm = confusion_matrix(gen_true_all, gen_pred_all, normalize="true")

    print("\n==== Emotion Confusion Matrix (row-normalized) ====")
    print(np.array_str(emo_cm, precision=3, suppress_small=True))

    print("\n==== Genre Confusion Matrix (row-normalized) ====")
    print(np.array_str(gen_cm, precision=3, suppress_small=True))

    print("\n==== Emotion Classification Report ====")
    print(classification_report(emo_true_all, emo_pred_all, target_names=le_emo.classes_))

    print("\n==== Genre Classification Report ====")
    print(classification_report(gen_true_all, gen_pred_all, target_names=le_gen.classes_))


if __name__ == "__main__":
    main()
