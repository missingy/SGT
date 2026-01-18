# eval_mlp_kfold.py  (Multi-task Learning version for XMIDI: Emotion + Genre)
# ------------------------------------------------------------
# This script trains a multi-task MLP classifier on MIDI2vec piece-level embeddings:
#   - Shared trunk (Dense layers)
#   - Two heads (softmax): emotion head and genre head
#
# Input:
#   - piece_vectors.bin : MIDI2vec embeddings (word2vec text format)
#   - labels.csv        : CSV with columns [piece, emotion, genre]
#
# Output:
#   - 10-fold CV metrics for emotion and genre (Accuracy, Macro-F1)
#   - Overall score (avg of the two tasks)
#   - Confusion matrices (row-normalized) and classification reports for both tasks
#
# Notes:
#   - To avoid information leakage, StandardScaler is fitted inside each fold.
#   - StratifiedKFold uses a combined label "emotion__genre" for stable stratification.
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers, callbacks, Model, Input

# ========= Paths & Hyperparams =========
PIECE_VEC_BIN = r"artifacts/midi-embs/myset/piece_vectors.bin"  # MIDI2vec piece embeddings (text format)
LABELS_CSV    = r"artifacts/midi-embs/myset/labels.csv"         # columns: piece, emotion, genre

N_SPLITS      = 10
EPOCHS        = 50
BATCH_SIZE    = 32
LR            = 1e-3
SEED          = 42
VAL_SPLIT     = 0.1  # explicit stratified split inside each fold (for early stopping)

# MLP architecture (tweak if needed)
TRUNK_1       = 256
TRUNK_2       = 128
DROP_TRUNK    = 0.3
HEAD_HIDDEN   = 64
DROP_HEAD     = 0.2
# ======================================


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_multitask_mlp(input_dim: int, n_emotion: int, n_genre: int) -> Model:
    """
    Multi-task MLP:

      Shared trunk:
        Dense(TRUNK_1, relu) -> Dropout(DROP_TRUNK)
        Dense(TRUNK_2, relu) -> Dropout(DROP_TRUNK)
      Heads:
        emotion: Dense(HEAD_HIDDEN, relu) -> Dropout(DROP_HEAD) -> Dense(n_emotion, softmax)
        genre  : Dense(HEAD_HIDDEN, relu) -> Dropout(DROP_HEAD) -> Dense(n_genre,   softmax)
    """
    inp = Input(shape=(input_dim,), name="x")

    # ---- shared trunk ----
    x = layers.Dense(TRUNK_1, activation="relu")(inp)
    x = layers.Dropout(DROP_TRUNK)(x)
    x = layers.Dense(TRUNK_2, activation="relu")(x)
    x = layers.Dropout(DROP_TRUNK)(x)

    # ---- emotion head ----
    emo = layers.Dense(HEAD_HIDDEN, activation="relu")(x)
    emo = layers.Dropout(DROP_HEAD)(emo)
    out_emo = layers.Dense(n_emotion, activation="softmax", name="emotion")(emo)

    # ---- genre head ----
    gen = layers.Dense(HEAD_HIDDEN, activation="relu")(x)
    gen = layers.Dropout(DROP_HEAD)(gen)
    out_gen = layers.Dense(n_genre, activation="softmax", name="genre")(gen)

    model = Model(inputs=inp, outputs=[out_emo, out_gen], name="mtl_mlp")

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
    """
    Load word2vec-format KeyedVectors and return a dict:
      piece_id -> embedding (np.ndarray)
    """
    kv = KeyedVectors.load_word2vec_format(vec_path, binary=False)
    out = {}
    for k in kv.key_to_index.keys():
        out[k] = kv.get_vector(k)
    return out


def main():
    set_seed(SEED)

    # ------------------------------------------------------------
    # 1) Load labels and embeddings
    # ------------------------------------------------------------
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(f"labels.csv not found: {LABELS_CSV}")
    if not os.path.exists(PIECE_VEC_BIN):
        raise FileNotFoundError(f"piece_vectors.bin not found: {PIECE_VEC_BIN}")

    df = pd.read_csv(LABELS_CSV)
    if not {"piece", "emotion", "genre"}.issubset(df.columns):
        raise ValueError("labels.csv must contain columns: piece, emotion, genre")

    emb_dict = load_piece_vectors(PIECE_VEC_BIN)

    # Align rows: keep only pieces that exist in embeddings
    pieces = []
    emo_text = []
    gen_text = []
    X_list = []

    for _, r in df.iterrows():
        pid = str(r["piece"])
        if pid in emb_dict:
            pieces.append(pid)
            emo_text.append(str(r["emotion"]))
            gen_text.append(str(r["genre"]))
            X_list.append(emb_dict[pid])

    X = np.vstack(X_list).astype(np.float32)
    y_emo_text = np.array(emo_text, dtype=object)
    y_gen_text = np.array(gen_text, dtype=object)

    # Encode labels
    le_emo = LabelEncoder()
    le_gen = LabelEncoder()
    y_emo = le_emo.fit_transform(y_emo_text)
    y_gen = le_gen.fit_transform(y_gen_text)

    n_emotion = len(le_emo.classes_)
    n_genre = len(le_gen.classes_)

    print(f"[OK] samples: {X.shape[0]}, dim: {X.shape[1]}, emotion_classes: {n_emotion}, genre_classes: {n_genre}")

    # ------------------------------------------------------------
    # 2) Stratified K-Fold: use combined label for stratification
    # ------------------------------------------------------------
    # Why: StratifiedKFold accepts only 1D y.
    # We create pair labels like "exciting__rock" to keep joint distribution stable across folds.
    pair_text = np.array([f"{e}__{g}" for e, g in zip(y_emo_text, y_gen_text)], dtype=object)
    le_pair = LabelEncoder()
    y_pair = le_pair.fit_transform(pair_text)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # Track metrics per fold
    emo_accs, emo_f1s = [], []
    gen_accs, gen_f1s = [], []
    overall_accs, overall_f1s = [], []

    # For global confusion matrix / report
    emo_true_all, emo_pred_all = [], []
    gen_true_all, gen_pred_all = [], []

    # ------------------------------------------------------------
    # 3) Cross-validation
    # ------------------------------------------------------------
    fold = 0
    for train_idx, test_idx in skf.split(X, y_pair):
        fold += 1

        # Standardization is applied AFTER an explicit inner train/val split (see below)

        # Build model for this fold
        model = build_multitask_mlp(input_dim=X.shape[1], n_emotion=n_emotion, n_genre=n_genre)

        # Early stopping on validation loss (sum of two task losses)
        es = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=0,
        )

        # ---- Explicit stratified train/val split inside fold ----
        # Keras 'validation_split' slices the tail of the array (not stratified),
        # which is fragile when samples are ordered (e.g., by filename/emotion).
        # We split by the same 1D y used for StratifiedKFold (y_pair) for stability.
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

        # ---- Standardization (fit on inner-train only; avoid leakage into val/test) ----
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_val = scaler.transform(X[val_idx])
        X_te = scaler.transform(X[test_idx])

        yemo_tr, yemo_val = y_emo[tr_idx], y_emo[val_idx]
        ygen_tr, ygen_val = y_gen[tr_idx], y_gen[val_idx]

        yemo_te = y_emo[test_idx]
        ygen_te = y_gen[test_idx]

        history = model.fit(
            X_tr,
            {"emotion": yemo_tr, "genre": ygen_tr},
            validation_data=(X_val, {"emotion": yemo_val, "genre": ygen_val}),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,
            callbacks=[es],
        )

        # Predict (two outputs)
        p_emo, p_gen = model.predict(X_te, verbose=0)
        pred_emo = np.argmax(p_emo, axis=1)
        pred_gen = np.argmax(p_gen, axis=1)

        # Metrics per task
        emo_acc = accuracy_score(yemo_te, pred_emo)
        emo_f1  = f1_score(yemo_te, pred_emo, average="macro")

        gen_acc = accuracy_score(ygen_te, pred_gen)
        gen_f1  = f1_score(ygen_te, pred_gen, average="macro")

        # Overall (simple average; adjust if you prefer weighted average)
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

    print("\n==== CV Results (Multi-task) ====")
    print(f"Emotion Accuracy : {emo_acc_m:.4f} ± {emo_acc_s:.4f}")
    print(f"Emotion Macro-F1 : {emo_f1_m:.4f} ± {emo_f1_s:.4f}")
    print(f"Genre   Accuracy : {gen_acc_m:.4f} ± {gen_acc_s:.4f}")
    print(f"Genre   Macro-F1 : {gen_f1_m:.4f} ± {gen_f1_s:.4f}")
    print(f"Overall Accuracy : {ov_acc_m:.4f} ± {ov_acc_s:.4f}")
    print(f"Overall Macro-F1 : {ov_f1_m:.4f} ± {ov_f1_s:.4f}")

    # Confusion matrices (row-normalized)
    emo_cm = confusion_matrix(emo_true_all, emo_pred_all, normalize="true")
    gen_cm = confusion_matrix(gen_true_all, gen_pred_all, normalize="true")

    print("\n==== Emotion Confusion Matrix (row-normalized) ====")
    print(np.array_str(emo_cm, precision=3, suppress_small=True))

    print("\n==== Genre Confusion Matrix (row-normalized) ====")
    print(np.array_str(gen_cm, precision=3, suppress_small=True))

    # Classification reports
    print("\n==== Emotion Classification Report ====")
    print(classification_report(emo_true_all, emo_pred_all, target_names=le_emo.classes_))

    print("\n==== Genre Classification Report ====")
    print(classification_report(gen_true_all, gen_pred_all, target_names=le_gen.classes_))


if __name__ == "__main__":
    main()
