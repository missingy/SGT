"""
sgt_features_standard.py
========================
Compute Sequence Graph Transform (SGT) features for each MIDI file,
but using "Standard" SGT definition: decay by sequence-index distance
instead of beat-aligned onset distance.

Everything else is kept the same as sgt_features.py:
- same token extraction (TOKEN_MODE, prog grouping, exclude drums)
- same vocab building (TOP_VOCAB)
- same output format (sgt_features.npy + sgt_ids.npy)

Only change:
  dt := (j - i)   (sequence index distance)
  weight := exp(-ALPHA * dt)
  MAX_DT_BEATS is reused as "MAX_DT_STEPS" (max index distance).
"""

import argparse
import math
import os
from collections import Counter

import numpy as np
import pretty_midi
from tqdm import tqdm

# ==============================
# Defaults (match sgt_features.py)
# ==============================
DATA_ROOT = r"data/Data"
OUTPUT_FEATURES = r"artifacts/midi-embs/myset/sgt_features.npy"
OUTPUT_IDS = r"artifacts/midi-embs/myset/sgt_ids.npy"

ALPHA = 0.8
MAX_DT_BEATS = 8  # reused as max index distance (steps)
TOKEN_MODE = "prog_pitch"  # 'pitch_class', 'prog_pitch', 'interval'
TOP_VOCAB = 64
# ==============================


def extract_events(midi_path):
    """
    Extract event sequence from a MIDI file.
    For compatibility with the original script, return (token, onset_beats)
    and sort by onset. Standard SGT ignores onset in compute_sgt().
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"[WARN] Skip {midi_path}: {e}")
        return []

    events = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        prog = inst.program // 8
        for note in inst.notes:
            onset = pm.time_to_tick(note.start) / pm.resolution
            pitch_class = note.pitch % 12

            if TOKEN_MODE == "pitch_class":
                token = f"pc:{pitch_class}"
            elif TOKEN_MODE == "prog_pitch":
                token = f"p{prog}_pc{pitch_class}"
            elif TOKEN_MODE == "interval":
                token = str(note.pitch)
            else:
                raise ValueError("TOKEN_MODE must be 'pitch_class' / 'prog_pitch' / 'interval'")

            events.append((token, onset))

    return sorted(events, key=lambda x: x[1])


def build_vocab(data_root):
    """Scan dataset tokens and build vocab."""
    all_tokens = []
    for root, _, files in os.walk(data_root):
        for f in files:
            if not f.lower().endswith((".mid", ".midi")):
                continue
            path = os.path.join(root, f)
            ev = extract_events(path)
            all_tokens.extend([t for t, _ in ev])

    counter = Counter(all_tokens)
    vocab = [tok for tok, _ in counter.most_common(TOP_VOCAB)]
    token2idx = {tok: i for i, tok in enumerate(vocab)}
    print(f"[OK] Built vocab of size {len(vocab)}")
    return token2idx


def compute_sgt(events, token2idx, alpha=ALPHA, max_dt_beats=MAX_DT_BEATS):
    """
    Standard SGT:
      - events already sorted by time
      - dt uses index distance: dt = j - i
      - weight = exp(-alpha * dt)
      - max_dt_beats name is kept for compatibility, semantics are steps
    """
    V = len(token2idx)
    S = np.zeros((V, V), dtype=np.float32)

    max_dt_steps = max_dt_beats

    tokens = [t for t, _ in events]
    n = len(tokens)

    for i in range(n):
        ti = tokens[i]
        if ti not in token2idx:
            continue
        idx_i = token2idx[ti]

        j_end = min(n, i + 1 + max_dt_steps)
        for j in range(i + 1, j_end):
            tj = tokens[j]
            if tj not in token2idx:
                continue
            idx_j = token2idx[tj]

            dt = j - i
            S[idx_i, idx_j] += math.exp(-alpha * dt)

    if S.max() > 0:
        S /= S.max()
    return S.flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DATA_ROOT, help="Dataset root directory")
    ap.add_argument("--out-features", default=OUTPUT_FEATURES, help="Output SGT features .npy")
    ap.add_argument("--out-ids", default=OUTPUT_IDS, help="Output SGT ids .npy")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--max-dt-beats", type=float, default=MAX_DT_BEATS)
    ap.add_argument("--token-mode", default=TOKEN_MODE)
    ap.add_argument("--top-vocab", type=int, default=TOP_VOCAB)
    args = ap.parse_args()

    globals()["ALPHA"] = args.alpha
    globals()["MAX_DT_BEATS"] = args.max_dt_beats
    globals()["TOKEN_MODE"] = args.token_mode
    globals()["TOP_VOCAB"] = args.top_vocab

    token2idx = build_vocab(args.data_root)
    features, ids = [], []

    for genre in os.listdir(args.data_root):
        subdir = os.path.join(args.data_root, genre)
        if not os.path.isdir(subdir):
            continue
        for f in tqdm(os.listdir(subdir), desc=f"Processing {genre}"):
            if not f.lower().endswith((".mid", ".midi")):
                continue
            midipath = os.path.join(subdir, f)
            events = extract_events(midipath)
            vec = compute_sgt(events, token2idx)
            features.append(vec)
            ids.append(f)

    features = np.vstack(features)
    out_dir = os.path.dirname(args.out_features)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.out_features, features)
    np.save(args.out_ids, np.array(ids))
    print(f"[DONE] Saved {features.shape} -> {args.out_features}, {args.out_ids}")
    print(
        f"[INFO] Standard SGT: dt=(j-i), alpha={ALPHA}, max_dt_steps={MAX_DT_BEATS}, "
        f"token_mode={TOKEN_MODE}, top_vocab={TOP_VOCAB}"
    )


if __name__ == "__main__":
    main()
