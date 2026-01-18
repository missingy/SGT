"""
sgt_features_drums.py
====================
Compute Sequence Graph Transform (SGT) features for each MIDI file.

This version enhances the original sgt_features.py by adding:
-  Include drum tracks (inst.is_drum) as tokens
-  Add rhythmic position tokens (e.g., 16th-grid within a bar) to capture groove
-  Control vocabulary explosion with drum/non-drum budgets
-  Optional fallback: if "token@pos" is OOV, fall back to base token (token without @pos)
-  TOP_VOCAB = 128 (SGT dim = 128^2 = 16384)

Notes
-----
1) Beat mapping is computed from pretty_midi's beat times (tempo-aware).
2) Bar position uses the first time signature if present, otherwise assumes 4/4.
3) By default, position encoding is applied to DRUM tokens only (recommended).
"""

import argparse
import os
import math
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pretty_midi
from tqdm import tqdm


# ==============================
# Config
# ==============================
DATA_ROOT = r"data/Data"  # e.g., ./data/Data/classical/*.mid
OUTPUT_FEATURES = r"artifacts/midi-embs/myset/sgt_features.npy"
OUTPUT_IDS = r"artifacts/midi-embs/myset/sgt_ids.npy"

# SGT parameters
ALPHA = 1.5
MAX_DT_BEATS = 8

# Token settings (non-drums)
TOKEN_MODE = "prog_pitch"  # 'pitch_class', 'prog_pitch', 'interval'
PROG_GROUP_SIZE = 8        # inst.program // 8 (keep your existing design)

# Vocabulary
TOP_VOCAB = 128            # SGT dim = TOP_VOCAB^2

# Drums
INCLUDE_DRUMS = True
DRUM_TOKEN_MODE = "gm_group"   # 'gm_pitch' or 'gm_group'
DRUM_GROUP_BIN = 4             # for gm_group

# Rhythm position encoding
USE_POS = True
POS_BINS = 16                   # try 8 / 16; 16 is recommended
POS_APPLY_TO = "drums"          # 'drums' | 'all' | 'none'
POS_FALLBACK_TO_BASE = True     # if token@pos OOV, try token without @pos

# Vocab budgets (sum may be <= TOP_VOCAB; remaining filled by frequency)
DRUM_VOCAB_SIZE = 64            # with @pos tokens, drums need more budget
# Remaining (TOP_VOCAB - DRUM_VOCAB_SIZE) goes to non-drum tokens
# ==============================


def _is_midi_file(name: str) -> bool:
    return name.lower().endswith((".mid", ".midi"))


def _is_drum_token(tok: str) -> bool:
    return tok.startswith("dr:") or tok.startswith("drg:")


def _strip_pos(tok: str) -> str:
    if "@p" in tok:
        return tok.split("@p", 1)[0]
    return tok


def _drum_base_token(pitch: int) -> str:
    # GM drum pitch is an instrument id, not tonal pitch.
    if DRUM_TOKEN_MODE == "gm_pitch":
        return f"dr:{pitch}"
    return f"drg:{pitch // DRUM_GROUP_BIN}"


@dataclass
class TimeGrid:
    beat_times: np.ndarray
    beats_per_bar: float
    pos_bins: int

    def time_to_beats(self, t: float) -> float:
        """Map time (seconds) to continuous beats using beat times (tempo-aware)."""
        bt = self.beat_times
        if bt.size < 2:
            return 0.0
        i = int(np.searchsorted(bt, t, side="right") - 1)
        i = max(0, min(i, bt.size - 2))
        t0, t1 = float(bt[i]), float(bt[i + 1])
        if t1 <= t0:
            return float(i)
        frac = (t - t0) / (t1 - t0)
        frac = max(0.0, min(1.0, frac))
        return float(i) + frac

    def beatpos_token(self, beats: float) -> str:
        """Convert beats to bar-position bin token '@p{bin}'."""
        if self.pos_bins <= 0:
            return ""
        b_in_bar = beats % self.beats_per_bar
        pos = int(math.floor((b_in_bar / self.beats_per_bar) * self.pos_bins))
        pos = max(0, min(pos, self.pos_bins - 1))
        return f"@p{pos}"


def _get_timegrid(pm: pretty_midi.PrettyMIDI, pos_bins: int) -> TimeGrid:
    beat_times = np.array(pm.get_beats(), dtype=np.float64)
    if beat_times.size < 2:
        beat_times = np.array([0.0, 0.5], dtype=np.float64)  # fallback

    if pm.time_signature_changes:
        ts0 = pm.time_signature_changes[0]
        num, den = ts0.numerator, ts0.denominator
    else:
        num, den = 4, 4

    beats_per_bar = float(num) * (4.0 / float(den))  # 4/4 => 4 beats/bar
    return TimeGrid(beat_times=beat_times, beats_per_bar=beats_per_bar, pos_bins=pos_bins)


# ---------- Step 1: Extract events ----------
def extract_events(midi_path: str) -> List[Tuple[str, float]]:
    """Extract (token, onset_beats) events from a MIDI file."""
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"[WARN] Skip {midi_path}: {e}")
        return []

    tg = _get_timegrid(pm, POS_BINS if USE_POS else 0)
    events: List[Tuple[str, float]] = []

    for inst in pm.instruments:
        # ---- drums ----
        if inst.is_drum:
            if not INCLUDE_DRUMS:
                continue
            for note in inst.notes:
                beats = tg.time_to_beats(note.start)
                base = _drum_base_token(note.pitch)
                tok = base
                if USE_POS and POS_APPLY_TO in ("drums", "all"):
                    tok = base + tg.beatpos_token(beats)
                events.append((tok, beats))
            continue

        # ---- non-drums ----
        prog = inst.program // PROG_GROUP_SIZE
        for note in inst.notes:
            beats = tg.time_to_beats(note.start)
            pc = note.pitch % 12

            if TOKEN_MODE == "pitch_class":
                base = f"pc:{pc}"
            elif TOKEN_MODE == "prog_pitch":
                base = f"p{prog}_pc{pc}"
            elif TOKEN_MODE == "interval":
                base = str(note.pitch)
            else:
                raise ValueError("TOKEN_MODE must be 'pitch_class' / 'prog_pitch' / 'interval'")

            tok = base
            if USE_POS and POS_APPLY_TO == "all":
                tok = base + tg.beatpos_token(beats)

            events.append((tok, beats))

    events.sort(key=lambda x: x[1])
    return events


# ---------- Step 2: Build vocab with budgets ----------
def build_vocab(data_root: str) -> Dict[str, int]:
    mel_cnt = Counter()
    drm_cnt = Counter()

    for root, _, files in os.walk(data_root):
        for f in files:
            if not _is_midi_file(f):
                continue
            path = os.path.join(root, f)
            ev = extract_events(path)
            for tok, _ in ev:
                if _is_drum_token(tok):
                    drm_cnt[tok] += 1
                else:
                    mel_cnt[tok] += 1

    drum_k = min(DRUM_VOCAB_SIZE, TOP_VOCAB)
    mel_k = max(0, TOP_VOCAB - drum_k)

    vocab_drm = [t for t, _ in drm_cnt.most_common(drum_k)]
    vocab_mel = [t for t, _ in mel_cnt.most_common(mel_k)]
    vocab = vocab_mel + vocab_drm

    # Fill remaining slots by frequency (mel first, then drum)
    if len(vocab) < TOP_VOCAB:
        need = TOP_VOCAB - len(vocab)
        used = set(vocab)
        for t, _ in mel_cnt.most_common():
            if t in used:
                continue
            vocab.append(t)
            used.add(t)
            need -= 1
            if need == 0:
                break

    if len(vocab) < TOP_VOCAB:
        need = TOP_VOCAB - len(vocab)
        used = set(vocab)
        for t, _ in drm_cnt.most_common():
            if t in used:
                continue
            vocab.append(t)
            used.add(t)
            need -= 1
            if need == 0:
                break

    vocab = vocab[:TOP_VOCAB]
    token2idx = {t: i for i, t in enumerate(vocab)}

    print(f"[OK] Built vocab size={len(vocab)} | mel={len(vocab_mel)} drum={len(vocab_drm)}")
    if INCLUDE_DRUMS and len(drm_cnt) == 0:
        print("[WARN] INCLUDE_DRUMS=True but no drum tokens found. Check your MIDI files / drum encoding.")
    if USE_POS:
        pos_ratio = sum(1 for t in vocab if "@p" in t) / max(1, len(vocab))
        print(f"[OK] Vocab pos-token ratio: {pos_ratio:.2f} (POS_BINS={POS_BINS}, apply_to={POS_APPLY_TO})")

    return token2idx


# ---------- Step 3: Compute SGT ----------
def compute_sgt(events: List[Tuple[str, float]], token2idx: Dict[str, int]) -> np.ndarray:
    V = len(token2idx)
    S = np.zeros((V, V), dtype=np.float32)

    for i in range(len(events)):
        ti, ti_time = events[i]
        idx_i = token2idx.get(ti, None)
        if idx_i is None and POS_FALLBACK_TO_BASE:
            idx_i = token2idx.get(_strip_pos(ti), None)
        if idx_i is None:
            continue

        for j in range(i + 1, len(events)):
            tj, tj_time = events[j]
            dt = tj_time - ti_time
            if dt < 0:
                continue
            if dt > MAX_DT_BEATS:
                break

            idx_j = token2idx.get(tj, None)
            if idx_j is None and POS_FALLBACK_TO_BASE:
                idx_j = token2idx.get(_strip_pos(tj), None)
            if idx_j is None:
                continue

            S[idx_i, idx_j] += math.exp(-ALPHA * dt)

    m = float(S.max())
    if m > 0:
        S /= m
    return S.reshape(-1)


# ---------- Step 4: Iterate dataset ----------
def _collect_files(data_root: str) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for genre in os.listdir(data_root):
        subdir = os.path.join(data_root, genre)
        if not os.path.isdir(subdir):
            continue
        for f in os.listdir(subdir):
            if not _is_midi_file(f):
                continue
            items.append((f, os.path.join(subdir, f)))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--out-features", default=OUTPUT_FEATURES)
    ap.add_argument("--out-ids", default=OUTPUT_IDS)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--max-dt-beats", type=float, default=MAX_DT_BEATS)
    ap.add_argument("--token-mode", default=TOKEN_MODE)
    ap.add_argument("--top-vocab", type=int, default=TOP_VOCAB)
    ap.add_argument("--prog-group-size", type=int, default=PROG_GROUP_SIZE)
    ap.add_argument("--include-drums", action="store_true")
    ap.add_argument("--exclude-drums", action="store_true")
    ap.add_argument("--drum-token-mode", default=DRUM_TOKEN_MODE)
    ap.add_argument("--drum-group-bin", type=int, default=DRUM_GROUP_BIN)
    ap.add_argument("--use-pos", action="store_true")
    ap.add_argument("--no-pos", action="store_true")
    ap.add_argument("--pos-bins", type=int, default=POS_BINS)
    ap.add_argument("--pos-apply-to", default=POS_APPLY_TO)
    ap.add_argument("--pos-fallback-to-base", action="store_true")
    ap.add_argument("--drum-vocab-size", type=int, default=DRUM_VOCAB_SIZE)
    args = ap.parse_args()

    global ALPHA, MAX_DT_BEATS, TOKEN_MODE, TOP_VOCAB
    global PROG_GROUP_SIZE, INCLUDE_DRUMS, DRUM_TOKEN_MODE, DRUM_GROUP_BIN
    global USE_POS, POS_BINS, POS_APPLY_TO, POS_FALLBACK_TO_BASE, DRUM_VOCAB_SIZE

    ALPHA = args.alpha
    MAX_DT_BEATS = args.max_dt_beats
    TOKEN_MODE = args.token_mode
    TOP_VOCAB = args.top_vocab
    PROG_GROUP_SIZE = args.prog_group_size
    DRUM_TOKEN_MODE = args.drum_token_mode
    DRUM_GROUP_BIN = args.drum_group_bin
    DRUM_VOCAB_SIZE = args.drum_vocab_size
    POS_BINS = args.pos_bins
    POS_APPLY_TO = args.pos_apply_to

    if args.include_drums:
        INCLUDE_DRUMS = True
    if args.exclude_drums:
        INCLUDE_DRUMS = False
    if args.use_pos:
        USE_POS = True
    if args.no_pos:
        USE_POS = False
    if args.pos_fallback_to_base:
        POS_FALLBACK_TO_BASE = True

    out_dir = os.path.dirname(args.out_features)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    token2idx = build_vocab(args.data_root)
    V = len(token2idx)
    D = V * V

    files = _collect_files(args.data_root)
    if not files:
        raise RuntimeError(f"No MIDI files found under DATA_ROOT='{args.data_root}'")

    N = len(files)
    print(f"[OK] total midi files: {N}, vocab={V}, sgt_dim={D}")
    print(f"[OK] SGT params: alpha={ALPHA}, max_dt_beats={MAX_DT_BEATS}, token_mode={TOKEN_MODE}, top_vocab={TOP_VOCAB}")
    print(f"[OK] Rhythm pos: use_pos={USE_POS}, pos_bins={POS_BINS}, apply_to={POS_APPLY_TO}, fallback_to_base={POS_FALLBACK_TO_BASE}")
    print(f"[OK] Drums: include={INCLUDE_DRUMS}, drum_token_mode={DRUM_TOKEN_MODE}, drum_vocab_size={DRUM_VOCAB_SIZE}")

    features = np.lib.format.open_memmap(
        args.out_features, mode="w+", dtype=np.float32, shape=(N, D)
    )
    ids: List[str] = []

    for k, (fid, path) in enumerate(tqdm(files, desc="Computing SGT")):
        ev = extract_events(path)
        if not ev:
            features[k, :] = 0.0
        else:
            features[k, :] = compute_sgt(ev, token2idx)
        ids.append(fid)

    np.save(args.out_ids, np.array(ids, dtype=object))
    features.flush()
    print(f"[DONE] Saved features {features.shape} -> {args.out_features}")
    print(f"[DONE] Saved ids      {len(ids)} -> {args.out_ids}")


if __name__ == "__main__":
    main()
