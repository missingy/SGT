import os
import random
import shutil
from collections import defaultdict

# ========= config area (edit as needed) =========
SOURCE_DIR = r"data/XMIDI_Dataset"
TARGET_DIR = r"data/Data_5000"
RANDOM_SEED = 25

# MODE options: "genre" / "emotion" / "tag" / "multitask"
MODE = "multitask"

# small experiment: modes 1,2,3 use this sample size
N_PER_GROUP = 100

# main experiment: samples per genre in multitask
N_PER_GENRE = 1000   # MAX = 2500

# mode 1 (genre)
# options: "classical","country","jazz","pop","rock"
TARGET_GENRES = {"classical", "country", "jazz", "pop", "rock"}

# mode 2 (emotion)
# options: "exciting","warm","happy","romantic","funny","sad","angry","lazy","quiet","fear","magnificent"
TARGET_EMOTIONS = {"sad", "happy", "romantic", "angry", "magnificent"}
# mode 4 requires both TARGET_GENRES and TARGET_EMOTIONS
# ==============================================

# mode 3 (emotion + genre)
TARGET_TAGS = {
    "magnificent_classical",
    "warm_country",
    "romantic_jazz",
    "happy_pop",
    "angry_rock"
}


def parse_emotion_genre(filename: str):
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split('_')
    if len(parts) < 4:
        return None, None
    return parts[1].lower(), parts[2].lower()


def collect_by_genre(root):
    result = defaultdict(list)
    for rootdir, _, files in os.walk(root):
        for f in files:
            if not f.lower().endswith((".mid", ".midi")):
                continue
            emotion, genre = parse_emotion_genre(f)
            if genre in TARGET_GENRES:
                result[genre].append(os.path.join(rootdir, f))
    return result


def collect_by_emotion(root):
    result = defaultdict(list)
    valid_emos = set(TARGET_EMOTIONS)
    for rootdir, _, files in os.walk(root):
        for f in files:
            if not f.lower().endswith((".mid", ".midi")):
                continue
            emotion, _ = parse_emotion_genre(f)
            if emotion in valid_emos:
                result[emotion].append(os.path.join(rootdir, f))
    return result


def collect_by_tag(root):
    result = defaultdict(list)
    valid_tags = set(TARGET_TAGS)
    for rootdir, _, files in os.walk(root):
        for f in files:
            if not f.lower().endswith((".mid", ".midi")):
                continue
            emotion, genre = parse_emotion_genre(f)
            if not emotion or not genre:
                continue
            tag = f"{emotion}_{genre}"
            if tag in valid_tags:
                result[tag].append(os.path.join(rootdir, f))
    return result


def collect_genre_emotion(root):
    # genre -> emotion -> [paths]
    ge = {g: {e: [] for e in TARGET_EMOTIONS} for g in TARGET_GENRES}
    for rootdir, _, files in os.walk(root):
        for f in files:
            if not f.lower().endswith((".mid", ".midi")):
                continue
            emo, gen = parse_emotion_genre(f)
            if (emo in ge.get(gen, {})):
                ge[gen][emo].append(os.path.join(rootdir, f))
    return ge


def sample_and_copy(groups_dict, mode_name):
    random.seed(RANDOM_SEED)

    # all groups use the same N_PER_GROUP
    for group, files in groups_dict.items():
        print(f"{mode_name} {group}: found {len(files)} tracks")

    selected = {}
    for group, files in groups_dict.items():
        if len(files) < N_PER_GROUP:
            print(f"[WARN] {mode_name} '{group}' has only {len(files)} tracks (< {N_PER_GROUP}), copy all")
            chosen = files
        else:
            chosen = random.sample(files, N_PER_GROUP)
        selected[group] = chosen

    # copy
    for group, files in selected.items():
        out_dir = os.path.join(TARGET_DIR, group)
        os.makedirs(out_dir, exist_ok=True)
        for src in files:
            shutil.copy2(src, os.path.join(out_dir, os.path.basename(src)))

    print("Copy complete. Output dir:", TARGET_DIR)


def sample_and_copy_multitask(ge):
    random.seed(RANDOM_SEED)
    base = N_PER_GENRE // len(TARGET_EMOTIONS)

    for gen in TARGET_GENRES:
        # all files under this genre
        all_files = []
        for emo in TARGET_EMOTIONS:
            all_files.extend(ge[gen][emo])

        print(f"Genre {gen}: total={len(all_files)}")
        if len(all_files) <= N_PER_GENRE:
            chosen = all_files
        else:
            chosen = []

            # (1) reserve per-emotion samples
            for emo in TARGET_EMOTIONS:
                pool = ge[gen][emo]
                if not pool:
                    continue
                k = min(base, len(pool))
                chosen.extend(random.sample(pool, k))

            # (2) fill remainder with random samples
            if len(chosen) < N_PER_GENRE:
                chosen_set = set(chosen)
                rest = [p for p in all_files if p not in chosen_set]
                need = N_PER_GENRE - len(chosen)
                chosen.extend(random.sample(rest, min(need, len(rest))))

        # create per-genre folder (compatible with make_sgt_features.py layout)
        out_dir = os.path.join(TARGET_DIR, gen)
        os.makedirs(out_dir, exist_ok=True)
        for src in chosen:
            shutil.copy2(src, os.path.join(out_dir, os.path.basename(src)))

    print("Copy complete. Output dir:", TARGET_DIR)


def main():
    mode = MODE.lower()

    if mode == "genre":
        print("Current mode: sample by Genre")
        groups = collect_by_genre(SOURCE_DIR)
        groups = {g: groups.get(g, []) for g in TARGET_GENRES}
        sample_and_copy(groups, "Genre")

    elif mode == "tag":
        print("Current mode: sample by Emotion+Genre")
        groups = collect_by_tag(SOURCE_DIR)
        groups = {t: groups.get(t, []) for t in TARGET_TAGS}
        sample_and_copy(groups, "Tag")

    elif mode == "emotion":
        print("Current mode: sample by Emotion")
        groups = collect_by_emotion(SOURCE_DIR)
        groups = {e: groups.get(e, []) for e in TARGET_EMOTIONS}
        sample_and_copy(groups, "Emotion")

    elif mode == "multitask":
        print("Current mode: multitask balanced sampling (genre quota + per-genre emotion floor)")
        ge = collect_genre_emotion(SOURCE_DIR)
        sample_and_copy_multitask(ge)

    else:
        raise ValueError("MODE must be one of: 'genre' / 'tag' / 'emotion' / 'multitask'.")


if __name__ == "__main__":
    main()
