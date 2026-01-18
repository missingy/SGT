import os
import random
import shutil
from collections import defaultdict

# ========= 配置区域（请按需要修改） =========
SOURCE_DIR = r"XMIDI_Dataset"
TARGET_DIR = r"data/Data_100"
RANDOM_SEED = 25

# MODE 可选: "genre" / "emotion" / "tag"  / "multitask"
MODE = "multitask"

# 小实验档：模式1,2,3共用抽样数量
N_PER_GROUP = 100

# 主实验档：multitask每个 Genre 抽多少
N_PER_GENRE = 100   # MAX = 2500

# 模式1（Genre）
TARGET_GENRES = {"classical", "country", "jazz", "pop", "rock"}

# 模式2（Emotion）
TARGET_EMOTIONS = {
    "exciting","warm","happy","romantic","funny",
    "sad","angry","lazy","quiet","fear","magnificent"
}
#配置模式4需要同时用到以上两个集合
# ==========================================

# 模式3（Emotion+Genre）
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
            emo, gen = parse_emotion_genre(f)  # 你已有：parts[1], parts[2] :contentReference[oaicite:2]{index=2}
            if (emo in ge.get(gen, {})):
                ge[gen][emo].append(os.path.join(rootdir, f))
    return ge


def sample_and_copy(groups_dict, mode_name):
    random.seed(RANDOM_SEED)

    # 所有 group 都统一抽 N_PER_GROUP
    for group, files in groups_dict.items():
        print(f"{mode_name} {group}: 找到 {len(files)} 首曲子")

    selected = {}
    for group, files in groups_dict.items():
        if len(files) < N_PER_GROUP:
            print(f"[WARN] {mode_name} '{group}' 只有 {len(files)} 首，少于 {N_PER_GROUP}，全部复制。")
            chosen = files
        else:
            chosen = random.sample(files, N_PER_GROUP)
        selected[group] = chosen

    # 复制
    for group, files in selected.items():
        out_dir = os.path.join(TARGET_DIR, group)
        os.makedirs(out_dir, exist_ok=True)
        for src in files:
            shutil.copy2(src, os.path.join(out_dir, os.path.basename(src)))

    print("完成复制！输出目录：", TARGET_DIR)

def sample_and_copy_multitask(ge):
    random.seed(RANDOM_SEED)
    base = N_PER_GENRE // len(TARGET_EMOTIONS)

    for gen in TARGET_GENRES:
        # 该 genre 下所有文件
        all_files = []
        for emo in TARGET_EMOTIONS:
            all_files.extend(ge[gen][emo])

        print(f"Genre {gen}: total={len(all_files)}")
        if len(all_files) <= N_PER_GENRE:
            chosen = all_files
        else:
            chosen = []

            # (1) emotion 保底
            for emo in TARGET_EMOTIONS:
                pool = ge[gen][emo]
                if not pool:
                    continue
                k = min(base, len(pool))
                chosen.extend(random.sample(pool, k))

            # (2) 不足部分用剩余随机补齐
            if len(chosen) < N_PER_GENRE:
                chosen_set = set(chosen)
                rest = [p for p in all_files if p not in chosen_set]
                need = N_PER_GENRE - len(chosen)
                chosen.extend(random.sample(rest, min(need, len(rest))))

        # 按 genre 建文件夹（兼容 make_sgt_features.py 的目录结构）:contentReference[oaicite:3]{index=3}
        out_dir = os.path.join(TARGET_DIR, gen)
        os.makedirs(out_dir, exist_ok=True)
        for src in chosen:
            shutil.copy2(src, os.path.join(out_dir, os.path.basename(src)))

    print("完成复制！输出目录：", TARGET_DIR)

def main():
    mode = MODE.lower()

    if mode == "genre":
        print("当前模式：按 Genre 抽样")
        groups = collect_by_genre(SOURCE_DIR)
        groups = {g: groups.get(g, []) for g in TARGET_GENRES}
        sample_and_copy(groups, "Genre")

    elif mode == "tag":
        print("当前模式：按 Emotion+Genre 抽样")
        groups = collect_by_tag(SOURCE_DIR)
        groups = {t: groups.get(t, []) for t in TARGET_TAGS}
        sample_and_copy(groups, "Tag")

    elif mode == "emotion":
        print("当前模式：按 Emotion 抽样")
        groups = collect_by_emotion(SOURCE_DIR)
        groups = {e: groups.get(e, []) for e in TARGET_EMOTIONS}
        sample_and_copy(groups, "Emotion")

    elif mode == "multitask":
        print("当前模式：多任务均衡抽样（Genre定额 + Genre内Emotion保底）")
        ge = collect_genre_emotion(SOURCE_DIR)
        sample_and_copy_multitask(ge)

    else:
        raise ValueError("MODE 必须是 'genre' / 'tag' / 'emotion' 之一。")


if __name__ == "__main__":
    main()
