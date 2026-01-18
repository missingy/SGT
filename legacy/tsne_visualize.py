# tsne_visualize.py  (fusion-ready, backward compatible)
import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# ---------------- 工具：可选的名称规范化（遇到对不齐时启用） ----------------
def _normalize(s: str) -> str:
    s = str(s).strip().lower().replace("\\", "/").split("/")[-1]
    s = os.path.splitext(s)[0]
    s = re.sub(r"\s+", "", s)
    return s

# ---------------- 原来单源的加载（保持兼容） ----------------
def load_data(vec_path, labels_csv):
    emb = KeyedVectors.load_word2vec_format(vec_path, binary=False)
    df  = pd.read_csv(labels_csv)  # 需要两列：piece,label
    ids = [p for p in df["piece"].tolist() if p in emb]
    X   = np.vstack([emb[p] for p in ids])
    y   = df.set_index("piece").loc[ids, "label"].to_numpy()
    return X, y, ids

# ---------------- 新增：融合加载（emb + SGT→PCA） ----------------
def load_fused(vec_path, labels_csv, sgt_feat_path, sgt_ids_path, pca_n=64, try_normalize=False):
    """
    返回融合后的 X_fused, y, ids
    - 先对 SGT 做 StandardScaler + PCA 到 pca_n
    - 再与 emb 拼接；最后再整体 StandardScaler（便于可视化）
    """
    kv = KeyedVectors.load_word2vec_format(vec_path, binary=False)
    df = pd.read_csv(labels_csv)

    # 读 SGT
    X_sgt_raw = np.load(sgt_feat_path)
    sgt_ids_raw = list(np.load(sgt_ids_path))

    # 直接尝试“精确匹配”的交集（与此前训练评测保持一致）
    emb_keys = set(kv.index_to_key)
    lab_ids  = df["piece"].astype(str).tolist()
    sgt_ids  = set(sgt_ids_raw)

    inter = [p for p in lab_ids if (p in emb_keys and p in sgt_ids)]

    # 如果严格匹配为空，退而求其次做“规范化匹配”（以免命名有微小差异）
    if try_normalize and len(inter) == 0:
        def build_norm_map(raw_list):
            m = {}
            for r in raw_list:
                n = _normalize(r)
                if n not in m:
                    m[n] = r
            return m
        emb_map = build_norm_map(kv.index_to_key)
        sgt_map = build_norm_map(sgt_ids_raw)
        lab_map = build_norm_map(lab_ids)
        inter_norm = list(set(emb_map) & set(sgt_map) & set(lab_map))
        if not inter_norm:
            raise RuntimeError("三路交集为空，请检查命名（扩展名/大小写/空格/路径）。")
        # 回到原始名序列，按 labels 顺序
        inter = [lab_map[n] for n in map(_normalize, lab_ids) if n in inter_norm]

    if len(inter) == 0:
        raise RuntimeError("三路交集为空。请确认 labels.csv 的 piece、emb 的键和 sgt_ids 完全一致（含扩展名）。")

    # 构造 emb 矩阵（按 labels 顺序）
    X_emb = np.vstack([kv[p] for p in inter])

    # 构造 SGT 矩阵（按 labels 顺序）
    sgt_index = {raw: i for i, raw in enumerate(sgt_ids_raw)}
    X_sgt = np.vstack([X_sgt_raw[sgt_index[p]] for p in inter])

    # SGT 标准化 + PCA
    sgt_scaler = StandardScaler()
    X_sgt_s = sgt_scaler.fit_transform(X_sgt)
    pca = PCA(n_components=min(pca_n, X_sgt_s.shape[1]), random_state=42)
    X_sgt_pca = pca.fit_transform(X_sgt_s)

    # 融合 + 整体标准化（便于可视化）
    X_fused = np.hstack([X_emb, X_sgt_pca])
    fused_scaler = StandardScaler()
    X_fused = fused_scaler.fit_transform(X_fused)

    # 标签
    y = df.set_index("piece").loc[inter, "label"].to_numpy()
    return X_fused, y, inter

# ---------------- 绘图（加“类中心大号X”） ----------------
def plot_embedding(Z, y_text, out_path, title, annotate_centers=True):
    classes = np.unique(y_text)
    le = LabelEncoder().fit(y_text)
    y  = le.transform(y_text)

    plt.figure(figsize=(9, 7))
    for k in range(len(classes)):
        idx = (y == k)
        plt.scatter(Z[idx, 0], Z[idx, 1], s=14, alpha=0.7, label=classes[k])

    if annotate_centers:
        for k in range(len(classes)):
            idx = (y == k)
            if idx.sum() == 0:
                continue
            c = Z[idx].mean(axis=0)
            # 大号“X”+ 类名
            plt.scatter([c[0]], [c[1]], marker='X', s=220, linewidths=1.8,
                        edgecolor='k', alpha=0.95, zorder=10)
            plt.text(c[0], c[1], f"  {classes[k]}", fontsize=11, weight='bold',
                     va='center', ha='left', color='k', zorder=11)

    plt.title(title)
    plt.legend(loc='best', fontsize=9, frameon=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"[OK] saved figure -> {out_path}")

def main():
    ap = argparse.ArgumentParser()
    # 原有参数（保持兼容）
    ap.add_argument("--vec", required=True, help="piece_vectors.bin (word2vec文本)")
    ap.add_argument("--labels", required=True, help="labels.csv (含列 piece,label)")
    ap.add_argument("--out", default="tsne.png", help="输出图片路径")
    ap.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--umap", action="store_true", help="使用UMAP替代t-SNE（需 pip install umap-learn）")
    ap.add_argument("--standardize", action="store_true", help="仅在纯 emb 模式下：对特征做StandardScaler")

    # 新增用于融合的可选参数
    ap.add_argument("--sgt-feat", help="sgt_features.npy（可选，提供则做融合）")
    ap.add_argument("--sgt-ids",  help="sgt_ids.npy（可选，提供则做融合）")
    ap.add_argument("--pca-n", type=int, default=64, help="SGT PCA 维度（融合时有效）")
    ap.add_argument("--try-normalize", action="store_true", help="若精确匹配为空，尝试规范化匹配")
    args = ap.parse_args()

    # 选择：纯 emb 可视化 or 融合可视化
    if args.sgt_feat and args.sgt_ids:
        X, y, ids = load_fused(
            args.vec, args.labels, args.sgt_feat, args.sgt_ids,
            pca_n=args.pca_n, try_normalize=args.try_normalize
        )
        title_suffix = f"Fused (100+{args.pca_n}d) → 2D"
    else:
        X, y, ids = load_data(args.vec, args.labels)
        if args.standardize:
            X = StandardScaler().fit_transform(X)
        title_suffix = f"Embedding (dim={X.shape[1]}) → 2D"

    print(f"[OK] samples={len(ids)}, dim={X.shape[1]}, classes={len(np.unique(y))}")

    if args.umap:
        try:
            import umap
        except ImportError:
            raise SystemExit("请先安装 UMAP： pip install umap-learn")
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=args.seed)
        Z = reducer.fit_transform(X)
        title = f"UMAP  |  {title_suffix}"
    else:
        tsne = TSNE(n_components=2, perplexity=args.perplexity, learning_rate='auto',
                    init='pca', random_state=args.seed)
        Z = tsne.fit_transform(X)
        title = f"t-SNE (perplexity={args.perplexity})  |  {title_suffix}"

    plot_embedding(Z, y, args.out, title)

if __name__ == "__main__":
    main()
