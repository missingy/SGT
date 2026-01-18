"""
tsne_visualize_fusion.py
========================
可视化三种表示：MIDI2Vec(100d)、SGT_PCA(64d)、融合(164d) 的 t-SNE/UMAP 2D 图。
要求：
- embeddings/yourset.bin            # 曲目级向量（文本版）
- sgt_features.npy, sgt_ids.npy     # 由 make_sgt_features.py 生成
- labels.csv                        # 列包含: piece, label
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gensim.models import KeyedVectors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ============ 配置 ============
EMB_PATH = "artifacts/midi-embs/myset/piece_vectors.bin"  # 你的曲目级 .bin（文本格式）
SGT_PATH = "sgt_features.npy"
SGT_IDS  = "sgt_ids.npy"
LABELS_CSV = "labels.csv"

USE_UMAP = True        # =True 用 UMAP，可 pip install umap-learn
PCA_N = 64              # 与评测时一致
TSNE_PERPLEXITY = 30    # 5~50 之间试
TSNE_SEED = 42
MARKER_SIZE = 18
ALPHA = 0.9
# ============================


def _normalize(s: str) -> str:
    """统一化：小写、取文件名、去扩展名、去空格"""
    s = str(s).strip().lower().replace("\\", "/").split("/")[-1]
    s = os.path.splitext(s)[0]
    s = re.sub(r"\s+", "", s)
    return s


def align_and_build_matrices():
    """
    对齐三路 ID，构造：
      X_emb: (N, 100)
      X_sgt: (N, D_sgt_raw) -> 再做 PCA 到 64
      y: 标签(字符串)
      ids: 原始 piece 名（含扩展名）
    """
    # 载入
    kv = KeyedVectors.load_word2vec_format(EMB_PATH, binary=False)
    emb_keys_raw = kv.index_to_key
    sgt_ids_raw = list(np.load(SGT_IDS))
    X_sgt_raw = np.load(SGT_PATH)
    df = pd.read_csv(LABELS_CSV)

    lab_ids_raw = df["piece"].astype(str).tolist()
    lab_labels  = df.set_index("piece")["label"].astype(str)

    # norm -> raw map
    def build_norm_map(raw_list):
        m = {}
        for r in raw_list:
            n = _normalize(r)
            if n not in m:
                m[n] = r
        return m

    emb_map = build_norm_map(emb_keys_raw)
    sgt_map = build_norm_map(sgt_ids_raw)
    lab_map = build_norm_map(lab_ids_raw)

    inter = set(emb_map) & set(sgt_map) & set(lab_map)
    if not inter:
        raise RuntimeError("三路交集为空，请检查文件命名是否一致（扩展名/大小写/空格等）。")

    # 按 labels.csv 顺序输出，避免打乱
    norms_in_order = [n for n in map(_normalize, lab_ids_raw) if n in inter]
    raw_ids_in_order = [lab_map[n] for n in norms_in_order]

    # 取三种数据
    X_emb = np.vstack([kv[emb_map[n]] for n in norms_in_order])

    sgt_index = {raw: i for i, raw in enumerate(sgt_ids_raw)}
    X_sgt = np.vstack([X_sgt_raw[sgt_index[sgt_map[n]]] for n in norms_in_order])

    y = np.array([lab_labels[lab_map[n]] for n in norms_in_order])
    return X_emb, X_sgt, y, raw_ids_in_order


def reduce_and_stack_for_plots(X_emb, X_sgt):
    """
    - 对 SGT 做标准化 + PCA 到 PCA_N
    - 融合 = [emb, sgt_pca]
    - 另外返回 emb_scaled, sgt_pca_scaled, fused_scaled，分别用于 t-SNE/UMAP
    """
    # EMB：仅标准化
    emb_scaler = StandardScaler()
    X_emb_s = emb_scaler.fit_transform(X_emb)

    # SGT：标准化 + PCA
    sgt_scaler = StandardScaler()
    X_sgt_s = sgt_scaler.fit_transform(X_sgt)

    pca = PCA(n_components=min(PCA_N, X_sgt_s.shape[1]), random_state=TSNE_SEED)
    X_sgt_pca = pca.fit_transform(X_sgt_s)

    # 融合后再整体标准化（让两侧量纲接近）
    fused = np.hstack([X_emb, X_sgt_pca])
    fused_scaler = StandardScaler()
    fused_s = fused_scaler.fit_transform(fused)

    # 单独画 emb 与 sgt_pca 时也各自标准化（上面已做）
    return X_emb_s, X_sgt_pca, fused_s


def embed_2d(X):
    """降到2维：t-SNE 或 UMAP"""
    if USE_UMAP:
        import umap
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=TSNE_SEED)
        return reducer.fit_transform(X)
    else:
        tsne = TSNE(
            n_components=2,
            perplexity=TSNE_PERPLEXITY,
            learning_rate='auto',
            init='pca',
            random_state=TSNE_SEED
        )
        return tsne.fit_transform(X)


def plot_class_centers(ax, Z, y_idx, classes, size=220):
    """
    在二维嵌入 Z 上为每个类别画一个大号 'X' 作为类中心。
    - Z: (N, 2) 该面板的2D坐标
    - y_idx: (N,) 类别索引（LabelEncoder之后）
    - classes: 类别名列表（用于选择颜色/注释）
    """
    from matplotlib.cm import get_cmap
    cmap = get_cmap('tab10')
    for c, name in enumerate(classes):
        mask = (y_idx == c)
        if mask.sum() == 0:
            continue
        center = Z[mask].mean(axis=0)
        # 用和散点一致的类别色，黑色描边，zorder调高些
        ax.scatter(center[0], center[1],
                   marker='X', s=size,
                   facecolor=cmap(c), edgecolor='k',
                   linewidths=1.5, alpha=0.95,
                   zorder=10)
        # 可选：在中心旁标注类名（不想标就注释掉下面两行）
        ax.text(center[0], center[1],
                f" {name}", fontsize=9, weight='bold',
                va='center', ha='left',
                color='k', zorder=11)



def plot_three(axs, Z_emb, Z_sgt, Z_fused, y):
    """画 1×3 子图：emb / sgt_pca / fused，并叠加每类的中心大号 'X'。"""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_idx = le.fit_transform(y)
    classes = le.classes_

    titles = [
        f"MIDI2Vec (100d) → 2D",
        f"SGT PCA ({PCA_N}d) → 2D",
        f"Fused (100+{PCA_N}d) → 2D"
    ]
    Z_list = [Z_emb, Z_sgt, Z_fused]

    for ax, Z, title in zip(axs, Z_list, titles):
        # 先画各类别散点
        for c in range(len(classes)):
            mask = (y_idx == c)
            ax.scatter(Z[mask, 0], Z[mask, 1],
                       s=MARKER_SIZE, alpha=ALPHA,
                       label=classes[c])
        # 叠加类中心 'X'
        plot_class_centers(ax, Z, y_idx, classes, size=220)

        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        ax.axis('equal')

    # 统一图例（只用类别点的句柄，不把中心再加进图例，避免重复）
    handles, labels = axs[-1].get_legend_handles_labels()
    axs[-1].legend(handles, labels, loc='best', fontsize=9, frameon=True)



def main():
    X_emb, X_sgt, y, ids = align_and_build_matrices()
    print(f"[OK] N={len(y)}, emb_dim={X_emb.shape[1]}, sgt_dim={X_sgt.shape[1]}")

    X_emb_s, X_sgt_pca, X_fused_s = reduce_and_stack_for_plots(X_emb, X_sgt)

    # 各自降到2D
    print("[INFO] Embedding to 2D (this may take a while)...")
    Z_emb   = embed_2d(X_emb_s)
    Z_sgt   = embed_2d(X_sgt_pca)
    Z_fused = embed_2d(X_fused_s)

    # 画图
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    plot_three(axs, Z_emb, Z_sgt, Z_fused, y)
    plt.suptitle("t-SNE/UMAP Comparison of Representations", fontsize=13)
    plt.show()


if __name__ == "__main__":
    main()
