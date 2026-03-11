# piece_vectors.py
# ------------------------------------------------------------
# Purpose:
#   Read node-level word2vec (text) vectors + 4 edgelists.
#   From each "file node", collect the track's graph nodes, average them,
#   and export track-level .bin vectors.
#   Also emit a debug CSV about alignment coverage.
# ------------------------------------------------------------
import argparse
import os
import re
import csv
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import networkx as nx
from gensim.models import KeyedVectors

# ============ file paths ============
EMB_PATH      = r"artifacts/midi-embs/myset/embeddings.bin"  # node-level embeddings (text format)
LABELS_CSV    = r"artifacts/midi-embs/myset/labels.csv"      # two columns: piece,label
EDGELIST_DIR  = r"external/midi2vec/edgelist"               # notes/program/tempo/time.signature.edgelist dir
OUT_PIECE_BIN = r"artifacts/midi-embs/myset/piece_vectors.bin"   # exported track-level word2vec (text)
OUT_DEBUG_CSV = r"artifacts/midi-embs/myset/piece_aggregate_stats.csv"  # debug stats (node counts per track)
# ===================================

# tunables
BINARY_EMB    = False     # embeddings are in text format; set False
MAX_HOPS      = 1         # recommend 1 (only 1-hop neighbors); 2 is optional; None means full component (not advised)
INCLUDE_FILE_NODE = True  # include the "file node" itself in aggregation
MIN_NODES_FOR_AGG = 5     # skip if fewer nodes; prevents noisy samples
# filter "global hub nodes" (cross-track bridges, likely noisy)
EXCLUDE_PREFIXES = (
    "vel:", "dur:", "tempo:",                 # common prefixes
    "http://purl.org/midi-ld/time",           # RDF time node
    "http://purl.org/midi-ld/tempo"           # RDF tempo node
)
# optional: include only "informative" prefixes (e.g. notes/instruments)
# INCLUDE_ANY_OF = ("note", "notes", "program", "pitch")  # example; empty means no filter


def stem_filename(name: str) -> str:
    s = name.lower().replace('\\', '/').split('/')[-1]
    s = re.sub(r'\.midi?$', '', s)
    return s


def load_labels(path: str) -> Dict[str, str]:
    labels = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        assert 'piece' in reader.fieldnames, "labels.csv must include column: piece"
        # compatibility: prefer genre, fallback to label
        key = 'genre' if 'genre' in reader.fieldnames else 'label'
        assert key in reader.fieldnames, "labels.csv must include column: genre (or legacy label)"
        for row in reader:
            labels[row['piece']] = row[key]
    return labels


def load_embeddings(path: str, binary: bool) -> KeyedVectors:
    kv = KeyedVectors.load_word2vec_format(path, binary=binary)
    print(f"[OK] embeddings loaded: {len(kv.index_to_key)} vectors, dim={kv.vector_size}")
    return kv


def read_all_edgelists(edgelist_dir: str) -> nx.Graph:
    g = nx.Graph()
    edges = 0
    for name in os.listdir(edgelist_dir):
        if not name.endswith(".edgelist"):
            continue
        p = os.path.join(edgelist_dir, name)
        with open(p, encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                a, b = parts[:2]
                g.add_edge(a, b)
                edges += 1
        print(f"[OK] loaded {name}")
    print(f"[OK] graph edges: {edges}, nodes: {g.number_of_nodes()}")
    return g


def index_file_nodes(all_keys: List[str]):
    """Pick keys that look like "file nodes" and build tail-index and genre-index."""
    file_like = []
    for k in all_keys:
        if re.match(r"^[A-Za-z]:", k) or k.lower().startswith('file:'):
            file_like.append(k)

    print(f"[INFO] detected {len(file_like)} file-like keys")

    suffix_index: Dict[str, List[str]] = defaultdict(list)
    genre_index: Dict[str, List[str]] = defaultdict(list)

    for k in file_like:
        parts = k.replace('\\','/').split('-')
        suf = parts[-1].lower()
        suffix_index[suf].append(k)

        m = re.search(r'-data-([a-zA-Z0-9]+)-', k)  # example: D:-PGM-Clone-data-pop-<basename>
        if m:
            genre = m.group(1).lower()
            genre_index[genre].append(k)

    return {"_all": file_like, "_suffix": suffix_index, "_genre": genre_index}


def find_piece_file_key(piece: str, label: str, file_index) -> Optional[str]:
    """Match by tail segment (basename) + genre; fallback to substring match."""
    base = stem_filename(piece)  # e.g. XMIDI_angry_pop_2IDEWIOS
    suf_hits = file_index["_suffix"].get(base, [])
    if len(suf_hits) == 1:
        return suf_hits[0]
    elif len(suf_hits) > 1:
        genre = (label or "").lower()
        if genre and genre in file_index["_genre"]:
            cand = [k for k in suf_hits if k in file_index["_genre"][genre]]
            if cand:
                return cand[0]
        print(f"[WARN] multiple file nodes match base '{base}', picking first: {suf_hits[0]}")
        return suf_hits[0]

    # tail miss: relaxed substring matching
    base_no_underscore = base.replace('_','').lower()
    for k in file_index["_all"]:
        low = k.lower()
        if base in low or base_no_underscore in low:
            genre = (label or "").lower()
            if genre and f"-{genre}-" in low:
                return k
            return k

    return None


def bfs_collect(g: nx.Graph, start: str, max_hops: Optional[int]) -> Set[str]:
    """BFS collect nodes; max_hops=None means full connected component."""
    if start not in g:
        return set()
    if max_hops is None:
        return set(nx.node_connected_component(g, start))
    seen = {start}
    q = deque([(start, 0)])
    while q:
        u, d = q.popleft()
        if d == max_hops:
            continue
        for v in g.neighbors(u):
            if v not in seen:
                seen.add(v)
                q.append((v, d+1))
    return seen


def useful_node(name: str) -> bool:
    """Filter global hub nodes; add INCLUDE rules here if you only want certain prefixes."""
    for p in EXCLUDE_PREFIXES:
        if name.startswith(p):
            return False
    # If you want to keep only certain types, uncomment next block and define INCLUDE_ANY_OF.
    # if INCLUDE_ANY_OF:
    #     return any(sub in name for sub in INCLUDE_ANY_OF)
    return True


def export_word2vec_text(out_path: str, vectors: Dict[str, np.ndarray], dim: int):
    """Write in word2vec text format: first line N dim, then 'key v1 v2 ...'."""
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(f"{len(vectors)} {dim}\n")
        for key, vec in vectors.items():
            vals = " ".join(f"{x:.6f}" for x in vec.tolist())
            f.write(f"{key} {vals}\n")
    print(f"[OK] saved piece vectors -> {out_path} ({len(vectors)} items)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", default=EMB_PATH, help="Node embeddings (word2vec text)")
    ap.add_argument("--binary-emb", action="store_true", help="Embeddings are binary word2vec")
    ap.add_argument("--labels", default=LABELS_CSV, help="labels.csv")
    ap.add_argument("--edgelist-dir", default=EDGELIST_DIR, help="edgelist directory")
    ap.add_argument("--out-piece-bin", default=OUT_PIECE_BIN, help="Output piece vectors (word2vec text)")
    ap.add_argument("--out-debug-csv", default=OUT_DEBUG_CSV, help="Output debug stats CSV")
    ap.add_argument("--max-hops", type=int, default=MAX_HOPS if MAX_HOPS is not None else -1)
    ap.add_argument("--min-nodes", type=int, default=MIN_NODES_FOR_AGG)
    ap.add_argument("--include-file-node", action="store_true")
    ap.add_argument("--exclude-file-node", action="store_true")
    args = ap.parse_args()

    max_hops = None if args.max_hops == -1 else args.max_hops
    include_file_node = INCLUDE_FILE_NODE
    if args.include_file_node:
        include_file_node = True
    if args.exclude_file_node:
        include_file_node = False

    # 1) load inputs
    labels = load_labels(args.labels)
    kv = load_embeddings(args.embeddings, binary=args.binary_emb)
    g = read_all_edgelists(args.edgelist_dir)

    # 2) match each piece to its file node
    idx = index_file_nodes(kv.index_to_key)
    piece2filenode: Dict[str, str] = {}
    miss = []
    for piece, lab in labels.items():
        k = find_piece_file_key(piece, lab, idx)
        if k is None:
            miss.append(piece)
        else:
            piece2filenode[piece] = k
    print(f"[OK] piece-to-file-node matched: {len(piece2filenode)} / {len(labels)}")
    if miss:
        print("[WARN] some pieces did not match a file node; showing first 10")
        for p in miss[:10]:
            print("  -", p)

    # 3) aggregate nodes into piece vectors
    dim = kv.vector_size
    piece_vecs: Dict[str, np.ndarray] = {}
    stats_rows = []
    total_vec_nodes = 0

    for piece, fnode in piece2filenode.items():
        nodes = bfs_collect(g, fnode, max_hops)
        if not nodes:
            continue
        if not include_file_node:
            nodes.discard(fnode)

        # keep nodes with embeddings and pass filters
        picked = [n for n in nodes if n in kv and useful_node(n)]
        if len(picked) < args.min_nodes:
            stats_rows.append([piece, fnode, len(nodes), len(picked), "SKIP(<min)"])
            continue

        vecs = [kv[n] for n in picked]
        piece_vecs[piece] = np.mean(vecs, axis=0)
        total_vec_nodes += len(picked)
        stats_rows.append([piece, fnode, len(nodes), len(picked), "OK"])

    print(f"[OK] built piece vectors: {len(piece_vecs)} / {len(labels)}, collected vec-nodes total: {total_vec_nodes}")

    # 4) export word2vec text
    os.makedirs(os.path.dirname(args.out_piece_bin), exist_ok=True)
    export_word2vec_text(args.out_piece_bin, piece_vecs, dim)

    # 5) export stats CSV
    os.makedirs(os.path.dirname(args.out_debug_csv), exist_ok=True)
    with open(args.out_debug_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["piece", "file_node", "nodes_in_component", "nodes_with_vectors_after_filter", "status"])
        w.writerows(stats_rows)
    print(f"[OK] saved stats -> {args.out_debug_csv}")


if __name__ == "__main__":
    main()
