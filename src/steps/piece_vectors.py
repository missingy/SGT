import subprocess
from typing import Any, Dict


STEP_ID = "piece_vectors"


def run(cfg: Dict[str, Any], logger) -> Dict[str, str]:
    step_cfg = cfg["steps"]["piece_vectors"]
    script = step_cfg.get("script", "src/piece_vectors.py")

    cmd = [
        "python",
        script,
        "--embeddings",
        step_cfg["embeddings"],
        "--labels",
        step_cfg["labels_csv"],
        "--edgelist-dir",
        step_cfg["edgelist_dir"],
        "--out-piece-bin",
        step_cfg["out_piece_bin"],
        "--out-debug-csv",
        step_cfg["out_debug_csv"],
        "--max-hops",
        str(step_cfg.get("max_hops", 1)),
        "--min-nodes",
        str(step_cfg.get("min_nodes", 5)),
    ]
    include_file_node = step_cfg.get("include_file_node")
    if include_file_node is True:
        cmd.append("--include-file-node")
    if include_file_node is False:
        cmd.append("--exclude-file-node")

    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return {
        "piece_vectors": step_cfg["out_piece_bin"],
        "piece_debug_csv": step_cfg["out_debug_csv"],
    }
