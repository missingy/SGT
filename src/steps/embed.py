import subprocess
from typing import Any, Dict


STEP_ID = "embed"


def run(cfg: Dict[str, Any], logger) -> Dict[str, str]:
    step_cfg = cfg["steps"]["embed"]
    script = step_cfg.get("script", "external/midi2vec/edgelist2vec/embed.py")
    edgelist_dir = step_cfg["edgelist_dir"]
    out_path = step_cfg["out_path"]

    cmd = [
        "python",
        script,
        "-i",
        edgelist_dir,
        "-o",
        out_path,
        "--walk_length",
        str(step_cfg.get("walk_length", 10)),
        "--num_walks",
        str(step_cfg.get("num_walks", 40)),
        "-p",
        str(step_cfg.get("p", 1)),
        "-q",
        str(step_cfg.get("q", 1)),
        "--dimensions",
        str(step_cfg.get("dimensions", 100)),
        "--window-size",
        str(step_cfg.get("window_size", 5)),
        "--iter",
        str(step_cfg.get("iter", 5)),
        "--workers",
        str(step_cfg.get("workers", 0)),
    ]
    exclude = step_cfg.get("exclude")
    if exclude:
        cmd.append("--exclude")
        cmd.extend([str(x) for x in exclude])

    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return {"embeddings_path": out_path}
