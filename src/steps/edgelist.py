import subprocess
from typing import Any, Dict


STEP_ID = "edgelist"


def run(cfg: Dict[str, Any], logger) -> Dict[str, str]:
    step_cfg = cfg["steps"]["edgelist"]
    script = step_cfg.get("script", "external/midi2vec/midi2edgelist/index.js")
    data_root = cfg["dataset"]["data_root"]

    out_dir = step_cfg["out_dir"]
    cmd = ["node", script, "-i", data_root, "-o", out_dir]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return {"edgelist_dir": out_dir}
