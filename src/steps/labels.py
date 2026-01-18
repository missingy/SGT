import subprocess
from typing import Any, Dict


STEP_ID = "labels"


def run(cfg: Dict[str, Any], logger) -> Dict[str, str]:
    step_cfg = cfg["steps"]["labels"]
    script = step_cfg.get("script", "src/make_XMIDI_labels.py")
    data_root = cfg["dataset"]["data_root"]
    out_csv = step_cfg["out_csv"]

    cmd = ["python", script, "--data-root", data_root, "--out-csv", out_csv]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return {"labels_csv": out_csv}
