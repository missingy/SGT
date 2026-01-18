import subprocess
from typing import Any, Dict


STEP_ID = "sgt"


def run(cfg: Dict[str, Any], logger) -> Dict[str, str]:
    step_cfg = cfg["steps"]["sgt"]
    script = step_cfg.get("script", "src/sgt_features.py")
    data_root = cfg["dataset"]["data_root"]

    cmd = [
        "python",
        script,
        "--data-root",
        data_root,
        "--out-features",
        step_cfg["out_features"],
        "--out-ids",
        step_cfg["out_ids"],
    ]

    if "alpha" in step_cfg:
        cmd.extend(["--alpha", str(step_cfg["alpha"])])
    if "max_dt_beats" in step_cfg:
        cmd.extend(["--max-dt-beats", str(step_cfg["max_dt_beats"])])
    if "token_mode" in step_cfg:
        cmd.extend(["--token-mode", str(step_cfg["token_mode"])])
    if "top_vocab" in step_cfg:
        cmd.extend(["--top-vocab", str(step_cfg["top_vocab"])])
    extra_args = step_cfg.get("extra_args")
    if extra_args:
        cmd.extend([str(x) for x in extra_args])

    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return {"sgt_features": step_cfg["out_features"], "sgt_ids": step_cfg["out_ids"]}
