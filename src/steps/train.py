import subprocess
from typing import Any, Dict


STEP_ID = "train"


def run(cfg: Dict[str, Any], logger) -> Dict[str, str]:
    step_cfg = cfg["steps"]["train"]
    script = step_cfg.get("script", "src/mlp_kfold.py")

    cmd = [
        "python",
        script,
        "--piece-vec",
        step_cfg["piece_vec"],
        "--labels",
        step_cfg["labels_csv"],
        "--sgt-feat",
        step_cfg["sgt_features"],
        "--task",
        step_cfg.get("task", "both"),
        "--n-splits",
        str(step_cfg.get("n_splits", 5)),
        "--seed",
        str(step_cfg.get("seed", 42)),
        "--epochs",
        str(step_cfg.get("epochs", 50)),
        "--batch-size",
        str(step_cfg.get("batch_size", 32)),
        "--lr",
        str(step_cfg.get("lr", 1e-3)),
        "--val-split",
        str(step_cfg.get("val_split", 0.1)),
    ]

    if step_cfg.get("results_jsonl"):
        cmd.extend(["--results-jsonl", step_cfg["results_jsonl"]])
    if step_cfg.get("results_csv"):
        cmd.extend(["--results-csv", step_cfg["results_csv"]])
    if step_cfg.get("run_id"):
        cmd.extend(["--run-id", step_cfg["run_id"]])
    if step_cfg.get("dataset_id"):
        cmd.extend(["--dataset-id", step_cfg["dataset_id"]])
    extra_args = step_cfg.get("extra_args")
    if extra_args:
        cmd.extend([str(x) for x in extra_args])

    sgt_ids = step_cfg.get("sgt_ids")
    if sgt_ids:
        cmd.extend(["--sgt-ids", sgt_ids])

    clean_cmd = [c for c in cmd if c]
    logger.info("Running: %s", " ".join(clean_cmd))
    subprocess.run(clean_cmd, check=True)
    return {}
