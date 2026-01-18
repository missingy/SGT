import argparse
import os
import sys
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.cache import hash_dict, outputs_exist, write_meta
from lib.config import load_and_merge
from lib.logging import init_logger
from lib.paths import build_paths, ensure_dir
from lib.seed import set_seed

from steps import edgelist, embed, labels, piece_vectors, sgt, train


STEP_ORDER = ["edgelist", "embed", "labels", "piece_vectors", "sgt", "train"]


def _step_meta(step_name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "step": step_name,
        "dataset": cfg["dataset"],
        "step_cfg": cfg["steps"][step_name],
    }


def _step_outputs(step_name: str, cfg: Dict[str, Any], run_id: str) -> Dict[str, str]:
    root = cfg["paths"]["artifacts_root"]
    dataset_id = cfg["dataset"]["id"]
    if step_name == "edgelist":
        return {"out_dir": os.path.join(root, dataset_id, "edgelist", run_id)}
    if step_name == "embed":
        return build_paths(root, dataset_id, "embed", run_id, {"out_path": "embeddings.bin"})
    if step_name == "labels":
        return build_paths(root, dataset_id, "labels", run_id, {"out_csv": "labels.csv"})
    if step_name == "piece_vectors":
        return build_paths(
            root,
            dataset_id,
            "piece_vectors",
            run_id,
            {"out_piece_bin": "piece_vectors.bin", "out_debug_csv": "piece_aggregate_stats.csv"},
        )
    if step_name == "sgt":
        return build_paths(
            root, dataset_id, "sgt", run_id, {"out_features": "sgt_features.npy", "out_ids": "sgt_ids.npy"}
        )
    return {}


def _expected_outputs(step_name: str, cfg: Dict[str, Any]) -> List[str]:
    step_cfg = cfg["steps"][step_name]
    if step_name == "edgelist":
        base = step_cfg["out_dir"]
        return [
            os.path.join(base, "notes.edgelist"),
            os.path.join(base, "program.edgelist"),
            os.path.join(base, "tempo.edgelist"),
            os.path.join(base, "time.signature.edgelist"),
            os.path.join(base, "names.csv"),
        ]
    if step_name == "embed":
        return [step_cfg["out_path"]]
    if step_name == "labels":
        return [step_cfg["out_csv"]]
    if step_name == "piece_vectors":
        return [step_cfg["out_piece_bin"], step_cfg["out_debug_csv"]]
    if step_name == "sgt":
        return [step_cfg["out_features"], step_cfg["out_ids"]]
    return []


def _run_step(step_name: str, cfg: Dict[str, Any], logger) -> None:
    steps_map = {
        "edgelist": edgelist,
        "embed": embed,
        "labels": labels,
        "piece_vectors": piece_vectors,
        "sgt": sgt,
        "train": train,
    }
    steps_map[step_name].run(cfg, logger)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base config (yaml/json)")
    ap.add_argument("--dataset", required=True, help="Dataset config (yaml/json)")
    ap.add_argument("--experiment", required=True, help="Experiment config (yaml/json)")
    ap.add_argument("--resume", action="store_true", help="Skip steps with existing outputs")
    ap.add_argument("--skip", default="", help="Comma-separated step names to skip")
    ap.add_argument("--force", default="", help="Comma-separated step names to force rerun")
    args = ap.parse_args()

    cfg = load_and_merge(args.base, args.dataset, args.experiment)
    cfg.setdefault("paths", {})
    cfg.setdefault("dataset", {})
    cfg.setdefault("steps", {})

    dataset_id = cfg["dataset"].get("id") or os.path.splitext(os.path.basename(args.dataset))[0]
    cfg["dataset"]["id"] = dataset_id

    log_dir = cfg["paths"].get("logs_root", "logs")
    ensure_dir(log_dir)
    logger = init_logger("runner", os.path.join(log_dir, f"run_{dataset_id}.log"))

    seed = int(cfg.get("seed", 42))
    set_seed(seed, deterministic_tf=bool(cfg.get("deterministic_tf", False)))

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    force = {s.strip() for s in args.force.split(",") if s.strip()}

    # Prepare per-step outputs with dependency-aware run_id
    run_ids: Dict[str, str] = {}
    for step in STEP_ORDER:
        if step not in cfg["steps"]:
            cfg["steps"][step] = {}

        deps: Dict[str, str] = {}
        if step == "embed" and "edgelist" in run_ids:
            deps["edgelist"] = run_ids["edgelist"]
        if step == "piece_vectors":
            for k in ("edgelist", "embed", "labels"):
                if k in run_ids:
                    deps[k] = run_ids[k]
        if step == "train":
            for k in ("piece_vectors", "sgt", "labels"):
                if k in run_ids:
                    deps[k] = run_ids[k]

        meta = _step_meta(step, cfg)
        if deps:
            meta["deps"] = deps
        run_id = hash_dict(meta)
        run_ids[step] = run_id

        outputs = _step_outputs(step, cfg, run_id)
        cfg["steps"][step].update(outputs)

    # Wire dependencies
    cfg["steps"]["edgelist"]["script"] = cfg["steps"]["edgelist"].get(
        "script", "external/midi2vec/midi2edgelist/index.js"
    )
    cfg["steps"]["embed"]["edgelist_dir"] = cfg["steps"]["edgelist"]["out_dir"]
    cfg["steps"]["labels"]["out_csv"] = cfg["steps"]["labels"]["out_csv"]
    cfg["steps"]["piece_vectors"]["edgelist_dir"] = cfg["steps"]["edgelist"]["out_dir"]
    cfg["steps"]["piece_vectors"]["embeddings"] = cfg["steps"]["embed"]["out_path"]
    cfg["steps"]["piece_vectors"]["labels_csv"] = cfg["steps"]["labels"]["out_csv"]
    cfg["steps"]["sgt"]["out_features"] = cfg["steps"]["sgt"]["out_features"]
    cfg["steps"]["sgt"]["out_ids"] = cfg["steps"]["sgt"]["out_ids"]
    cfg["steps"]["train"]["piece_vec"] = cfg["steps"]["piece_vectors"]["out_piece_bin"]
    cfg["steps"]["train"]["labels_csv"] = cfg["steps"]["labels"]["out_csv"]
    cfg["steps"]["train"]["sgt_features"] = cfg["steps"]["sgt"]["out_features"]
    cfg["steps"]["train"]["sgt_ids"] = cfg["steps"]["sgt"]["out_ids"]
    cfg["steps"]["train"]["results_jsonl"] = cfg["paths"].get("results_jsonl", "results/results.jsonl")
    cfg["steps"]["train"]["results_csv"] = cfg["paths"].get("results_csv", "results/results.csv")
    cfg["steps"]["train"]["run_id"] = hash_dict({"dataset": dataset_id, "experiment": cfg.get("experiment", {})})
    cfg["steps"]["train"]["dataset_id"] = dataset_id

    for step in STEP_ORDER:
        if step in skip:
            logger.info("Skip step: %s", step)
            continue

        expected = _expected_outputs(step, cfg)
        if args.resume and step not in force and expected and outputs_exist(expected):
            logger.info("Resume: step %s already done", step)
            continue

        logger.info("=== Step: %s ===", step)
        _run_step(step, cfg, logger)

        step_cfg = cfg["steps"][step]
        if step == "edgelist":
            meta_dir = step_cfg["out_dir"]
        elif step == "embed":
            meta_dir = os.path.dirname(step_cfg["out_path"])
        elif step == "labels":
            meta_dir = os.path.dirname(step_cfg["out_csv"])
        elif step == "piece_vectors":
            meta_dir = os.path.dirname(step_cfg["out_piece_bin"])
        elif step == "sgt":
            meta_dir = os.path.dirname(step_cfg["out_features"])
        else:
            meta_dir = cfg["paths"].get("artifacts_root", "artifacts")

        write_meta(os.path.join(meta_dir, "meta.json"), _step_meta(step, cfg))


if __name__ == "__main__":
    main()
