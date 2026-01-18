import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.cache import hash_dict, outputs_exist
from lib.config import load_and_merge
from lib.logging import init_logger
from lib.paths import build_paths
from steps import edgelist, embed, labels, piece_vectors, sgt, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_and_merge(args.base, args.dataset, args.experiment)
    dataset_id = cfg["dataset"].get("id") or os.path.splitext(os.path.basename(args.dataset))[0]
    cfg["dataset"]["id"] = dataset_id

    logger = init_logger("runner", os.path.join(cfg["paths"].get("logs_root", "logs"), f"run_{dataset_id}.log"))

    steps_map = {
        "edgelist": edgelist,
        "embed": embed,
        "labels": labels,
        "piece_vectors": piece_vectors,
        "sgt": sgt,
        "train": train,
    }

    if args.step not in steps_map:
        raise SystemExit(f"Unknown step: {args.step}")

    run_id = hash_dict({"dataset": cfg["dataset"], "step": cfg["steps"].get(args.step, {})})
    if args.step == "edgelist":
        cfg["steps"].setdefault("edgelist", {})
        cfg["steps"]["edgelist"]["out_dir"] = os.path.join(cfg["paths"]["artifacts_root"], dataset_id, "edgelist", run_id)
    elif args.step == "embed":
        cfg["steps"].setdefault("embed", {})
        cfg["steps"]["embed"].update(build_paths(cfg["paths"]["artifacts_root"], dataset_id, "embed", run_id, {"out_path": "embeddings.bin"}))
    elif args.step == "labels":
        cfg["steps"].setdefault("labels", {})
        cfg["steps"]["labels"].update(build_paths(cfg["paths"]["artifacts_root"], dataset_id, "labels", run_id, {"out_csv": "labels.csv"}))
    elif args.step == "piece_vectors":
        cfg["steps"].setdefault("piece_vectors", {})
        cfg["steps"]["piece_vectors"].update(
            build_paths(cfg["paths"]["artifacts_root"], dataset_id, "piece_vectors", run_id, {"out_piece_bin": "piece_vectors.bin", "out_debug_csv": "piece_aggregate_stats.csv"})
        )
    elif args.step == "sgt":
        cfg["steps"].setdefault("sgt", {})
        cfg["steps"]["sgt"].update(build_paths(cfg["paths"]["artifacts_root"], dataset_id, "sgt", run_id, {"out_features": "sgt_features.npy", "out_ids": "sgt_ids.npy"}))

    expected = []
    if args.step == "edgelist":
        base = cfg["steps"]["edgelist"]["out_dir"]
        expected = [
            os.path.join(base, "notes.edgelist"),
            os.path.join(base, "program.edgelist"),
            os.path.join(base, "tempo.edgelist"),
            os.path.join(base, "time.signature.edgelist"),
            os.path.join(base, "names.csv"),
        ]
    elif args.step == "embed":
        expected = [cfg["steps"]["embed"]["out_path"]]
    elif args.step == "labels":
        expected = [cfg["steps"]["labels"]["out_csv"]]
    elif args.step == "piece_vectors":
        expected = [cfg["steps"]["piece_vectors"]["out_piece_bin"], cfg["steps"]["piece_vectors"]["out_debug_csv"]]
    elif args.step == "sgt":
        expected = [cfg["steps"]["sgt"]["out_features"], cfg["steps"]["sgt"]["out_ids"]]

    if expected and outputs_exist(expected) and not args.force:
        logger.info("Step already done: %s", args.step)
        return

    steps_map[args.step].run(cfg, logger)


if __name__ == "__main__":
    main()
