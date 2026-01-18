import os
from typing import Dict


def ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def artifact_dir(root: str, dataset_id: str, step: str, run_id: str) -> str:
    return os.path.join(root, dataset_id, step, run_id)


def build_paths(root: str, dataset_id: str, step: str, run_id: str, files: Dict[str, str]) -> Dict[str, str]:
    base = artifact_dir(root, dataset_id, step, run_id)
    ensure_dir(base)
    out = {}
    for k, fname in files.items():
        out[k] = os.path.join(base, fname)
    return out
