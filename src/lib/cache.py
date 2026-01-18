import hashlib
import json
import os
from typing import Any, Dict, Iterable


def hash_dict(d: Dict[str, Any]) -> str:
    payload = json.dumps(d, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def outputs_exist(paths: Iterable[str]) -> bool:
    return all(p and os.path.exists(p) for p in paths)


def write_meta(path: str, meta: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=True, indent=2)
