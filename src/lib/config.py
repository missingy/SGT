import json
from copy import deepcopy
from typing import Any, Dict


def deep_update(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_config(path: str) -> Dict[str, Any]:
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if path.lower().endswith((".yml", ".yaml")):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("PyYAML is required to load YAML configs.") from e
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    raise ValueError(f"Unsupported config format: {path}")


def load_and_merge(*paths: str) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for p in paths:
        if not p:
            continue
        cfg = deep_update(cfg, load_config(p))
    return cfg
