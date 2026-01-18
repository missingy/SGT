import os
import random
from typing import Optional


def set_seed(seed: int, deterministic_tf: bool = False) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import tensorflow as tf  # type: ignore
        tf.random.set_seed(seed)
        if deterministic_tf:
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except Exception:
        pass
