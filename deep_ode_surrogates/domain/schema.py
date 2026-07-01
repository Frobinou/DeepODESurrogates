from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Trajectory:
    t: np.ndarray
    y: np.ndarray
    params: dict
    run_id: int | None = None