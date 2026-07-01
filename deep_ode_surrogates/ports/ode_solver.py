from typing import Protocol
import numpy as np
from deep_ode_surrogates.domain.odes.base import BaseODE
from deep_ode_surrogates.domain.schema import Trajectory


class ODESolverPort(Protocol):
    def solve(
        self,
        ode: BaseODE,
        x0: np.ndarray,
        t_span: tuple[float, float],
        n_steps: int,
        run_id: int | None = None,
    ) -> Trajectory:
        ...