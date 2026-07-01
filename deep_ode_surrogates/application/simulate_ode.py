import numpy as np
from deep_ode_surrogates.domain.odes.base import BaseODE
from deep_ode_surrogates.domain.schema import Trajectory
from deep_ode_surrogates.ports.ode_solver import ODESolverPort


class SimulateODEUseCase:
    def __init__(self, solver: ODESolverPort):
        self.solver = solver

    def execute(
        self,
        ode: BaseODE,
        x0: np.ndarray,
        t_span: tuple[float, float],
        n_steps: int,
        run_id: int | None = None,
    ) -> Trajectory:
        return self.solver.solve(
            ode=ode,
            x0=x0,
            t_span=t_span,
            n_steps=n_steps,
            run_id=run_id,
        )