import numpy as np

from deep_ode_surrogates.domain.odes.base import BaseODE
from deep_ode_surrogates.domain.schema import Trajectory


class ScipyODESolver:
    def solve(
        self,
        ode: BaseODE,
        x0: np.ndarray,
        params: dict,
        t_span: tuple[float, float],
        n_steps: int,
        run_id: int | None = None,
    ) -> Trajectory:
        sol = ode.simulate(
            t_span=t_span,
            x0=x0,
            nb_points=n_steps,
        )

        return Trajectory(
            t=sol.t,
            y=sol.y.T,
            params=params,
            run_id=run_id,
        )