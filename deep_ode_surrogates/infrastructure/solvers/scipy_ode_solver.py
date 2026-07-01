import numpy as np
from scipy.integrate import solve_ivp

from deep_ode_surrogates.domain.odes.base import BaseODE
from deep_ode_surrogates.domain.schema import Trajectory


class ScipyODESolver:
    def solve(
        self,
        ode: BaseODE,
        x0,
        t_span: tuple[float, float],
        n_steps: int,
        run_id: int | None = None,
        method: str = "Radau",
        max_step: float = np.inf,
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ) -> Trajectory:
        t_eval = np.linspace(*t_span, n_steps)

        sol = solve_ivp(
            fun=ode.dynamics,
            t_span=t_span,
            y0=np.asarray(x0, dtype=float),
            t_eval=t_eval,
            method=method,
            max_step=max_step,
            rtol=rtol,
            atol=atol,
        )

        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")

        return Trajectory(
            t=sol.t,
            y=sol.y.T,
            params=ode.params,
            run_id=run_id,
        )
