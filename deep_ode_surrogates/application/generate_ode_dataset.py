import numpy as np
import pandas as pd

from deep_ode_surrogates.domain.odes.base import BaseODE
from deep_ode_surrogates.application.simulate_ode import SimulateODEUseCase


class GenerateODEDatasetUseCase:
    def __init__(
        self,
        simulator: SimulateODEUseCase,
        target_cols: list[str],
    ):
        self.simulator = simulator
        self.target_cols = target_cols

    def execute(
        self,
        ode: BaseODE,
        x0: np.ndarray | list | None = None,
        t_span: tuple[float, float] = (0, 10),
        n_steps: int = 200,
        n_sims: int = 100,
        x0_sampler=None,
        param_sampler=None,
        seed: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)

        rows = []

        base_params = ode.params
        single_run = x0_sampler is None and param_sampler is None
        n_runs = 1 if single_run else n_sims

        for run_id in range(n_runs):
            current_params = (
                param_sampler(rng)
                if param_sampler is not None
                else base_params
            )

            current_x0 = (
                x0_sampler(rng)
                if x0_sampler is not None
                else x0
            )

            if current_x0 is None:
                current_x0 = np.ones(len(self.target_cols))

            ode.update_params(current_params)

            trajectory = self.simulator.execute(
                ode=ode,
                x0=np.asarray(current_x0, dtype=float),
                params=current_params.model_dump() if hasattr(current_params, "model_dump") else dict(current_params),
                t_span=t_span,
                n_steps=n_steps,
                run_id=None if single_run else run_id,
            )

            df = pd.DataFrame(trajectory.y, columns=self.target_cols)
            df["t"] = trajectory.t

            params_dict = (
                current_params.model_dump()
                if hasattr(current_params, "model_dump")
                else dict(current_params)
            )

            for key, value in params_dict.items():
                df[key] = value

            if trajectory.run_id is not None:
                df["run_id"] = trajectory.run_id

            rows.append(df)

        return pd.concat(rows, ignore_index=True)