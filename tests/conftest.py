# tests/conftest.py

from pathlib import Path

import pandas as pd
import pytest

from deep_ode_surrogates.application.config.ode import AvailablesODE
from deep_ode_surrogates.application.generate_ode_dataset import GenerateODEDatasetUseCase
from deep_ode_surrogates.application.simulate_ode import SimulateODEUseCase
from deep_ode_surrogates.domain.odes.ode_lotka_voltera import ParamsLotkaVolterra
from deep_ode_surrogates.infrastructure.registries.ode_registry import ode_registry
from deep_ode_surrogates.infrastructure.solvers.scipy_ode_solver import ScipyODESolver


@pytest.fixture
def tmp_path():
    return Path("tests") / "units" / "tmp"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "trajectory_id": [0, 0, 0, 1, 1, 1],
            "t": [0.0, 0.1, 0.2, 0.0, 0.1, 0.2],
            "prey": [10, 11, 12, 20, 21, 22],
            "predator": [5, 6, 7, 8, 9, 10],
        }
    )
    return df


@pytest.fixture
def generated_dataframe() -> pd.DataFrame:
    solver = ScipyODESolver()
    simulator = SimulateODEUseCase(solver)

    generator = GenerateODEDatasetUseCase(
        simulator=simulator,
        target_cols=["prey", "predator"],
    )

    params = ParamsLotkaVolterra(alpha=1.0, beta=0.1, delta=0.075, gamma=1.5)
    ode = ode_registry.create(AvailablesODE.LOTKA_VOLTERA, params)

    df = generator.execute(
        ode=ode,
        n_sims=1,
        x0=[10.0, 1.0],
        n_steps=5,
        t_span=(0.0, 10.0),
    )
    return df
