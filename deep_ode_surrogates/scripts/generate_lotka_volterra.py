import numpy as np 
from pathlib import Path

from deep_ode_surrogates.application.simulate_ode import SimulateODEUseCase
from deep_ode_surrogates.application.generate_ode_dataset import GenerateODEDatasetUseCase
from deep_ode_surrogates.infrastructure.solvers.scipy_ode_solver import ScipyODESolver
from deep_ode_surrogates.infrastructure.persistence.parquet_dataset_writer import ParquetDatasetWriter
from deep_ode_surrogates.application.config.enums import AvailablesODE
from deep_ode_surrogates.domain.odes.ode_lotka_voltera import ParamsLotkaVoltera
from deep_ode_surrogates.infrastructure.registries.ode_registry import ode_registry

solver = ScipyODESolver()
simulator = SimulateODEUseCase(solver)

generator = GenerateODEDatasetUseCase(
    simulator=simulator,
    target_cols=["prey", "predator"],
)

params = ParamsLotkaVoltera(alpha=1.0, beta=0.1, delta=0.075, gamma=1.5)
ode = ode_registry.create(AvailablesODE.LOTKA_VOLTERA, params)

df = generator.execute(
    ode=ode,
    n_sims=1,
    x0=[10., 1.],
)

path = Path("data/generated_dataset_LV.parquet")
writer = ParquetDatasetWriter()
writer.save(df, path)