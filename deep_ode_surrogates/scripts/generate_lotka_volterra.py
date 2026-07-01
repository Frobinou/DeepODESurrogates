from pathlib import Path

from deep_ode_surrogates.application.simulate_ode import SimulateODEUseCase
from deep_ode_surrogates.application.generate_ode_dataset import GenerateODEDatasetUseCase
from deep_ode_surrogates.infrastructure.solvers.scipy_ode_solver import ScipyODESolver
from deep_ode_surrogates.infrastructure.data.parquet.parquet_dataset_writer import ParquetDatasetWriter
from deep_ode_surrogates.application.config.ode import AvailablesODE
from deep_ode_surrogates.domain.odes.ode_lotka_voltera import ParamsLotkaVolterra
from deep_ode_surrogates.infrastructure.registries.ode_registry import ode_registry
from deep_ode_surrogates.infrastructure.visualization.plotly.trajectory_plots import plot_phase_space, plot_trajectory

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
    x0=[10., 1.],
    n_steps=200,
    t_span=(0.0, 10.0),
)

fig = plot_trajectory(df['t'], y=df[['prey', 'predator']].values, state_names=["Prey", "Predator"], title="Lotka-Volterra Trajectory")
fig.show()

fig_phase = plot_phase_space(y=df[['prey', 'predator']].values, 
                             x_idx=0,
                             y_idx=1,
                             state_names=["Prey", "Predator"], 
                             title="Lotka-Volterra Phase Space")
fig_phase.show()

path = Path("data/generated_dataset_LV.parquet")
writer = ParquetDatasetWriter()
writer.save(df, path)