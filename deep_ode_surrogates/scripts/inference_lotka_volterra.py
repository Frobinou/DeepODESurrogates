# scripts/infer_lotka_volterra.py

from pathlib import Path

import torch

from deep_ode_surrogates.application.inference import InferSurrogateUseCase
from deep_ode_surrogates.domain.models import AvailablesAIModel
from deep_ode_surrogates.infrastructure.persistence.checkpoints.model_loader import (
    load_model_from_checkpoint,
)
from deep_ode_surrogates.infrastructure.registries.bootstrap import bootstrap
from deep_ode_surrogates.infrastructure.registries.model_registry import model_registry
from deep_ode_surrogates.infrastructure.training.torch.normalizer import TimeNormalizer
from deep_ode_surrogates.infrastructure.visualization.plotly.trajectory_plots import (
    plot_trajectory,
)

bootstrap()

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = Path(
    r"runs\lotka_volterra\experiment_2026-07-03_15-57-24\save\epoch_1999_loss_7.523561.pt"
)

t_span = (0.0, 10.0)
n_steps = 200
time_normalize = TimeNormalizer(t_min=t_span[0], t_max=t_span[1])
t = torch.linspace(
    -1,
    1,
    n_steps,
).reshape(-1, 1)

model = model_registry.create(
    AvailablesAIModel.BASIC_PINN,
    input_dim=1,
    output_dim=2,
)

model = load_model_from_checkpoint(
    model=model,
    checkpoint_path=checkpoint_path,
    device=device,
)

y_pred = InferSurrogateUseCase().execute(
    model=model,
    t=t,
    device=device,
)

fig = plot_trajectory(
    t=t.cpu().numpy().ravel() * time_normalize.dtau_dt,
    y=y_pred.numpy(),
    state_names=["x0", "x1"],
    title="Lotka-Volterra surrogate inference",
)

fig.show()
