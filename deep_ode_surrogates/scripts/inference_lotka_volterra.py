# scripts/infer_lotka_volterra.py

from pathlib import Path
import torch
import numpy as np

from deep_ode_surrogates.application.inference import InferSurrogateUseCase
from deep_ode_surrogates.infrastructure.persistence.checkpoints.model_loader import (
    load_model_from_checkpoint,
)
from deep_ode_surrogates.infrastructure.registries.model_registry import model_registry
from deep_ode_surrogates.infrastructure.registries.bootstrap import bootstrap
from deep_ode_surrogates.infrastructure.visualization.plotly.trajectory_plots import (
    plot_trajectory,
)
from deep_ode_surrogates.domain.models import AvailablesAIModel

bootstrap()

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = Path("runs\lotka_volterra\experiment_2026-07-01_20-12-10\save\epoch_19_loss_0.122524.pt")

t_span = (0.0, 10.0)
n_steps = 200

t = torch.linspace(
    t_span[0],
    t_span[1],
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
    t=t.cpu().numpy().ravel(),
    y=y_pred.numpy(),
    state_names=["x0", "x1"],
    title="Lotka-Volterra surrogate inference",
)

fig.show()