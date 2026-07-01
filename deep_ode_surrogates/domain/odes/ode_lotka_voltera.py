import torch
import numpy as np
from pydantic import BaseModel
import matplotlib.pyplot as plt
from io import BytesIO

from deep_ode_surrogates.domain.odes.ode_base import BaseODE
from deep_ode_surrogates.olddddd.repositories.odes.visualizers.base_visualizer import VisualizationMixin
from deep_ode_surrogates.infrastructure.registries.ode_registry import register_ode
from deep_ode_surrogates.application.config.enums import AvailablesODE


class ParamsLotkaVoltera(BaseModel):
    alpha: float = 1.0
    beta: float = 2.0
    delta: float = 1.0
    gamma: float = 2.0

@register_ode(AvailablesODE.LOTKA_VOLTERA)
class LotkaVolteraODE(BaseODE, VisualizationMixin):
    def __init__(self, params: ParamsLotkaVoltera):
        self.name = "Lotka Volterra"
        self.params = params

    def update_params(self, new_params: ParamsLotkaVoltera):
        self.params = new_params

    # ---------- Core dynamics (factorisée) ----------
    def _dynamics(self, x, t=None):
        dx = self.params.alpha * x[:, 0] - self.params.beta * x[:, 0] * x[:, 1]
        dy = self.params.delta * x[:, 0] * x[:, 1] - self.params.gamma * x[:, 1]
        return dx, dy

    # ---------- Torch version ----------
    def torch_ode(self, x: torch.Tensor) -> torch.Tensor:
        dx, dy = self._dynamics(x)
        return torch.stack((dx, dy), dim=1)

    # ---------- Numpy / SciPy version ----------
    def _dynamics_numpy(self, t, x):
        dx = self.params.alpha * x[0] - self.params.beta * x[0] * x[1]
        dy = self.params.delta * x[0] * x[1] - self.params.gamma * x[1]
        return [dx, dy]

    # ---------- Validation ----------
    def log_trajectory_plot(self, t_true, y_true, y_pred):
        fig, ax = plt.subplots()
        ax.plot(t_true, y_true[:, 0], label="Proies (SciPy)")
        ax.plot(t_true, y_true[:, 1], label="Prédateurs (SciPy)")
        ax.plot(t_true, y_pred[:, 0], "--", label="Proies (NN)")
        ax.plot(t_true, y_pred[:, 1], "--", label="Prédateurs (NN)")

        ax.legend()
        ax.set_xlabel("t")
        ax.set_ylabel("Population")

        return self.fig_to_tensor(fig)

    # ---------- Simple visualization ----------
    def log_trajectory_phase_space_plot(self, y_true, y_pred):
        fig, ax = plt.subplots()
        ax.plot(y_true[:, 0], y_true[:, 1], label="Proies / Prédateurs (SciPy)")
        ax.plot(y_pred[:, 0], y_pred[:, 1], "--", label="Proies / Prédateurs (NN)")

        ax.legend()
        ax.set_xlabel("Proies")
        ax.set_ylabel("Prédateurs")

        return self.fig_to_tensor(fig)
