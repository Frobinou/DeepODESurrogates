from typing import Any

import torch
from pydantic import BaseModel

from deep_ode_surrogates.domain.odes import AvailablesODE
from deep_ode_surrogates.domain.odes.base import BaseODE
from deep_ode_surrogates.infrastructure.registries.ode_registry import register_ode


class ParamsLotkaVolterra(BaseModel):
    alpha: float = 1.0
    beta: float = 2.0
    delta: float = 1.0
    gamma: float = 2.0


@register_ode(AvailablesODE.LOTKA_VOLTERA)
class LotkaVolteraODE(BaseODE):
    def __init__(self, params: ParamsLotkaVolterra):
        self.name = AvailablesODE.LOTKA_VOLTERA
        self.params = params
        self.state_names = ["Prey", "Predator"]

    def update_params(self, new_params: ParamsLotkaVolterra):
        self.params = new_params

    def dynamics(self, t: float, x: Any) -> Any:
        p = self.params

        if isinstance(x, torch.Tensor):
            prey = x[:, 0]
            predator = x[:, 1]

            dprey = p.alpha * prey - p.beta * prey * predator
            dpredator = p.delta * prey * predator - p.gamma * predator

            return torch.stack((dprey, dpredator), dim=1)

        prey, predator = x

        dprey = p.alpha * prey - p.beta * prey * predator
        dpredator = p.delta * prey * predator - p.gamma * predator

        return [dprey, dpredator]
