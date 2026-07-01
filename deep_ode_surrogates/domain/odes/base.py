from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel
from scipy.integrate import solve_ivp


class BaseODE(ABC):
    name: str = "base_ode"

    def __init__(self, params: BaseModel):
        self.params = params

    def update_params(self, new_params: BaseModel) -> None:
        self.params = new_params

    @abstractmethod
    def dynamics(self, t: float, x: Any) -> Any:
        """
        Defines dx/dt = f(t, x).

        Must support:
        - x as np.ndarray with shape (n_states,)
        - x as torch.Tensor with shape (batch_size, n_states)
        """
        ...

    def scipy_rhs(self, t: float, x: np.ndarray) -> np.ndarray:
        dx = self.dynamics(t, x)
        return np.asarray(dx, dtype=float)

    def torch_ode(self, x: torch.Tensor, t: float = 0.0) -> torch.Tensor:
        dx = self.dynamics(t, x)

        if isinstance(dx, torch.Tensor):
            return dx

        return torch.stack(dx, dim=1)

