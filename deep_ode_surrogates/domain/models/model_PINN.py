import torch.nn as nn

from deep_ode_surrogates.domain.models import AvailablesAIModel
from deep_ode_surrogates.infrastructure.registries.model_registry import register_model


@register_model(AvailablesAIModel.BASIC_PINN)
class BasicPINN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=20, output_dim=1):
        super().__init__()
        self.name = AvailablesAIModel.BASIC_PINN
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


"""

class BasicPINN(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 2,
        hidden_dim: int = 64,
        num_hidden_layers: int = 4,
        activation: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

"""
