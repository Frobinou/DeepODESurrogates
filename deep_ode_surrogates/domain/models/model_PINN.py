import torch.nn as nn
from deep_ode_surrogates.domain.models import AvailablesAIModel
from deep_ode_surrogates.infrastructure.registries.model_registry import register_model 

@register_model(AvailablesAIModel.BASIC_PINN)
class BasicPINN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=20, output_dim=1):
        super(BasicPINN, self).__init__()
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
