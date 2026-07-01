import torch
import torch.nn as nn

from deep_ode_surrogates.domain.models import AvailablesAIModel
from deep_ode_surrogates.infrastructure.registries.model_registry import register_model


@register_model(AvailablesAIModel.DIFFUSION_SCORENET)
class ScoreNet(nn.Module):
    """
    Réseau de score utilisé dans un modèle de diffusion pour prédire le bruit ε
    ajouté à une trajectoire x₀ à un instant de diffusion t.
    """

    def __init__(self, traj_len: int = 100, hidden: int = 256) -> None:
        super().__init__()

        self.time_emb = nn.Embedding(1000, 32)

        self.net = nn.Sequential(
            nn.Linear(traj_len + 32, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, traj_len),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        h = torch.cat([x_t, t_emb], dim=-1)
        return self.net(h)


class ConditionalScoreNet(nn.Module):
    def __init__(
        self,
        traj_len: int = 100,
        cond_dim: int = 3,
        hidden: int = 256,
    ) -> None:
        super().__init__()

        self.time_emb = nn.Embedding(1000, 32)

        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
        )

        self.net = nn.Sequential(
            nn.Linear(traj_len + 32 + 32, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, traj_len),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.time_emb(t)
        c_emb = self.cond_encoder(c)

        h = torch.cat([x_t, t_emb, c_emb], dim=-1)

        return self.net(h)
