from src.repositories.models import AvailablesAIModel
from src.core.registry import REGISTRY  

import torch
import torch.nn as nn

@REGISTRY.models.register(AvailablesAIModel.DIFFUSION_SCORENET)
class ScoreNet(nn.Module):
    """
    Réseau de score utilisé dans un modèle de diffusion pour prédire le bruit ε
    ajouté à une trajectoire x₀ à un instant de diffusion t.
    """
    def __init__(self, traj_len=100, hidden=256):
        super().__init__()

        # Embedding discret du pas temporel de diffusion
        self.time_emb = nn.Embedding(1000, 32)

        # MLP principal
        self.net = nn.Sequential(
            nn.Linear(traj_len + 32, hidden),
            nn.SiLU(),

            nn.Linear(hidden, hidden),
            nn.SiLU(),

            # Sortie : estimation du bruit ε
            nn.Linear(hidden, traj_len)
        )

    def forward(self, x_t, t):
        # Encodage temporel
        t_emb = self.time_emb(t)

        # Fusion trajectoire bruitée + embedding temporel
        h = torch.cat([x_t, t_emb], dim=-1)

        # Prédiction du bruit
        return self.net(h)


# -----------------------------
# Boucle d'entraînement
# -----------------------------

model = ScoreNet(traj_len=100)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for x0 in dataloader:
    """
    x0 : batch de trajectoires propres issues d’un système dynamique
    """

    # Échantillonnage aléatoire du niveau de bruit
    t = torch.randint(1, 1000, (x0.shape[0],))

    # Bruit gaussien cible
    eps = torch.randn_like(x0)

    # Coefficient ᾱ_t du processus de diffusion
    alpha_bar_t = alpha_bar[t].unsqueeze(-1)

    # Construction de la trajectoire bruitée x_t
    x_t = (
        torch.sqrt(alpha_bar_t) * x0 +
        torch.sqrt(1 - alpha_bar_t) * eps
    )

    # Prédiction du bruit par le réseau
    eps_pred = model(x_t, t)

    # Perte MSE entre bruit réel et bruit prédit
    loss = ((eps - eps_pred) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



class ConditionalScoreNet(nn.Module):

    def __init__(
        self,
        traj_len=100,
        cond_dim=3,
        hidden=256
    ):
        super().__init__()

        self.time_emb = nn.Embedding(1000, 32)

        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 32)
        )

        self.net = nn.Sequential(
            nn.Linear(traj_len + 32 + 32, hidden),
            nn.SiLU(),

            nn.Linear(hidden, hidden),
            nn.SiLU(),

            nn.Linear(hidden, traj_len)
        )

    def forward(self, x_t, t, c):

        t_emb = self.time_emb(t)

        c_emb = self.cond_encoder(c)

        h = torch.cat([x_t, t_emb, c_emb], dim=-1)

        return self.net(h)