# Simulation d'ODE par processus de diffusion

## Vue d'ensemble

Trois approches permettent d'utiliser l'IA pour simuler des ODE. Chacune exploite un paradigme différent : le **Neural ODE** paramétrise directement la dérivée, le **Score-based DDPM** modélise la distribution des trajectoires par débruitage itératif, et la **Probability Flow ODE** reformule le débruitage comme une ODE déterministe.

```
x(t) ODE cible
      │
      ├─ Neural ODE          → apprend dx/dt = fθ(x,t)
      ├─ Score-based DDPM    → apprend la distribution p(x₀)
      └─ Probability Flow    → débruitage = ODE déterministe
```

---

## Approche 1 — Neural ODE

### Principe

Le réseau neuronal **paramétrise directement la dérivée** de l'état. On intègre ensuite numériquement pour obtenir la trajectoire.

$$\frac{dx}{dt} = f_\theta(x, t), \quad x(t_0) = x_0$$

L'intégration est différentiable grâce à la **méthode de l'adjoint**, ce qui permet de rétropropager le gradient sans stocker toutes les étapes intermédiaires.

### Architecture

```
Entrée : (x, t)   — état courant + temps continu
Réseau : MLP ou ResNet avec time embedding sinusoïdal
Sortie : dx/dt    — même dimension que x
```

### Exemple minimal PyTorch (`torchdiffeq`)

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

class ODEFunc(nn.Module):
    """Paramétrise dx/dt = fθ(x, t)"""
    def __init__(self, dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, dim)
        )

    def forward(self, t, x):
        # t : scalaire, x : (B, dim)
        t_vec = t.expand(x.shape[0], 1)
        return self.net(torch.cat([x, t_vec], dim=-1))


# Entraînement
func = ODEFunc(dim=2)
optimizer = torch.optim.Adam(func.parameters(), lr=1e-3)

for x0, x_target, t_span in dataloader:
    # x_target : (B, T_obs, dim) — trajectoire observée
    t_eval = torch.linspace(0, 1, x_target.shape[1])
    x_pred = odeint(func, x0, t_eval, method='dopri5')  # (T, B, dim)
    loss = ((x_pred.permute(1,0,2) - x_target) ** 2).mean()
    loss.backward(); optimizer.step(); optimizer.zero_grad()


# Inférence
@torch.no_grad()
def simulate(func, x0, t_span):
    return odeint(func, x0, t_span, method='dopri5')
```

### Avantages / limites

| ✓ Avantages | ✗ Limites |
|---|---|
| Peu de données nécessaires | Pas d'incertitude sur la trajectoire |
| Inférence rapide (1 passe) | Peut mal extrapoler hors distribution |
| Solveur adaptatif (précision variable) | Instable pour dynamiques chaotiques |
| Temps continu natif | Requiert des trajectoires labellisées |

---

## Approche 2 — Score-based DDPM

### Principe

Le **DDPM (Denoising Diffusion Probabilistic Model)** appliqué aux ODE consiste à traiter les **trajectoires** `x(t)` comme des données à modéliser. Le modèle apprend la distribution des solutions, puis peut en générer de nouvelles par débruitage itératif.

---

## 1. Processus forward (bruitage)

On détruit progressivement la trajectoire en ajoutant du bruit gaussien à chaque étape `t` :

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\, \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t I\right)$$

**Raccourci direct** (sans récursion) :

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

avec $\bar{\alpha}_t = \prod_{i=1}^{t}(1-\beta_i)$.

| t | $\bar{\alpha}_t$ | Contenu |
|---|---|---|
| 0 | 1.0 | Trajectoire ODE propre |
| T/2 | ~0.3 | Signal + bruit important |
| T | ~0.0 | Bruit gaussien pur |

---

## 2. Processus reverse (débruitage)

Le réseau $\varepsilon_\theta$ apprend à inverser le bruitage :

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left(x_{t-1};\, \mu_\theta(x_t, t),\, \sigma_t^2 I\right)$$

avec :

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\varepsilon_\theta(x_t, t)\right)$$

---

## 3. Fonction de perte

$$\mathcal{L} = \mathbb{E}_{t,\, x_0,\, \varepsilon}\!\left[\|\varepsilon - \varepsilon_\theta(\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon,\; t)\|^2\right]$$

Le réseau prédit simplement **le bruit ajouté**, pas directement la trajectoire.

---

## 4. Architecture du réseau (pour ODE)

```
Entrée : (x_t, t)
         x_t  → vecteur de la trajectoire discrétisée (longueur L)
         t    → entier ∈ [1, T], encodé via embedding

Réseau : MLP ou U-Net 1D
         - Time embedding  : sinusoïdal ou nn.Embedding
         - Couches cachées : Linear + SiLU (ou attention si trajectoire longue)

Sortie : ε̂ ∈ ℝᴸ  (bruit estimé, même dimension que x_t)
```

### Exemple minimal PyTorch

```python
import torch
import torch.nn as nn

class ScoreNet(nn.Module):
    def __init__(self, traj_len=100, hidden=256):
        super().__init__()
        self.time_emb = nn.Embedding(1000, 32)
        self.net = nn.Sequential(
            nn.Linear(traj_len + 32, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),        nn.SiLU(),
            nn.Linear(hidden, traj_len)
        )

    def forward(self, x_t, t):
        return self.net(torch.cat([x_t, self.time_emb(t)], dim=-1))


# Boucle d'entraînement
model = ScoreNet(traj_len=100)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
alpha_bar = ...  # tenseur de taille T+1

for x0 in dataloader:                         # x0 : (B, L) trajectoires ODE
    t   = torch.randint(1, 1000, (x0.shape[0],))
    eps = torch.randn_like(x0)
    ab  = alpha_bar[t].unsqueeze(-1)
    x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps
    loss = ((eps - model(x_t, t)) ** 2).mean()
    loss.backward(); optimizer.step(); optimizer.zero_grad()
```

---

## 5. Inférence (simulation)

```python
@torch.no_grad()
def sample(model, T=1000, traj_len=100):
    x = torch.randn(1, traj_len)          # bruit pur x_T
    for t in reversed(range(1, T + 1)):
        ts = torch.tensor([t])
        eps_hat = model(x, ts)
        # formule DDPM reverse
        ab, ab_prev = alpha_bar[t], alpha_bar[t-1]
        x0_hat = (x - torch.sqrt(1-ab) * eps_hat) / torch.sqrt(ab)
        mean = torch.sqrt(ab_prev) * x0_hat + torch.sqrt(1-ab_prev) * eps_hat
        z = torch.randn_like(x) if t > 1 else 0
        x = mean + torch.sqrt(beta[t]) * z
    return x  # trajectoire simulée
```

---

## 6. Schedules βₜ

| Schedule | Formule | Recommandé pour |
|---|---|---|
| Linéaire | $\beta_t = \beta_1 + (t/T)(\beta_T - \beta_1)$ | Basique, DDPM original |
| Cosinus | $\bar{\alpha}_t = \cos^2\!\left(\frac{t/T + s}{1+s}\cdot\frac{\pi}{2}\right)$ | Trajectoires ODE (préserve mieux le début) |
| Sigmoid | $\beta_t = \sigma(-6 + 12t/T)$ | Dynamiques non-linéaires |

---

## 7. Accélération de l'inférence

| Méthode | Pas | Qualité | Notes |
|---|---|---|---|
| DDPM standard | 1000 | ★★★ | Référence, lent |
| DDIM | 20–50 | ★★★ | Déterministe, invertible |
| DPM-Solver | 10–20 | ★★★ | Rapide, recommandé |
| DDIM + PLMS | 25 | ★★☆ | Bon compromis |

---

## 8. Extensions pour ODE

### Conditionnement paramétrique

Passer les paramètres de l'ODE (ex. `μ, σ, k`) comme condition :

```python
# Conditionnement par concaténation
def forward(self, x_t, t, params):
    h = torch.cat([x_t, self.time_emb(t), params], dim=-1)
    return self.net(h)
```

### Génération d'ensembles (uncertainty quantification)

Générer N trajectoires depuis N bruits différents → distribution empirique des solutions.

### Problème inverse

Conditionner sur des observations partielles `y = x₀[mask]` via guidance au score :

$$\nabla_{x_t} \log p(y \mid x_t) \approx -\frac{1}{\sigma^2}(x_0^{\text{pred}} - y)[mask]$$

---

## Approche 3 — Probability Flow ODE

### Principe

Toute SDE de diffusion admet une **ODE déterministe** qui préserve exactement les marginales `p_t(x)` à chaque instant. Cette ODE s'appelle la *probability flow ODE* (Song et al., 2020) :

$$\frac{dx}{dt} = f(x,t) - \frac{1}{2}\,g(t)^2\,\nabla_x \log p_t(x)$$

où `∇_x log p_t(x)` est le **score** — la dérivée log-densité — appris par le même réseau `εθ` que le DDPM. En pratique :

$$\nabla_x \log p_t(x) \approx -\frac{\varepsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}}$$

### Relation avec DDPM et DDIM

| | DDPM | DDIM | Probability Flow ODE |
|---|---|---|---|
| Stochastique ? | Oui | Non | Non |
| Basé sur | Markov chain | Sous-séquence DDPM | ODE continue |
| Invertible ? | ✗ | Partiellement | ✓ exact |
| Pas min. | ~1000 | ~20 | ~10–50 |

DDIM est une **discrétisation** de la probability flow ODE — les deux convergent vers la même trajectoire à pas infiniment petit.

### Formulation SDE → ODE

La SDE de Variance Preserving (VP-SDE, cadre général) :

$$dx = -\frac{1}{2}\beta(t)\,x\,dt + \sqrt{\beta(t)}\,dW_t \quad \text{(forward)}$$

Son ODE probability flow équivalente :

$$\frac{dx}{dt} = -\frac{1}{2}\beta(t)\left[x + \nabla_x \log p_t(x)\right]$$

### Exemple : inférence avec un solveur ODE

```python
import torch
from torchdiffeq import odeint

def score_fn(x_t, t_continuous, model, alpha_bar_fn):
    """Convertit t continu ∈ [0,1] en step discret et calcule le score."""
    t_disc = (t_continuous * 999).long().clamp(1, 999)
    ab = alpha_bar_fn(t_disc).unsqueeze(-1)
    eps_hat = model(x_t, t_disc)
    return -eps_hat / torch.sqrt(1 - ab)

def probability_flow_ode(model, alpha_bar_fn, beta_fn):
    def ode_func(t, x):
        score = score_fn(x, t, model, alpha_bar_fn)
        b = beta_fn(t)
        return -0.5 * b * (x + score)   # dx/dt
    return ode_func

@torch.no_grad()
def sample_pfode(model, alpha_bar_fn, beta_fn, traj_len=100, steps=50):
    x = torch.randn(1, traj_len)          # bruit pur à t=1
    t_span = torch.linspace(1.0, 0.0, steps)  # intégration de T→0
    ode_func = probability_flow_ode(model, alpha_bar_fn, beta_fn)
    trajectory = odeint(ode_func, x, t_span, method='rk4')
    return trajectory[-1]  # x à t=0 : trajectoire simulée


# Encodage (inversion exacte) : x₀ → xT
@torch.no_grad()
def encode_pfode(model, x0, alpha_bar_fn, beta_fn, steps=50):
    t_span = torch.linspace(0.0, 1.0, steps)  # intégration de 0→T
    ode_func = probability_flow_ode(model, alpha_bar_fn, beta_fn)
    return odeint(ode_func, x0, t_span, method='rk4')[-1]  # xT
```

### Cas d'usage spécifiques

**Problème inverse / interpolation** : l'invertibilité exacte permet d'encoder deux trajectoires `x₀^A` et `x₀^B` dans l'espace latent (`xT^A`, `xT^B`), d'interpoler linéairement, puis de décoder — utile pour explorer l'espace des solutions.

**Accélération maximale** : en combinant avec DPM-Solver++ (ordre 2–3), on peut simuler en 5–10 pas avec une qualité proche de 1000 pas DDPM.

### Avantages / limites

| ✓ Avantages | ✗ Limites |
|---|---|
| Inversion exacte (encodage) | Mêmes données que DDPM nécessaires |
| Inférence rapide (ODE solver) | Implémentation plus complexe |
| Latent space interpolable | Sensible au choix du solveur ODE |
| Pas de variance stochastique | Pas d'incertitude sur la trajectoire |

---

## 9. Librairies clés

| Librairie | Approche | Usage |
|---|---|---|
| `torchdiffeq` | Neural ODE + PF-ODE | Intégration ODE différentiable, méthode adjoint |
| `diffrax` (JAX) | Neural ODE + PF-ODE | Solveurs ODE haute performance en JAX |
| `diffusers` (HuggingFace) | DDPM / DDIM | Pipelines score-based clé en main |
| `score_sde` (Yang Song) | DDPM + PF-ODE | Référence SDE/ODE, VP/VE schedules |
| `torchsde` | SDE stochastique | SDE différentiable (base du PF-ODE) |

---

## 10. Résumé comparatif

| Critère | Neural ODE | Score-based DDPM | Probability Flow ODE |
|---|---|---|---|
| Type de sortie | Trajectoire unique | Ensemble stochastique | Trajectoire déterministe |
| Incertitude | ✗ | ✓ | ✗ |
| Vitesse inférence | Rapide | Lent (DDPM) / Moyen (DDIM) | Rapide |
| Données nécessaires | Peu | Beaucoup | Beaucoup |
| Invertibilité | ✓ | ✗ | ✓ |
| Conditionnement | Facile | Facile | Moyen |

---

## 11. Références bibliographiques

### Neural ODE

- **[1]** Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018).
  *Neural Ordinary Differential Equations.*
  NeurIPS 2018. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)
  > Article fondateur — introduit la méthode de l'adjoint pour rétropropager à travers un solveur ODE.

- **[2]** Rubanova, Y., Chen, R. T. Q., & Duvenaud, D. (2019).
  *Latent ODEs for Irregularly-Sampled Time Series.*
  NeurIPS 2019. [arXiv:1907.03907](https://arxiv.org/abs/1907.03907)
  > Extension aux séries temporelles irrégulières — très pertinent pour les données ODE partiellement observées.

- **[3]** Kidger, P., Morrill, J., Foster, J., & Lyons, T. (2020).
  *Neural Controlled Differential Equations for Irregular Time Series.*
  NeurIPS 2020. [arXiv:2005.08926](https://arxiv.org/abs/2005.08926)
  > Introduit les Neural CDE, plus stables que les Neural ODE pour les entrées continues.

- **[4]** Kidger, P. (2022).
  *On Neural Differential Equations.*
  PhD Thesis, University of Oxford. [arXiv:2202.02435](https://arxiv.org/abs/2202.02435)
  > Synthèse complète sur les ODE/SDE/CDE neuronaux — référence de fond recommandée.

---

### Score-based DDPM

- **[5]** Ho, J., Jain, A., & Abbeel, P. (2020).
  *Denoising Diffusion Probabilistic Models.*
  NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
  > Article fondateur du DDPM — définit le processus forward/reverse et la loss de débruitage.

- **[6]** Song, Y., & Ermon, S. (2019).
  *Generative Modeling by Estimating Gradients of the Data Distribution.*
  NeurIPS 2019. [arXiv:1907.05600](https://arxiv.org/abs/1907.05600)
  > Introduction du score matching et des modèles basés sur le score (NCSN).

- **[7]** Nichol, A., & Dhariwal, P. (2021).
  *Improved Denoising Diffusion Probabilistic Models.*
  ICML 2021. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
  > Améliore DDPM : schedule cosinus, variance apprise — recommandé pour les applications ODE.

- **[8]** Batzolis, G., Stanczuk, J., Schönlieb, C.-B., & Etmann, C. (2021).
  *Conditional Image Generation with Score-Based Diffusion Models.*
  [arXiv:2111.13606](https://arxiv.org/abs/2111.13606)
  > Conditionnement des modèles de diffusion — applicable au conditionnement sur les paramètres d'ODE.

---

### Probability Flow ODE & DDIM

- **[9]** Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021).
  *Score-Based Generative Modeling through Stochastic Differential Equations.*
  ICLR 2021 (Outstanding Paper). [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)
  > Article clé — unifie DDPM et score matching via les SDE, introduit la probability flow ODE.

- **[10]** Song, J., Meng, C., & Ermon, S. (2021).
  *Denoising Diffusion Implicit Models.*
  ICLR 2021. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
  > Introduit DDIM — inférence non-markovienne, déterministe, 10–50× plus rapide que DDPM.

- **[11]** Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. (2022).
  *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps.*
  NeurIPS 2022. [arXiv:2206.00927](https://arxiv.org/abs/2206.00927)
  > Solveur ODE d'ordre élevé pour la probability flow ODE — 10–20 pas suffisent.

- **[12]** Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. (2022).
  *DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models.*
  [arXiv:2211.01095](https://arxiv.org/abs/2211.01095)
  > Extension conditionnelle de DPM-Solver, ordre 2–3, recommandé pour l'inférence rapide.

---

### Applications aux ODE scientifiques

- **[13]** Rackauckas, C., Ma, Y., Martensen, J., Warner, C., Zubov, K., Supekar, R., ... & Edelman, A. (2020).
  *Universal Differential Equations for Scientific Machine Learning.*
  [arXiv:2001.04385](https://arxiv.org/abs/2001.04385)
  > Combine solveurs numériques et réseaux de neurones pour les ODE physiques.

- **[14]** Haussmann, M., Gerwinn, S., Look, A., Rakitsch, B., & Kandemir, M. (2021).
  *Inferring Latent Dynamics Underlying Neural Population Activity with Variational Sequential Monte Carlo.*
  ICML 2021. [arXiv:2105.04390](https://arxiv.org/abs/2105.04390)
  > Inférence bayésienne sur des dynamiques latentes — liens avec l'approche score-based.

- **[15]** Gao, Y., Shi, J., Luo, D., Wen, H., & Li, Q. (2023).
  *EHRDiff: Exploring Realistic EHR Synthesis with Diffusion Models.*
  [arXiv:2303.05656](https://arxiv.org/abs/2303.05656)
  > Exemple applicatif : diffusion pour générer des séries temporelles médicales (format proche des trajectoires ODE).

---

### Ressources complémentaires

- **Blog** : Lilian Weng, *"What are Diffusion Models?"* (2021) — [lilianweng.github.io](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
  > Synthèse pédagogique très complète sur DDPM, score matching et SDE.

- **Blog** : Yang Song, *"Generative Modeling by Estimating Gradients of the Data Distribution"* — [yang-song.net](https://yang-song.net/blog/2021/score/)
  > Explication intuitive du score matching par l'auteur des articles de référence.

- **Code** : `score_sde` (Yang Song) — [github.com/yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)
  > Implémentation de référence VP-SDE / VE-SDE / Probability Flow ODE.
