# ODE Simulator

L’objectif de ce dépôt est de proposer un framework de simulation de systèmes dynamiques basé sur l’apprentissage profond, combinant deux familles de modèles complémentaires : 
- les Physics-Informed Neural Networks (PINN), permettant d’incorporer explicitement les contraintes physiques dans l’entraînement, 
- et les modèles de diffusion, capables de générer et prédire des dynamiques complexes. 

Le projet est développé avec PyTorch et fournit une infrastructure complète d’expérimentation incluant le suivi des métriques avec TensorBoard, la gestion automatique des checkpoints, ainsi que la sauvegarde des meilleurs modèles (top-k checkpointing) afin de faciliter l’entraînement, l’évaluation et la reproductibilité des expériences.

---

## Vue d'ensemble

Le paquet expose une interface unifiée `Trainer` pour entraîner et simuler des ODE, quel que soit le paradigme choisi :

```
models
    │
    ├── PINN          → minimise un résidu physique + perte sur données
    └── Diffusion     → apprend la distribution des trajectoires
          ├── Score-based DDPM       (génération stochastique)
          └── Probability Flow ODE   (inversion déterministe)
```

---

## Modèles

### PINN — Réseau informé par la physique

Le PINN minimise une perte composite qui contraint le réseau à satisfaire simultanément l'ODE et les observations empiriques :

$$
\mathcal{L} = \lambda_{\text{ode}} \cdot (\mathcal{L}_{\text{ode}} + \mathcal{L}_{\text{ic}}) + \lambda_{\text{data}} \cdot \mathcal{L}_{\text{data}}
$$

| Terme | Description |
|---|---|
| $\mathcal{L}_{\text{ode}}$ | Résidu quadratique moyen de l'ODE sur la grille de collocation |
| $\mathcal{L}_{\text{ic}}$ | Erreur quadratique moyenne sur les conditions initiales en $t=0$ |
| $\mathcal{L}_{\text{data}}$ | Erreur quadratique moyenne sur les observations empiriques (Parquet) |

Trois modes d'entraînement sont disponibles selon les valeurs de $\lambda$ :

| Mode | `lambda_ode` | `lambda_data` |
|---|---|---|
| Physique seule | > 0 | 0 |
| Données seules | 0 | > 0 |
| PINN hybride | > 0 | > 0 |

**Quand l'utiliser :** peu de données, structure de l'ODE connue, besoin d'une trajectoire unique et continue.

---

### Modèles de diffusion

Les modèles de diffusion traitent les **trajectoires** `x(t)` comme des données à modéliser. Plutôt que d'imposer une contrainte physique explicite, ils apprennent la **distribution** des solutions de l'ODE par un processus de bruitage progressif puis de débruitage itératif.

Deux variantes sont disponibles :

#### Score-based DDPM

Le réseau $\varepsilon_\theta$ apprend à prédire le bruit ajouté à chaque étape $t$ du processus forward :

$$
\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{t,\, x_0,\, \varepsilon}\!\left[\|\varepsilon - \varepsilon_\theta(\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon,\; t)\|^2\right]
$$

L'inférence procède par débruitage itératif depuis un bruit gaussien pur $x_T \sim \mathcal{N}(0, I)$ jusqu'à une trajectoire simulée $\hat{x}_0$. Générer plusieurs trajectoires depuis des bruits distincts produit un **ensemble de solutions**, utile pour quantifier l'incertitude.

**Quand l'utiliser :** beaucoup de données disponibles, besoin d'explorer la variabilité des solutions, problèmes inverses avec observations partielles.

#### Probability Flow ODE

Reformulation déterministe du DDPM : le processus de débruitage est exprimé comme une ODE continue, ce qui rend le mapping $x_0 \leftrightarrow x_T$ **exactement invertible** :

$$
\frac{dx}{dt} = f(x,t) - \frac{1}{2}\,g(t)^2\,\nabla_x \log p_t(x)
$$

Le score $\nabla_x \log p_t(x)$ est approché par le même réseau $\varepsilon_\theta$ que le DDPM. L'inférence utilise un solveur ODE standard (RK4, DPM-Solver) en 10–50 pas au lieu de 1 000.

**Quand l'utiliser :** besoin d'inférence rapide, d'interpolation dans l'espace latent des trajectoires, ou d'encodage inverse exact ($x_0 \to x_T$).

---

### Comparaison des approches

| Critère | PINN | Score-based DDPM | Probability Flow ODE |
|---|---|---|---|
| Type de sortie | Trajectoire unique | Ensemble stochastique | Trajectoire déterministe |
| Incertitude | ✗ | ✓ | ✗ |
| Données nécessaires | Peu | Beaucoup | Beaucoup |
| Vitesse d'inférence | Rapide | Lent / Moyen (DDIM) | Rapide |
| Structure ODE requise | ✓ | ✗ | ✗ |
| Invertibilité | ✓ | ✗ | ✓ |

---

## Installation

```bash
pip install -r requirements.txt
```

**Dépendances requises :**

- `torch >= 2.0`
- `numpy`
- `matplotlib`
- `scipy`
- `tensorboard`
- `pyarrow` / `pandas` (pour le chargement des données Parquet)

---

## Structure du projet

```
src/
├── core/
│   ├── checkpoint_manager.py   # Logique de sauvegarde top-k
│   ├── data_loader.py          # Construction du DataLoader Parquet
│   └── models.py               # Dataclass TrainingStepLog
├── data_models.py              # TrainingConfig, DataConfig, ODEExperiment
├── ai_model_repository.py      # Registre des réseaux de neurones
├── odes/                       # Registre et implémentations des ODE
├── diffusion/                  # Modèles de diffusion (DDPM, PF-ODE)
├── logger.py                   # Configuration du logger
└── trainer.py                  # Entraîneur (ce module)
```

---

## Démarrage rapide

### Mode PINN

```python
from pathlib import Path
from src.trainer import Trainer
from src.data_models import TrainingConfig, ODEExperiment

training_config = TrainingConfig(
    epochs=5000,
    l_r=1e-3,
    log_frequency=100,
    top_k_save_frequency=5,
    model_name="MLP",
)

ode_experiment_config = ODEExperiment(
    ode_config=...,         # Nom de l'ODE, paramètres, grid_size, t_span, lambda_ode
    data_config=None,       # Remplacer par DataConfig(...) pour le mode hybride
    model_dimension=2,
    initial_conditions=[1.0, 0.5],
    input_cols=["t"],
    target_cols=["prey", "predator"],
)

trainer = Trainer(
    training_config=training_config,
    ode_experiment_config=ode_experiment_config,
    output_folder_path=Path("outputs/"),
)

trainer.run()
```

### Mode Diffusion

```python
from src.trainer import Trainer
from src.data_models import TrainingConfig, ODEExperiment, DiffusionConfig

training_config = TrainingConfig(
    epochs=5000,
    l_r=1e-4,
    log_frequency=100,
    top_k_save_frequency=5,
    model_name="ScoreNet",
)

ode_experiment_config = ODEExperiment(
    ode_config=None,            # Pas de contrainte physique explicite
    data_config=...,            # DataConfig requis : trajectoires d'entraînement
    diffusion_config=DiffusionConfig(
        variant="ddpm",         # "ddpm" ou "probability_flow"
        T=1000,
        beta_schedule="cosine",
    ),
    model_dimension=2,
    input_cols=["t"],
    target_cols=["prey", "predator"],
)

trainer = Trainer(
    training_config=training_config,
    ode_experiment_config=ode_experiment_config,
    output_folder_path=Path("outputs/"),
)

trainer.run()
```

---

## Configuration

### `TrainingConfig`

| Champ | Type | Description |
|---|---|---|
| `epochs` | `int` | Nombre d'époques d'entraînement |
| `l_r` | `float` | Taux d'apprentissage Adam |
| `log_frequency` | `int` | Intervalle d'époques entre deux écritures TensorBoard |
| `top_k_save_frequency` | `int` | Nombre maximum de checkpoints à conserver |
| `model_name` | `str` | Clé dans `AIMODEL_REPOSITORY` |

### `ODEExperiment`

| Champ | Type | Description |
|---|---|---|
| `ode_config` | `ODEConfig \| None` | Spécification de l'ODE ; `None` désactive la perte physique |
| `data_config` | `DataConfig \| None` | Spécification des données ; `None` désactive la perte sur données |
| `diffusion_config` | `DiffusionConfig \| None` | Paramètres du modèle de diffusion ; `None` pour le mode PINN |
| `model_dimension` | `int` | Nombre de variables d'état de l'ODE |
| `initial_conditions` | `list[float]` | Vecteur d'état initial $x_0$ |
| `input_cols` | `list[str]` | Noms des colonnes d'entrée dans le fichier Parquet |
| `target_cols` | `list[str]` | Noms des colonnes cibles |

### `ODEConfig`

| Champ | Type | Description |
|---|---|---|
| `ode_name` | `str` | Clé dans `ODE_REPOSITORY` |
| `parameters` | `dict` | Paramètres spécifiques à l'ODE (ex. `{"alpha": 1.0, "beta": 0.1}`) |
| `grid_size` | `int` | Nombre de points de collocation |
| `t_span` | `tuple[float, float]` | Intervalle d'intégration $[t_0, t_f]$ |
| `lambda_ode` | `float` | Poids de la perte physique |

### `DiffusionConfig`

| Champ | Type | Description |
|---|---|---|
| `variant` | `str` | `"ddpm"` ou `"probability_flow"` |
| `T` | `int` | Nombre de pas de diffusion (entraînement) |
| `beta_schedule` | `str` | `"linear"`, `"cosine"` ou `"sigmoid"` |
| `inference_steps` | `int` | Nombre de pas à l'inférence (défaut : `T` pour DDPM, `50` pour PF-ODE) |

---

## Structure du répertoire de sortie

Chaque appel à `run()` crée un dossier d'expérience horodaté :

```
outputs/
└── experiment_YYYY-MM-DD_HH-MM-SS/
    ├── tensorboard_logs/       # Sortie du SummaryWriter
    ├── save/                   # Top-k checkpoints (.pt)
    ├── training_config.json
    └── ode_experiment_config.json
```

Lancer TensorBoard avec :

```bash
tensorboard --logdir outputs/
```

---

## Métriques TensorBoard

### Scalaires d'entraînement

| Tag | Description |
|---|---|
| `Training/loss/total` | Perte totale pondérée |
| `Training/loss/physics` | Perte résiduelle ODE — mode PINN uniquement |
| `Training/loss/data` | Perte sur données (MSE ou DDPM) |
| `Training/loss/diffusion` | Perte de débruitage $\|\varepsilon - \varepsilon_\theta\|^2$ — mode diffusion uniquement |
| `Training/residuals/<var_name>` | Résidu absolu moyen par variable d'état |
| `Training/residuals/mean` | Résidu moyen global |
| `Training/residuals/max` | Résidu absolu maximal |
| `Training/gradients/global_norm` | Norme L2 globale des gradients |

### Scalaires d'évaluation

| Tag | Description |
|---|---|
| `Evaluation/MSE` | MSE globale par rapport à la solution de référence scipy |
| `Evaluation/MSE/<var_name>` | MSE par variable |

### Images d'évaluation

| Tag | Description |
|---|---|
| `Evaluation/Observables/DynamicTrajectories` | Trajectoires temporelles : prédiction vs référence |
| `Evaluation/Observables/DynamicPhaseTrajectories` | Portrait de l'espace des phases à l'époque courante |
| `Evaluation/Observables/PhasePortraitOverlay` | Superposition des portraits de phase sur les époques |
| `Evaluation/Residuals/CollocationHeatmap` | Carte de chaleur du résidu sur la grille de collocation (PINN) |
| `Evaluation/Diffusion/SampleTrajectories` | Trajectoires générées par échantillonnage (mode diffusion) |
| `Evaluation/Diffusion/EnsembleSpread` | Dispersion de l'ensemble — écart-type par variable (DDPM) |

---

## Exemple : Lotka-Volterra

### Avec PINN

```python
ode_config = ODEConfig(
    ode_name="LotkaVolterra",
    parameters={"alpha": 1.0, "beta": 0.1, "delta": 0.075, "gamma": 1.5},
    grid_size=1000,
    t_span=(0.0, 15.0),
    lambda_ode=1.0,
)

ode_experiment_config = ODEExperiment(
    ode_config=ode_config,
    data_config=None,
    model_dimension=2,
    initial_conditions=[10.0, 5.0],
    input_cols=["t"],
    target_cols=["prey", "predator"],
)
```

### Avec DDPM

```python
ode_experiment_config = ODEExperiment(
    ode_config=None,
    data_config=DataConfig(path="data/lotka_volterra.parquet"),
    diffusion_config=DiffusionConfig(
        variant="ddpm",
        T=1000,
        beta_schedule="cosine",
        inference_steps=50,
    ),
    model_dimension=2,
    input_cols=["t"],
    target_cols=["prey", "predator"],
)
```

---

## Ajouter une nouvelle ODE

1. Implémenter une classe héritant de l'interface ODE de base :

```python
class MyODE:
    def __init__(self, params: dict): ...

    def torch_ode(self, y: torch.Tensor) -> torch.Tensor:
        """Membre de droite F(y, t) — forme (N, n_vars)."""
        ...

    def simulate(self, t_span, x0, nb_points):
        """Solution de référence scipy."""
        ...

    def log_trajectory_plot(self, t_true, y_true, y_pred) -> np.ndarray | None:
        """Retourner un tableau uint8 CHW ou None pour ignorer la journalisation."""
        ...

    def log_trajectory_phase_space_plot(self, y_true, y_pred) -> np.ndarray | None:
        ...
```

2. L'enregistrer :

```python
ODE_REPOSITORY.register("MyODE", MyODE)
```

---

## Problèmes connus et limitations

- `var_names` doit être défini explicitement sur l'instance du trainer (`trainer.var_names = ["prey", "predator"]`) ou déduit de `target_cols` — pas de câblage automatique pour l'instant.
- `_phase_overlay_history` croît sans limite. Utiliser `deque(maxlen=N)` pour les longues exécutions :

```python
from collections import deque
trainer._phase_overlay_history = deque(maxlen=20)
```

- `physics_loss_residual` utilise `torch.autograd.grad` et **ne doit pas** être appelé dans un contexte `torch.no_grad()`.
- En mode diffusion, `inference_steps` doit être ≤ `T`. Pour la variante `"probability_flow"`, une valeur de 20–50 est suffisante ; descendre en dessous de 10 peut dégrader la qualité des trajectoires générées.