# Architecture du framework

DeepODESurrogates est organisé autour d'un petit nombre de composants
indépendants. Chaque composant possède une responsabilité unique et peut
être étendu sans modifier le reste du framework.

``` text
ExperimentConfig
        │
        ▼
Fabrique du pipeline d'entraînement
        │
        ├── Fabrique de modèles
        ├── Fabrique d'EDO
        ├── Fabrique de fonctions de perte
        ├── Fabrique d'évaluateurs
        └── Fabrique de trainer
                     │
                     ▼
                  Trainer
        ├── Modèle
        ├── Optimiseur
        ├── Callbacks
        ├── Évaluateurs
        └── Gestionnaire de points de contrôle
```

------------------------------------------------------------------------

# Configuration

L'ensemble du framework est piloté par un unique objet de configuration
validé.

``` text
ExperimentConfig
├── ODESConfig
├── DataConfig
├── PhysicsWeights
└── TrainingConfig
```

Cet objet contient tous les hyperparamètres nécessaires pour reproduire
une expérience, des paramètres de l'EDO jusqu'à la configuration de
l'optimiseur.

::: deep_ode_surrogates.application.config.experiment.ExperimentConfig

------------------------------------------------------------------------

# Trainer

Le `Trainer` orchestre l'ensemble du processus d'optimisation.

Ses responsabilités comprennent :

-   exécuter la boucle d'entraînement ;
-   calculer la fonction de perte ;
-   effectuer la rétropropagation ;
-   exécuter les callbacks ;
-   évaluer les métriques ;
-   sauvegarder les points de contrôle.

Contrairement aux modèles ou aux fonctions de perte, le trainer ne
contient **aucune logique spécifique au problème**. Tout est délégué à
des composants interchangeables.

::: deep_ode_surrogates.infrastructure.training.torch.trainer.Trainer

------------------------------------------------------------------------

# Fabriques

Les fabriques sont responsables de la construction des objets du
framework à partir d'une `ExperimentConfig`.

Cette conception permet de conserver des scripts d'entraînement très
concis tout en garantissant une instanciation cohérente de chaque
composant.

Les fabriques actuellement disponibles sont :

-   Fabrique de modèles
-   Fabrique d'EDO
-   Fabrique de fonctions de perte
-   Fabrique d'évaluateurs
-   Fabrique de trainer
-   Fabrique du pipeline d'entraînement

------------------------------------------------------------------------

# Callbacks

Les callbacks exécutent des effets de bord pendant l'entraînement sans
modifier la boucle d'entraînement elle-même.

Cas d'utilisation typiques :

-   journalisation TensorBoard ;
-   sauvegarde des points de contrôle ;
-   arrêt anticipé (early stopping) ;
-   surveillance personnalisée.

Pour créer un nouveau callback, héritez de `Callback` et redéfinissez
uniquement les hooks nécessaires.

::: deep_ode_surrogates.infrastructure.training.callbacks.base.Callback

------------------------------------------------------------------------

# Évaluateurs

Les évaluateurs calculent des métriques indépendamment du processus
d'optimisation.

Contrairement aux callbacks, les évaluateurs ne modifient jamais l'état
de l'entraînement. Ils observent simplement le modèle courant et
renvoient des métriques numériques.

Exemples d'évaluateurs :

-   MSE sur l'ensemble de validation ;
-   résidu de l'EDO ;
-   métriques physiques personnalisées.

::: deep_ode_surrogates.infrastructure.training.evaluator.base.Evaluator

------------------------------------------------------------------------

# Registre

DeepODESurrogates utilise des registres pour instancier dynamiquement
les modèles, les EDO, les fonctions de perte et les chargeurs de données
à partir des fichiers de configuration.

L'ajout d'un nouveau composant ne nécessite que :

1.  d'implémenter la classe ;
2.  de l'enregistrer dans le registre approprié.

::: deep_ode_surrogates.infrastructure.registries

------------------------------------------------------------------------

# Modèles

Les modèles sont des modules PyTorch standards enregistrés dans le
framework.

Chaque modèle reçoit une configuration d'EDO et prédit les variables
d'état à partir des coordonnées d'entrée.

Exemples d'implémentations :

-   PINN de base ;
-   futures implémentations de DeepONet ;
-   opérateurs neuronaux de Fourier.

------------------------------------------------------------------------

# EDO

Chaque EDO encapsule la définition mathématique d'un système dynamique.

Une implémentation d'EDO est responsable de :

-   définir les équations gouvernantes ;
-   exposer la dimension du système ;
-   calculer le résidu physique.

De nouvelles EDO peuvent être ajoutées sans modifier le trainer ni la
fonction de perte.

------------------------------------------------------------------------

# Points de contrôle

Les points de contrôle d'entraînement stockent l'état complet d'une
expérience.

Un point de contrôle contient :

-   les paramètres du modèle ;
-   l'état de l'optimiseur ;
-   la configuration de l'expérience ;
-   les métadonnées de l'entraînement.

L'entraînement peut démarrer à partir de poids précédemment sauvegardés
via `TrainingConfig(init_from_checkpoint=...)`.

:::
deep_ode_surrogates.infrastructure.persistence.checkpoints.checkpoint_manager.CheckpointManager
