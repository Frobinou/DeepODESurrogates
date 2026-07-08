# Framework Architecture

DeepODESurrogates is organised around a small number of independent components.
Each component has a single responsibility and can be extended without modifying
the rest of the framework.

```text
ExperimentConfig
        │
        ▼
Training Pipeline Factory
        │
        ├── Model Factory
        ├── ODE Factory
        ├── Loss Factory
        ├── Evaluator Factory
        └── Trainer Factory
                     │
                     ▼
                  Trainer
        ├── Model
        ├── Optimizer
        ├── Callbacks
        ├── Evaluators
        └── CheckpointManager
```

---

# Configuration

The whole framework is driven by a single validated configuration object.

```text
ExperimentConfig
├── ODESConfig
├── DataConfig
├── PhysicsWeights
└── TrainingConfig
```

This object contains every hyperparameter required to reproduce an experiment,
from the ODE parameters to the optimizer configuration.

::: deep_ode_surrogates.application.config.experiment.ExperimentConfig

---

# Trainer

The `Trainer` orchestrates the complete optimisation process.

Its responsibilities include:

- running the training loop;
- computing the loss;
- performing backpropagation;
- executing callbacks;
- evaluating metrics;
- saving checkpoints.

Unlike models or losses, the trainer contains **no problem-specific logic**.
Everything is delegated to interchangeable components.

::: deep_ode_surrogates.infrastructure.training.torch.trainer.Trainer

---

# Factories

Factories are responsible for constructing framework objects from an
`ExperimentConfig`.

This design keeps training scripts extremely small while ensuring every component
is instantiated consistently.

Current factories include:

- Model Factory
- ODE Factory
- Loss Factory
- Evaluator Factory
- Trainer Factory
- Training Pipeline Factory

---

# Callbacks

Callbacks execute side effects during training without modifying the training
loop itself.

Typical use cases include:

- TensorBoard logging
- checkpoint saving
- early stopping
- custom monitoring

To create a new callback, inherit from `Callback` and override only the hooks
you need.

::: deep_ode_surrogates.infrastructure.training.callbacks.base.Callback

---

# Evaluators

Evaluators compute metrics independently from the optimisation process.

Unlike callbacks, evaluators never modify the training state.
They simply observe the current model and return numerical metrics.

Typical evaluators include:

- validation MSE;
- ODE residual;
- custom physics metrics.

::: deep_ode_surrogates.infrastructure.training.evaluator.base.Evaluator

---

# Registry

DeepODESurrogates uses registries to dynamically instantiate models,
ODEs, losses and dataloaders from configuration files.

Adding a new component only requires:

1. implementing the class;
2. registering it with the appropriate registry.

::: deep_ode_surrogates.infrastructure.registries

---

# Models

Models are standard PyTorch modules registered inside the framework.

Every model receives an ODE configuration and predicts the state variables from
the input coordinates.

Example implementations include:

- Basic PINN
- future DeepONet implementations
- Fourier Neural Operators

---

# ODEs

Each ODE encapsulates the mathematical definition of a dynamical system.

An ODE implementation is responsible for:

- defining the governing equations;
- exposing the system dimension;
- computing the physics residual.

New ODEs can be added without modifying the trainer or the loss.

---

# Checkpoints

Training checkpoints store the complete experiment state.

A checkpoint contains:

- model parameters;
- optimizer state;
- experiment configuration;
- training metadata.

Training can optionally start from previously saved weights using
`TrainingConfig(init_from_checkpoint=...)`.

::: deep_ode_surrogates.infrastructure.persistence.checkpoints.checkpoint_manager.CheckpointManager
