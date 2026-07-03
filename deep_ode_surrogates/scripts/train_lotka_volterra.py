# scripts/train_lotka_volterra.py

from datetime import datetime
from pathlib import Path

import torch

from deep_ode_surrogates.application.config.data import DataConfig
from deep_ode_surrogates.application.config.evaluation import EvaluatorConfig
from deep_ode_surrogates.application.config.experiment import ExperimentConfig, PhysicsWeights
from deep_ode_surrogates.application.config.ode import ODESConfig
from deep_ode_surrogates.application.config.task import TaskConfig
from deep_ode_surrogates.application.config.training import TrainingConfig
from deep_ode_surrogates.application.train import TrainUseCase
from deep_ode_surrogates.domain.losses import AvailablesLoss
from deep_ode_surrogates.domain.models import AvailablesAIModel
from deep_ode_surrogates.domain.odes.ode_lotka_voltera import ParamsLotkaVolterra
from deep_ode_surrogates.infrastructure.factories.training_pipeline_factory import (
    build_training_pipeline,
)
from deep_ode_surrogates.infrastructure.registries.bootstrap import bootstrap
from deep_ode_surrogates.infrastructure.training.callbacks.schemas import (
    CheckpointCallbackConfig,
    EarlyStoppingCallbackConfig,
    TensorboardCallbackConfig,
)
from deep_ode_surrogates.infrastructure.training.schemas import CallbackConfig

bootstrap()

# Create Experiment FOLDER DIRECTORY
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

EXPERIMENT_DIR = Path("runs") / "lotka_volterra" / f"experiment_{timestamp}"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"

task = TaskConfig(x_names=["t"], y_names=["prey", "predator"])

ode_config = ODESConfig(
    ode_name="lotka_volterra",
    parameters=ParamsLotkaVolterra(
        alpha=1.0,
        beta=0.1,
        delta=0.075,
        gamma=1.5,
    ),
    grid_size=2000,
    t_span=(0.0, 10.0),
    initial_conditions=[10.0, 1.0],
    dimension=2,
)

data_config = DataConfig(
    type="parquet",
    data_path=Path("data") / "generated_dataset_LV.parquet",
    batch_size=64,
    train_ratio=0.7,
    val_ratio=0.15,
    **task.model_dump(),
)

training_config = TrainingConfig(
    lr=1e-3,
    epochs=100,
    checkpoint_k=5,
    model_name=AvailablesAIModel.BASIC_PINN,
    optimizer="Adam",
    evaluators_frequency=10,
    **task.model_dump(),
)

physics_weights = PhysicsWeights(
    name=AvailablesLoss.PINN_LOSS,
    lambda_ode=1.0,
    lambda_data=0.0,
    lambda_ic=1.0,
)

callbacks = CallbackConfig(
    tensorboard=TensorboardCallbackConfig(
        log_dir=EXPERIMENT_DIR / "tensorboard",
        log_frequency=200,
        log_gradients=True,
        log_figures_frequency=10,
    ),
    early_stopping=EarlyStoppingCallbackConfig(
        patience=100,
        best=float("inf"),
    ),
    checkpoint=CheckpointCallbackConfig(save_dir=EXPERIMENT_DIR / "save", top_k=5),
)

evaluation = EvaluatorConfig(use_mse=True, use_trajectory=True, use_plot=True)

experiment_config = ExperimentConfig(
    ode=ode_config,
    data=data_config,
    physics_weights=physics_weights,
    training=training_config,
    callbacks=callbacks,
    evaluation=evaluation,
    device=device,
)

pipeline = build_training_pipeline(experiment_config=experiment_config, output_dir=EXPERIMENT_DIR)

TrainUseCase().execute(
    training_pipeline=pipeline,
    epochs=training_config.epochs,
)
