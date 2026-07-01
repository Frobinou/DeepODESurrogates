# scripts/train_lotka_volterra.py

from pathlib import Path
import torch

from deep_ode_surrogates.application.train import TrainUseCase
from deep_ode_surrogates.application.config.experiment import ExperimentConfig, PhysicsWeights
from deep_ode_surrogates.application.config.ode import ODESConfig
from deep_ode_surrogates.application.config.data import DataConfig
from deep_ode_surrogates.application.config.training import TrainingConfig
from deep_ode_surrogates.application.config.evaluation import EvaluatorConfig
from deep_ode_surrogates.infrastructure.training.schemas import CallbackConfig

from deep_ode_surrogates.domain.models import AvailablesAIModel
from deep_ode_surrogates.domain.losses import AvailablesLoss
from deep_ode_surrogates.domain.odes.ode_lotka_voltera import ParamsLotkaVolterra


from deep_ode_surrogates.domain.odes.ode_lotka_voltera import ParamsLotkaVolterra
from deep_ode_surrogates.infrastructure.factories.training_pipeline_factory import (
    build_training_pipeline,
)
from deep_ode_surrogates.infrastructure.logging.logger import setup_logger

from deep_ode_surrogates.infrastructure.registries.bootstrap import bootstrap
bootstrap()

device = "cuda" if torch.cuda.is_available() else "cpu"

ode_config = ODESConfig(
    ode_name="lotka_volterra",
    parameters=ParamsLotkaVolterra(
        alpha=0.67,
        beta=1.33,
        delta=1.0,
        gamma=1.0,
    ),
    grid_size=200,
    t_span=(0.0, 50.0),
    initial_conditions=[1.0, 1.0],
    dimension=2,
)

data_config = DataConfig(
    type="parquet",
    data_path=Path("data\lotka_volterra.parquet"),
    input_cols=["t"],
    target_cols=["prey", "predator"],
    batch_size=64,
    train_ratio=0.7,
    val_ratio=0.15,
)

training_config = TrainingConfig(
    lr=1e-3,
    epochs=20,
    checkpoint_k=5,
    log_frequency=50,
    model_name=AvailablesAIModel.BASIC_PINN,
    optimizer="Adam",
)

physics_weights = PhysicsWeights(
    name=AvailablesLoss.PINN_LOSS,
    lambda_ode=1.0,
    lambda_data=1.0,
)

callbacks = CallbackConfig(use_tensorboard=True, use_checkpoint=True,   use_early_stopping=True, checkpoint_k=5, early_stopping_patience=10)

evaluation= EvaluatorConfig(use_mse=True)

experiment_config = ExperimentConfig(
    ode=ode_config,
    data=data_config,
    physics_weights=physics_weights,
    training=training_config,
    callbacks=callbacks,
    evaluation=evaluation,
    device=device,
)

pipeline = build_training_pipeline(
    experiment_config=experiment_config,
    base_output_dir=Path("runs") / "lotka_volterra",
    logger=setup_logger(),
)

TrainUseCase().execute(
    training_pipeline=pipeline,
    epochs=training_config.epochs,
)