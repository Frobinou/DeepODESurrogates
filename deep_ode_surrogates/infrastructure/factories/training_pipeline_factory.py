# infrastructure/training/training_pipeline_factory.py

from pathlib import Path

from deep_ode_surrogates.application.config.experiment import ExperimentConfig
from deep_ode_surrogates.application.config.training import TrainingPipeline
from deep_ode_surrogates.infrastructure.factories.callback_factory import build_callbacks
from deep_ode_surrogates.infrastructure.factories.evaluator_factory import build_evaluators
from deep_ode_surrogates.infrastructure.factories.trainer_factory import build_trainer
from deep_ode_surrogates.infrastructure.persistence.experiments.experiment_io import save_experiment
from deep_ode_surrogates.infrastructure.registries.dataloader_registry import dataloader_registry


def build_training_pipeline(
    experiment_config: ExperimentConfig, base_output_dir: Path, logger=None
):
    experiment_path = save_experiment(
        experiment_config,
        base_dir=base_output_dir,
    )

    dataloader = dataloader_registry.create(
        experiment_config.data.type, data_config=experiment_config.data
    )

    trainer = build_trainer(
        ode_config=experiment_config.ode,
        loss_config=experiment_config.physics_weights,
        training_config=experiment_config.training,
        device=experiment_config.device,
    )

    callbacks = build_callbacks(
        experiment_path=experiment_path,
        training_config=experiment_config.training,
        callback_config=experiment_config.callbacks,
        logger=logger,
    )

    evaluators = build_evaluators(
        evaluation_config=experiment_config.evaluation, data_loader=dataloader
    )

    return TrainingPipeline(
        trainer=trainer,
        dataloader=dataloader,
        callbacks=callbacks,
        evaluators=evaluators,
        experiment_path=experiment_path,
    )
