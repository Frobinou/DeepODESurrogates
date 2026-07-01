# infrastructure/training/evaluators/evaluator_factory.py

from deep_ode_surrogates.infrastructure.training.evaluator.mse_evaluator import (
    MSEEvaluator,
)


def build_evaluators(
    evaluation_config,
    data_loader,
):
    evaluators = []

    if evaluation_config.use_mse:
        evaluators.append(MSEEvaluator(data_loader))

    return evaluators
