from deep_ode_surrogates.infrastructure.training.evaluator.mse_evaluator import (
    MSEEvaluator,
)
from deep_ode_surrogates.infrastructure.training.evaluator.trajectory_evaluator import (
    TrajectoryEvaluator,
)


def build_evaluators(evaluation_config, data_loader):
    evaluators = []

    if evaluation_config.use_mse:
        evaluators.append(MSEEvaluator(data_loader))

    if evaluation_config.use_trajectory:
        evaluators.append(TrajectoryEvaluator(data_loader))

    return evaluators
