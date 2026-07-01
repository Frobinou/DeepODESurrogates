from deep_ode_surrogates.infrastructure.training.callbacks.base import Callback


class FinalEvaluationCallback(Callback):
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def on_train_end(self, trainer):
        self.evaluator.run(trainer)
