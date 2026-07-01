from deep_ode_surrogates.infrastructure.training.callbacks.base import Callback


class FinalEvaluationCallback(Callback):
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def on_train_end(self, trainer):
        evaluation_results = self.evaluator.run(trainer)

        for callback in trainer.callbacks:
            callback.on_evaluation_end(
                trainer=trainer,
                evaluation_results=evaluation_results,
            )
