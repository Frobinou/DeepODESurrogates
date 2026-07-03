from abc import ABC, abstractmethod

from deep_ode_surrogates.infrastructure.training.evaluator.schemas import EvaluatorResults


class Evaluator(ABC):
    @abstractmethod
    def run(self, trainer) -> EvaluatorResults:
        pass
