import torch

from deep_ode_surrogates.infrastructure.training.evaluator.base import Evaluator
from deep_ode_surrogates.infrastructure.training.evaluator.schemas import (
    EvaluatorResults,
    MetricName,
)


class MSEEvaluator(Evaluator):
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def run(self, trainer):
        model = trainer.model
        model.eval()

        total_mse = 0.0
        n = 0

        with torch.no_grad():
            for batch in self.dataloader.val_loader:
                batch = {k: v.to(trainer.device) for k, v in batch.items()}
                x = batch["x"]
                y_true = batch["y"]
                y_pred = model(x)

                mse = ((y_pred - y_true) ** 2).mean()
                total_mse += mse.item()
                n += 1

        total_mse /= n

        trainer.state["metrics"]["mse"] = total_mse

        model.train()

        return EvaluatorResults(metrics={MetricName.MSE: mse})
