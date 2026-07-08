# infrastructure/training/evaluator/trajectory_evaluator.py

import numpy as np
import torch

from deep_ode_surrogates.infrastructure.training.evaluator.base import Evaluator
from deep_ode_surrogates.infrastructure.training.evaluator.schemas import (
    EvaluatorResults,
    FigureName,
)
from deep_ode_surrogates.infrastructure.visualization.plotly.trajectory_plots import (
    plot_phase_space,
    plot_trajectory,
)


class TrajectoryEvaluator(Evaluator):
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def _get_train_points_for_trajectory(self, trajectory_id) -> np.ndarray:
        """Récupère les temps `t` des points d'entraînement appartenant à cette trajectoire."""
        train_subset = self.data_loader.train_loader.dataset
        full_dataset = train_subset.dataset
        train_indices = np.asarray(train_subset.indices)

        run_ids = np.asarray(full_dataset.run_ids)[train_indices]
        mask = run_ids == trajectory_id

        return full_dataset.x[train_indices][mask][:, 0]

    def run(self, trainer) -> EvaluatorResults:
        batch = next(iter(self.data_loader.test_loader))
        x = batch["x"].to(trainer.device)
        y_true = batch["y"].to(trainer.device)
        trainer.model.eval()
        with torch.no_grad():
            y_pred = trainer.model(x)

        trajectory_id = batch["run_id"][0]
        mask = batch["run_id"] == trajectory_id

        # See a single trajectory
        x = x[mask]
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        order = torch.argsort(x[:, 0])
        x = x[order]
        y_true = y_true[order]
        y_pred = y_pred[order]

        t = x[:, 0].detach().cpu().numpy()
        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()

        train_t = self._get_train_points_for_trajectory(trajectory_id.item())

        figures = {
            FigureName.TRAJECTORY: plot_trajectory(
                t=t,
                y=y_true_np,
                y_pred=y_pred_np,
                state_names=self.data_loader.state_names,
                train_t=train_t,
            ),
        }

        if y_true_np.shape[1] >= 2:
            figures[FigureName.PHASE_SPACE] = plot_phase_space(
                y=y_true_np,
                y_pred=y_pred_np,
            )

        return EvaluatorResults(figures=figures)
