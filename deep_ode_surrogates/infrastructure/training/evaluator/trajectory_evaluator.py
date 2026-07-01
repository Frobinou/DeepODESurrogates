# infrastructure/training/evaluator/trajectory_evaluator.py

import torch

from deep_ode_surrogates.infrastructure.visualization.plotly.trajectory_plots import (
    plot_phase_space,
    plot_trajectory,
)


class TrajectoryEvaluator:
    def __init__(self, data_loader, max_points: int = 500):
        self.data_loader = data_loader
        self.max_points = max_points

    def run(self, trainer):
        batch = next(iter(self.data_loader.test_loader))

        x = batch["x"].to(trainer.device)
        y_true = batch["y"].to(trainer.device)

        trainer.model.eval()
        with torch.no_grad():
            y_pred = trainer.model(x)

        t = x[:, 0].detach().cpu().numpy()
        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()

        t = t[: self.max_points]
        y_true_np = y_true_np[: self.max_points]
        y_pred_np = y_pred_np[: self.max_points]

        figures = {
            "trajectory": plot_trajectory(t=t, y=y_true_np, y_pred=y_pred_np),
        }

        if y_true_np.shape[1] >= 2:
            figures["phase_space"] = plot_phase_space(
                y=y_true_np,
                y_pred=y_pred_np,
            )

        return {"metrics": {}, "figures": figures}
