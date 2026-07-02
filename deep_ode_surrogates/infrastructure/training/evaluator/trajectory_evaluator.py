# infrastructure/training/evaluator/trajectory_evaluator.py

import torch

from deep_ode_surrogates.infrastructure.visualization.plotly.trajectory_plots import (
    plot_phase_space,
    plot_trajectory,
)


class TrajectoryEvaluator:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def run(self, trainer):
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

        figures = {
            "trajectory": plot_trajectory(
                t=t, y=y_true_np, y_pred=y_pred_np, state_names=self.data_loader.state_names
            ),
        }

        if y_true_np.shape[1] >= 2:
            figures["phase_space"] = plot_phase_space(
                y=y_true_np,
                y_pred=y_pred_np,
            )

        return {"metrics": {}, "figures": figures}
