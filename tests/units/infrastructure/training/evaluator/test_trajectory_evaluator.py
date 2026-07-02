# tests/infrastructure/training/evaluator/test_trajectory_evaluator.py

from types import SimpleNamespace

import numpy as np
import torch

from deep_ode_surrogates.infrastructure.training.evaluator.trajectory_evaluator import (
    TrajectoryEvaluator,
)


class DummyModel(torch.nn.Module):
    def forward(self, x):
        # Prédiction facilement vérifiable : y_pred = 10 * t
        return x[:, :1] * 10


def test_trajectory_evaluator_sorts_trajectory_before_plot(monkeypatch):
    # t volontairement désordonné
    x = torch.tensor([[0.3], [0.1], [0.2]], dtype=torch.float32)

    # ground truth aligné avec chaque t avant tri
    y = torch.tensor([[30.0], [10.0], [20.0]], dtype=torch.float32)

    data_loader = SimpleNamespace(test_loader=[{"x": x, "y": y}])

    trainer = SimpleNamespace(
        model=DummyModel(),
        device=torch.device("cpu"),
    )

    captured = {}

    def fake_plot_trajectory(t, y, y_pred=None, **kwargs):
        captured["t"] = t
        captured["y"] = y
        captured["y_pred"] = y_pred
        return "trajectory-figure"

    def fake_plot_phase_space(*args, **kwargs):
        return "phase-space-figure"

    monkeypatch.setattr(
        "deep_ode_surrogates.infrastructure.training.evaluator.trajectory_evaluator.plot_trajectory",
        fake_plot_trajectory,
    )
    monkeypatch.setattr(
        "deep_ode_surrogates.infrastructure.training.evaluator.trajectory_evaluator.plot_phase_space",
        fake_plot_phase_space,
    )

    evaluator = TrajectoryEvaluator(data_loader=data_loader, max_points=10)

    result = evaluator.run(trainer)

    assert result["figures"]["trajectory"] == "trajectory-figure"

    np.testing.assert_allclose(captured["t"], np.array([0.1, 0.2, 0.3]))
    np.testing.assert_allclose(captured["y"].ravel(), np.array([10.0, 20.0, 30.0]))
    np.testing.assert_allclose(captured["y_pred"].ravel(), np.array([1.0, 2.0, 3.0]))
