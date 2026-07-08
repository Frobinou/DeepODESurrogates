from types import SimpleNamespace

import numpy as np
import torch

from deep_ode_surrogates.infrastructure.training.evaluator.trajectory_evaluator import (
    TrajectoryEvaluator,
)


class DummyModel(torch.nn.Module):
    def forward(self, x):
        return x[:, :1] * 10


def test_trajectory_evaluator_filters_and_sorts_single_run_before_plot(monkeypatch):
    x = torch.tensor([[0.3], [0.1], [0.2], [0.1]], dtype=torch.float32)
    y = torch.tensor([[30.0], [10.0], [20.0], [999.0]], dtype=torch.float32)
    run_id = torch.tensor([0, 0, 0, 1], dtype=torch.long)

    batch = {
        "x": x,
        "y": y,
        "x0": torch.tensor([[0.0]], dtype=torch.float32),
        "y0": torch.tensor([[0.0]], dtype=torch.float32),
        "run_id": run_id,
    }

    full_dataset = SimpleNamespace(
        x=np.array([[0.0], [0.1], [0.2], [0.3], [0.1]], dtype=np.float32),
        run_ids=np.array([0, 0, 0, 0, 1]),
    )

    train_subset = SimpleNamespace(
        dataset=full_dataset,
        indices=np.array([0, 1, 2, 4]),
    )

    data_loader = SimpleNamespace(
        test_loader=[batch],
        train_loader=SimpleNamespace(dataset=train_subset),
        state_names=["state"],
    )

    trainer = SimpleNamespace(
        model=DummyModel(),
        device=torch.device("cpu"),
    )

    captured = {}

    def fake_plot_trajectory(t, y, y_pred=None, state_names=None, train_t=None, **kwargs):
        captured["t"] = t
        captured["y"] = y
        captured["y_pred"] = y_pred
        captured["state_names"] = state_names
        captured["train_t"] = train_t
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

    evaluator = TrajectoryEvaluator(data_loader=data_loader)

    result = evaluator.run(trainer)

    assert "trajectory" in result.figures

    np.testing.assert_allclose(captured["t"], np.array([0.1, 0.2, 0.3]))
    np.testing.assert_allclose(captured["y"].ravel(), np.array([10.0, 20.0, 30.0]))
    np.testing.assert_allclose(captured["y_pred"].ravel(), np.array([1.0, 2.0, 3.0]))

    assert captured["state_names"] == ["state"]
    np.testing.assert_allclose(captured["train_t"], np.array([0.0, 0.1, 0.2]))
