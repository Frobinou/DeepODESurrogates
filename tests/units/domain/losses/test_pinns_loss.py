import torch

from deep_ode_surrogates.domain.losses.pinn_losses import PINNLoss


class LinearModel(torch.nn.Module):
    """
    y0 = 2t
    y1 = 3t
    donc dy/dt = [2, 3]
    """

    def forward(self, t):
        return torch.cat([2.0 * t, 3.0 * t], dim=1)


class ConstantODE:
    """
    ODE compatible avec LinearModel :
    f(y, t) = [2, 3]
    """

    def torch_ode(self, y):
        return torch.stack(
            [
                torch.full_like(y[:, 0], 2.0),
                torch.full_like(y[:, 1], 3.0),
            ],
            dim=1,
        )


class WrongConstantODE:
    """
    ODE volontairement fausse :
    f(y, t) = [0, 0]
    """

    def torch_ode(self, y):
        return torch.zeros_like(y)


def test_compute_derivative_returns_one_derivative_per_output_variable():
    loss = PINNLoss()

    t = torch.linspace(0, 1, 5).reshape(-1, 1)
    t.requires_grad_(True)

    model = LinearModel()
    y_pred = model(t)

    dy_dt = loss._compute_derivative(y_pred, t)

    expected = torch.tensor(
        [[2.0, 3.0]] * 5,
        dtype=dy_dt.dtype,
    )

    assert dy_dt.shape == (5, 2)
    torch.testing.assert_close(dy_dt, expected)


def test_physics_loss_is_zero_when_model_satisfies_ode():
    loss = PINNLoss(
        ode=ConstantODE(),
        lambda_ode=1.0,
        lambda_data=0.0,
    )

    model = LinearModel()
    t = torch.linspace(0, 1, 10).reshape(-1, 1)
    t.requires_grad_(True)

    physics_loss = loss._physics_loss(model, t)

    torch.testing.assert_close(
        physics_loss,
        torch.tensor(0.0),
        atol=1e-6,
        rtol=1e-6,
    )


def test_physics_loss_is_positive_when_model_does_not_satisfy_ode():
    loss = PINNLoss(
        ode=WrongConstantODE(),
        lambda_ode=1.0,
        lambda_data=0.0,
    )

    model = LinearModel()
    t = torch.linspace(0, 1, 10).reshape(-1, 1)
    t.requires_grad_(True)

    physics_loss = loss._physics_loss(model, t)

    assert physics_loss.item() > 0.0


def test_data_loss_is_zero_when_prediction_matches_observation():
    loss = PINNLoss(lambda_ode=0.0, lambda_data=1.0)

    model = LinearModel()

    x = torch.tensor(
        [[0.0], [1.0], [2.0]],
        dtype=torch.float32,
    )
    y = model(x).detach()

    batch = {
        "x": x,
        "y": y,
    }

    data_loss = loss._data_loss(model, batch)

    torch.testing.assert_close(
        data_loss,
        torch.tensor(0.0),
        atol=1e-6,
        rtol=1e-6,
    )


def test_data_loss_is_positive_when_prediction_differs_from_observation():
    loss = PINNLoss(lambda_ode=0.0, lambda_data=1.0)

    model = LinearModel()

    batch = {
        "x": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        "y": torch.zeros((2, 2), dtype=torch.float32),
    }

    data_loss = loss._data_loss(model, batch)

    assert data_loss.item() > 0.0


def test_total_loss_combines_physics_and_data_terms():
    loss = PINNLoss(
        ode=WrongConstantODE(),
        lambda_ode=2.0,
        lambda_data=3.0,
    )

    model = LinearModel()

    batch = {
        "x": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        "y": torch.zeros((2, 2), dtype=torch.float32),
    }

    t = torch.linspace(0, 1, 5)

    loss_dict = loss(
        model=model,
        batch=batch,
        t=t,
    )

    expected_total = 2.0 * loss_dict["physics"] + 3.0 * loss_dict["data"]

    torch.testing.assert_close(
        loss_dict["total"],
        expected_total,
    )


def test_physics_loss_is_none_when_lambda_ode_is_zero():
    loss = PINNLoss(
        ode=WrongConstantODE(),
        lambda_ode=0.0,
        lambda_data=1.0,
    )

    model = LinearModel()

    batch = {
        "x": torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        "y": torch.zeros((2, 2), dtype=torch.float32),
    }

    loss_dict = loss(
        model=model,
        batch=batch,
        t=torch.linspace(0, 1, 5),
    )

    assert loss_dict["physics"] is None
    assert loss_dict["data"] is not None
    torch.testing.assert_close(
        loss_dict["total"],
        loss_dict["data"],
    )


def test_data_loss_is_none_when_lambda_data_is_zero():
    loss = PINNLoss(
        ode=WrongConstantODE(),
        lambda_ode=1.0,
        lambda_data=0.0,
    )

    model = LinearModel()

    loss_dict = loss(
        model=model,
        batch=None,
        t=torch.linspace(0, 1, 5),
    )

    assert loss_dict["data"] is None
    assert loss_dict["physics"] is not None
    torch.testing.assert_close(
        loss_dict["total"],
        loss_dict["physics"],
    )


def test_t_can_be_1d_or_2d():
    loss = PINNLoss(
        ode=ConstantODE(),
        lambda_ode=1.0,
        lambda_data=0.0,
    )

    model = LinearModel()

    loss_1d = loss(
        model=model,
        batch=None,
        t=torch.linspace(0, 1, 5),
    )

    loss_2d = loss(
        model=model,
        batch=None,
        t=torch.linspace(0, 1, 5).reshape(-1, 1),
    )

    torch.testing.assert_close(
        loss_1d["total"],
        loss_2d["total"],
    )
