import torch

from deep_ode_surrogates.application.config.experiment import PhysicsWeights
from deep_ode_surrogates.domain.losses import AvailablesLoss
from deep_ode_surrogates.infrastructure.registries.loss_registry import register_loss
from deep_ode_surrogates.infrastructure.training.torch.schemas import LossName


@register_loss(AvailablesLoss.PINN_LOSS)
class PINNLoss:
    """
    PINN loss = lambda_ode * L_physics + lambda_data * L_data

    Args:
        ode:          ODE object exposing a ``torch_ode(y) -> dy/dt`` method.
        lambda_ode:   Weight for the physics residual loss.
        lambda_data:  Weight for the supervised data loss.
    """

    def __init__(self, device, ode=None, config: PhysicsWeights | None = None):
        self.ode = ode
        if config is None:
            config = PhysicsWeights()
        self.lambda_ode = config.lambda_ode
        self.lambda_data = config.lambda_data
        self.lambda_ic = config.lambda_ic
        self.device = device

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_derivative(
        self,
        y_pred: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        derivatives = []

        for variable_index in range(y_pred.shape[1]):
            y_variable = y_pred[:, variable_index : variable_index + 1]

            dy_variable_dt = torch.autograd.grad(
                outputs=y_variable,
                inputs=t,
                grad_outputs=torch.ones_like(y_variable),
                create_graph=True,
                retain_graph=True,
            )[0]

            derivatives.append(dy_variable_dt)

        return torch.cat(derivatives, dim=1)

    def _physics_loss(self, model: torch.nn.Module, t: torch.Tensor) -> torch.Tensor:
        """
        ODE residual loss: mean((dy/dt - F(y, t))²)

        Note:
            Must NOT be called inside torch.no_grad().

        Args:
            model: PINN model.
            t:     Collocation points, requires_grad=True.

        Returns:
            Scalar loss tensor.
        """
        t = t.detach().clone().requires_grad_(True)
        y_pred = model(t)
        dy_dt = self._compute_derivative(y_pred, t)
        residuals = (dy_dt - self.ode.torch_ode(y_pred)) ** 2
        return residuals.mean(), residuals

    def _initial_condition_loss(self, model, batch):
        y0_pred = model(batch["x0"])
        ic_loss = torch.mean((y0_pred - batch["y0"]) ** 2)
        return ic_loss

    def _data_loss(self, model: torch.nn.Module, batch: dict) -> torch.Tensor:
        """
        Supervised MSE loss: mean((y_pred - y_obs)²)

        Args:
            model: PINN model.
            batch: Dict with keys "x" (N, input_dim) and "y" (N, output_dim).

        Returns:
            Scalar loss tensor.
        """
        x, y_obs = batch["x"], batch["y"]
        y_pred = model(x)
        return ((y_pred - y_obs) ** 2).mean()

    # ── Public interface ──────────────────────────────────────────────────────
    def __call__(
        self,
        model: torch.nn.Module,
        batch: dict,
        time_grid: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        if time_grid is not None and time_grid.ndim == 1:
            time_grid = time_grid.unsqueeze(1)
        total = torch.zeros((), device=self.device)

        physics_loss = None
        data_loss = None
        initial_condition_loss = None
        residuals_tensor = None

        if self.lambda_ode > 0 and self.ode is not None:
            physics_loss, residuals_tensor = self._physics_loss(model, time_grid)
            total = total + self.lambda_ode * physics_loss

        if self.lambda_ic > 0 and self.ode is not None:
            initial_condition_loss = self._initial_condition_loss(model, batch)
            total = total + self.lambda_ic * initial_condition_loss

        if self.lambda_data > 0 and batch is not None:
            data_loss = self._data_loss(model, batch)
            total = total + self.lambda_data * data_loss

        return {
            LossName.TOTAL: total,
            LossName.PHYSICS: physics_loss,
            LossName.DATA: data_loss,
            LossName.IC: initial_condition_loss,
            LossName.RESIDUALS: residuals_tensor,
        }
