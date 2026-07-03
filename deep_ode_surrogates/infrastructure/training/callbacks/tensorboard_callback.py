from torch.utils.tensorboard import SummaryWriter

from deep_ode_surrogates.infrastructure.training.callbacks.base import Callback
from deep_ode_surrogates.infrastructure.training.callbacks.schemas import TensorboardCallbackConfig


class TensorBoardCallback(Callback):
    def __init__(
        self,
        config: TensorboardCallbackConfig,
    ):
        self.writer = SummaryWriter(config.log_dir)
        self.log_frequency = config.log_frequency
        self.log_gradients = config.log_gradients
        self.log_figures_frequency = config.log_figures_frequency

    # -------------------------
    # Hooks
    # -------------------------

    def on_epoch_end(self, trainer, epoch):
        if trainer.current_epoch % self.log_frequency != 0:
            return

        self._log_losses(trainer)

        if self.log_gradients:
            self._log_gradients(trainer)

    def on_train_end(self, trainer):
        pass

    def on_teardown(self, trainer):
        self.writer.close()

    def on_batch_end(self, trainer, loss):
        return super().on_batch_end(trainer, loss)

    def on_evaluation_end(self, trainer, evaluation_results):
        step = trainer.current_epoch
        self.log_dict(evaluation_results.metrics, step, prefix="Evaluation")

        if step % self.log_figures_frequency == 0:
            self.log_plotly_figures(evaluation_results.figures, step, prefix="Evaluation")

    def on_epoch_start(self, trainer, epoch):
        return super().on_epoch_start(trainer, epoch)

    def on_train_start(self, trainer):
        self._log_experiment_info(trainer)

        layout = {
            "Training": {
                "losses": [
                    "Multiline",
                    [
                        "Training/loss/total",
                        "Training/loss/physics",
                        "Training/loss/ic",
                        "Training/loss/data",
                    ],
                ],
            },
        }
        self.writer.add_custom_scalars(layout)
        return super().on_train_start(trainer)

    # -------------------------
    # Core logging
    # -------------------------

    def _log_experiment_info(self, trainer):
        model_name = trainer.model.__class__.__name__
        loss_name = trainer.loss_fn.__class__.__name__

        rows = [
            ("Modèle", model_name),
            ("Loss", loss_name),
        ]

        for attr, label in (
            ("lambda_ode", "lambda_ode (physics)"),
            ("lambda_data", "lambda_data"),
            ("lambda_ic", "lambda_ic"),
        ):
            if hasattr(trainer.loss_fn, attr):
                rows.append((label, str(getattr(trainer.loss_fn, attr))))

        markdown = "| Paramètre | Valeur |\n|---|---|\n"
        markdown += "\n".join(f"| {key} | `{value}` |" for key, value in rows)

        self.writer.add_text("Experiment/info", markdown, global_step=0)

    def _log_losses(self, trainer):
        step = trainer.current_epoch

        loss_dict = trainer.current_state.get_losses()
        for name, value in loss_dict.items():
            if value is not None:
                self.writer.add_scalar(
                    f"Training/loss/{name}",
                    value.item(),
                    step,
                )

        residuals = trainer.current_state.get_residuals()

        if residuals is not None:
            var_names = getattr(trainer, "var_names", None)

            if var_names is not None:
                for name, res in zip(var_names, residuals.mean(dim=0), strict=False):
                    self.writer.add_scalar(
                        f"Training/residuals/{name}",
                        res.item(),
                        step,
                    )

            self.writer.add_scalar(
                "Training/residuals/mean",
                residuals.mean().item(),
                step,
            )

            self.writer.add_scalar(
                "Training/residuals/max",
                residuals.abs().max().item(),
                step,
            )

    def _log_gradients(self, trainer):
        total_norm = 0.0

        for p in trainer.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2

        total_norm = total_norm**0.5

        self.writer.add_scalar(
            "Training/gradients/global_norm",
            total_norm,
            trainer.current_epoch,
        )

    # -------------------------
    # Public API (pour evaluators)
    # -------------------------

    def log_dict(self, scalar_dict: dict, step: int, prefix: str = "Evaluation"):
        for key, value in scalar_dict.items():
            if isinstance(value, dict):
                self.log_dict(value, step, prefix=f"{prefix}/{key}")
            else:
                self.writer.add_scalar(
                    f"{prefix}/{key}",
                    value,
                    step,
                )

    def log_plotly_figures(self, figures: dict, step: int, prefix: str = "Evaluation"):
        import io

        import numpy as np
        from PIL import Image

        for key, fig in figures.items():
            png_bytes = fig.to_image(format="png")
            image = Image.open(io.BytesIO(png_bytes))
            image = np.asarray(image)

            self.writer.add_image(
                f"{prefix}/{key}",
                image,
                global_step=step,
                dataformats="HWC",
            )
