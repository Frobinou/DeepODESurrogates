from torch.utils.tensorboard import SummaryWriter

from deep_ode_surrogates.infrastructure.training.callbacks.base import Callback


class TensorBoardCallback(Callback):
    def __init__(
        self,
        log_dir: str,
        log_frequency: int = 10,
        log_gradients: bool = True,
        log_figures_frequency: int = 100,
    ):
        self.writer = SummaryWriter(log_dir)
        self.log_frequency = log_frequency
        self.log_gradients = log_gradients
        self.log_figures_frequency = log_figures_frequency

    # -------------------------
    # Hooks
    # -------------------------

    def on_epoch_end(self, trainer, epoch):
        if trainer.epoch_step % self.log_frequency != 0:
            return

        self._log_losses(trainer)

        if self.log_gradients:
            self._log_gradients(trainer)

    def on_train_end(self, trainer):
        self.writer.close()

    def on_batch_end(self, trainer, loss):
        return super().on_batch_end(trainer, loss)

    def on_evaluation_end(self, trainer, evaluation_results):
        step = trainer.epoch_step

        metrics = evaluation_results.get("metrics", {})
        figures = evaluation_results.get("figures", {})

        self.log_dict(metrics, step, prefix="Evaluation")

        if step % self.log_figures_frequency == 0:
            self.log_plotly_figures(figures, step, prefix="Evaluation")

    def on_epoch_start(self, trainer, epoch):
        return super().on_epoch_start(trainer, epoch)

    def on_train_start(self, trainer):
        return super().on_train_start(trainer)

    # -------------------------
    # Core logging
    # -------------------------

    def _log_losses(self, trainer):
        step = trainer.epoch_step
        loss_dict = getattr(trainer.state, "loss", None)

        if loss_dict is None:
            return

        losses_to_group = {}

        for name in ["total", "physics", "data", "ic"]:
            value = loss_dict.get(name)

            if value is not None:
                scalar_value = value.item()

                self.writer.add_scalar(
                    f"Training/loss/{name}",
                    scalar_value,
                    step,
                )

                losses_to_group[name] = scalar_value

        if losses_to_group:
            self.writer.add_scalars(
                "Training/losses",
                losses_to_group,
                step,
            )

        residuals = loss_dict.get("residuals", None)

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
            trainer.epoch_step,
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
