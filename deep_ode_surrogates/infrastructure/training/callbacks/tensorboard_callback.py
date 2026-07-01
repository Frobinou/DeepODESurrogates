from torch.utils.tensorboard import SummaryWriter

from deep_ode_surrogates.infrastructure.training.callbacks.base import Callback


class TensorBoardCallback(Callback):
    def __init__(
        self,
        log_dir: str,
        log_frequency: int = 10,
        log_gradients: bool = True,
    ):
        self.writer = SummaryWriter(log_dir)
        self.log_frequency = log_frequency
        self.log_gradients = log_gradients

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

        loss_dict = getattr(trainer, "last_losses", None)

        if loss_dict is None:
            return

        # total loss
        self.writer.add_scalar(
            "Training/loss/total",
            loss_dict["total"].item(),
            step,
        )

        # physics loss
        if "physics" in loss_dict and loss_dict["physics"] is not None:
            self.writer.add_scalar(
                "Training/loss/physics",
                loss_dict["physics"].item(),
                step,
            )

        # data loss
        if "data" in loss_dict and loss_dict["data"] is not None:
            self.writer.add_scalar(
                "Training/loss/data",
                loss_dict["data"].item(),
                step,
            )

        # residuals
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
