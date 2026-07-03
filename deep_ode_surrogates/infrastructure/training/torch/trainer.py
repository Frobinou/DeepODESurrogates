import torch
from deep_ode_surrogates.domain.data_loader.base import BaseDataLoader


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        t: torch.Tensor,
        device="cpu",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.t = t

        self.callbacks = []
        self.evaluators = []
        self.stop_training = False

        self.epoch_step = 0
        self.last_loss = None
        self.state = {}

        self.dataloader = None

    def fit(self, dataloader: BaseDataLoader, epochs: int) -> None:
        self.dataloader = dataloader

        for cb in self.callbacks:
            cb.on_train_start(self)

        for epoch in range(epochs):
            self._reset_epoch_state()
            if self.stop_training:
                break

            self.epoch_step += 1
            self.model.train()

            for cb in self.callbacks:
                cb.on_epoch_start(self, epoch)

            epoch_loss = 0.0
            num_batches = 0
            loss_dict = {}
            loss_sums = {"total": 0.0, "physics": 0.0, "ic": 0.0, "data": 0.0}
            loss_counts = {"total": 0, "physics": 0, "ic": 0, "data": 0}
            residuals_sum = None
            residuals_count = 0

            for batch in self.dataloader.train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad()
                loss_dict = self.loss_fn(self.model, batch, self.t.to(self.device))
                loss_dict["total"].backward()
                self.optimizer.step()

                epoch_loss += loss_dict["total"].item()
                num_batches += 1

                for key in loss_sums.keys():
                    value = loss_dict.get(key)
                    if value is not None:
                        loss_sums[key] += value.item()
                        loss_counts[key] += 1
                residuals = loss_dict.get("residuals")
                if residuals is not None:
                    residuals = residuals.detach().cpu()
                    residuals_sum = (
                        residuals if residuals_sum is None else residuals_sum + residuals
                    )
                    residuals_count += 1

                for cb in self.callbacks:
                    cb.on_batch_end(self, loss_dict)

            if num_batches > 0:
                epoch_loss /= num_batches

            self.last_loss = epoch_loss

            self.state["loss"] = {
                key: (
                    None
                    if loss_counts[key] == 0
                    else torch.tensor(loss_sums[key] / loss_counts[key])
                )
                for key in loss_sums
            }

            self.state["epoch"] = self.epoch_step

            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch)

            if self.epoch_step % 10 == 0:
                for ev in self.evaluators:
                    evaluation_results = ev.run(self)

                    for cb in self.callbacks:
                        cb.on_evaluation_end(self, evaluation_results)

        for cb in self.callbacks:
            cb.on_train_end(self)

        for cb in self.callbacks:  # Close all ressources
            cb.on_teardown(self)

    def _reset_epoch_state(self):
        self.state["loss"] = {}
        self.state["metrics"] = {}
