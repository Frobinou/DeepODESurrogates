import torch
from deep_ode_surrogates.domain.data_loader.base import BaseDataLoader
from deep_ode_surrogates.infrastructure.training.torch.schemas import EpochStats


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        time_grid: torch.Tensor,
        evaluators_frequency: int,
        device="cpu",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.time_grid = time_grid
        self.evaluators_frequency: int = evaluators_frequency

        self.callbacks = []
        self.evaluators = []
        self.stop_training = False

        self.current_state: EpochStats = None
        self.dataloader = None
        self.current_epoch: int = -1

    def _fit_batch(self, batch) -> None:
        self.optimizer.zero_grad()
        loss_dict = self.loss_fn(self.model, batch, time_grid=self.time_grid.to(self.device))
        loss_dict["total"].backward()
        self.optimizer.step()

        self.current_state.update(loss_dict)

        for cb in self.callbacks:
            cb.on_batch_end(self, loss_dict)

    def fit(self, dataloader: BaseDataLoader, epochs: int) -> None:
        self.dataloader = dataloader

        for cb in self.callbacks:
            cb.on_train_start(self)

        for epoch in range(epochs):
            self._reset_epoch_state()
            if self.stop_training:
                break

            self.model.train()

            for cb in self.callbacks:
                cb.on_epoch_start(self, epoch)

            for batch in self.dataloader.train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self._fit_batch(batch=batch)

                if self.loss_fn.lambda_data == 0:
                    break

            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch)

            if epoch % self.evaluators_frequency == 0:
                for ev in self.evaluators:
                    evaluation_results = ev.run(self)

                    for cb in self.callbacks:
                        cb.on_evaluation_end(self, evaluation_results)

        for cb in self.callbacks:
            cb.on_train_end(self)

        for cb in self.callbacks:  # Close all ressources
            cb.on_teardown(self)

    def _reset_epoch_state(self):
        self.current_epoch += 1
        self.current_state = EpochStats()
