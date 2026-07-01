from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from deep_ode_surrogates.application.config.data import DataConfig


class BaseDataLoader(ABC):
    """
    Abstract base class for all DataLoaders.
    Handles train/val/test splitting logic.
    Subclasses just need to implement `build_dataset()`.
    """

    def __init__(
        self,
        data_config: DataConfig,
    ):
        assert data_config.train_ratio + data_config.val_ratio < 1.0, (
            "train_ratio + val_ratio must be < 1"
        )

        self.batch_size = data_config.batch_size
        self.num_workers = data_config.num_workers
        self.pin_memory = data_config.pin_memory
        self.train_ratio = data_config.train_ratio
        self.val_ratio = data_config.val_ratio
        self.seed = data_config.seed

        dataset = self.build_dataset()
        self.train_loader, self.val_loader, self.test_loader = self._split(dataset)

    @abstractmethod
    def build_dataset(self) -> Dataset:
        """
        Build and return the full Dataset.
        Must be implemented by subclasses.
        """
        ...

    def _split(self, dataset: Dataset) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Splits the dataset and returns the three DataLoaders."""
        n = len(dataset)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        n_test = n - n_train - n_val

        generator = torch.Generator().manual_seed(self.seed)
        train_ds, val_ds, test_ds = random_split(
            dataset, [n_train, n_val, n_test], generator=generator
        )

        kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return (
            DataLoader(train_ds, shuffle=True, **kwargs),
            DataLoader(val_ds, shuffle=False, **kwargs),
            DataLoader(test_ds, shuffle=False, **kwargs),
        )

    def get_train_loader(self) -> DataLoader:
        return self.train_loader

    def get_val_loader(self) -> DataLoader:
        return self.val_loader

    def get_test_loader(self) -> DataLoader:
        return self.test_loader
