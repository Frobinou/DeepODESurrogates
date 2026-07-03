from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from deep_ode_surrogates.application.config.data import DataConfig
from deep_ode_surrogates.domain.data_loader.base import BaseDataLoader
from deep_ode_surrogates.infrastructure.registries.dataloader_registry import register_dataloader


class ParquetDataset(Dataset):
    """
    Dataset that loads data from a Parquet file.
    Suitable for PINNs or supervised learning.
    """

    def __init__(
        self,
        parquet_path: Path,
        input_cols: list[str],
        target_cols: list[str],
        run_id_col: str = "run_id",
    ):
        """
        Args:
            parquet_path (str): path to .parquet file
            input_cols (list): feature columns (x)
            target_cols (list): target columns (y)
        """

        self.df = pd.read_parquet(parquet_path)
        self.input_cols = input_cols
        self.target_cols = target_cols
        self.run_id_col = run_id_col

        # Health check
        required_cols = set(input_cols + target_cols + [self.run_id_col])
        missing_cols = required_cols - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in parquet: {sorted(missing_cols)}")

        self.df = self.df.sort_values([run_id_col, *input_cols]).reset_index(drop=True)
        initial_rows = self.df.groupby(run_id_col, sort=False).first().reset_index()

        x0_by_traj = {
            row[run_id_col]: row[input_cols].to_numpy(dtype="float32")
            for _, row in initial_rows.iterrows()
        }

        y0_by_traj = {
            row[run_id_col]: row[target_cols].to_numpy(dtype="float32")
            for _, row in initial_rows.iterrows()
        }

        # Convert once to numpy for efficiency
        self.x = self.df[input_cols].values.astype("float32")
        self.y = self.df[target_cols].values.astype("float32")
        self.run_ids = self.df[run_id_col].to_numpy()

        self.x0 = torch.stack(
            [torch.tensor(x0_by_traj[traj_id], dtype=torch.float32) for traj_id in self.run_ids]
        )

        self.y0 = torch.stack(
            [torch.tensor(y0_by_traj[traj_id], dtype=torch.float32) for traj_id in self.run_ids]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.x[idx]),
            "y": torch.tensor(self.y[idx]),
            "x0": self.x0[idx],
            "y0": self.y0[idx],
            "run_id": torch.tensor(self.run_ids[idx], dtype=torch.long),
        }


@register_dataloader("parquet")
class ParquetDataLoader(BaseDataLoader):
    def __init__(self, data_config: DataConfig):
        self.parquet_path = Path(data_config.data_path)
        self.input_cols = data_config.input_cols
        self.target_cols = data_config.target_cols
        self.run_id_col = data_config.run_id_col
        super().__init__(data_config=data_config)  # déclenche build_dataset() + split

    @property
    def state_names(self):
        return self.target_cols

    def build_dataset(self) -> Dataset:
        return ParquetDataset(
            parquet_path=self.parquet_path,
            input_cols=self.input_cols,
            target_cols=self.target_cols,
            run_id_col=self.run_id_col,
        )
