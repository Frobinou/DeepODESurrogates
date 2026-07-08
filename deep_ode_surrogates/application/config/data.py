from pathlib import Path

from pydantic import Field, model_validator

from deep_ode_surrogates.application.config.task import TaskConfig


class DataConfig(TaskConfig):
    """Everything needed to build a DataLoader, nothing more."""

    type: str = "parquet"  # for now we only support parquet, but this allows to easily add more data sources in the future
    data_path: Path
    run_id_col: str = Field(
        "run_id", description="Column name for the trajectory/run ID in the dataset."
    )
    batch_size: int = Field(32, gt=0)
    train_ratio: float = Field(0.7, gt=0.0, lt=1.0)
    val_ratio: float = Field(0.15, gt=0.0, lt=1.0)
    pin_memory: bool = True
    num_workers: int = 0
    seed: int = 42

    @model_validator(mode="after")
    def check_ratios(self) -> "DataConfig":
        if self.train_ratio + self.val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be < 1.0")
        return self
