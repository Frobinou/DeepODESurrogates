from pathlib import Path

from pydantic import BaseModel


class LoggingCallbackConfig(BaseModel):
    frequency: int = 10


class TensorboardCallbackConfig(BaseModel):
    log_dir: Path = Path("")
    log_frequency: int = 10
    log_gradients: bool = True
    log_figures_frequency: int = 100


class EarlyStoppingCallbackConfig(BaseModel):
    patience: int = 10
    best: float = float("inf")


class CheckpointCallbackConfig(BaseModel):
    save_dir: Path = Path("")
    top_k: int = 5
