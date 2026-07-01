# application/config/inference.py

from pathlib import Path
from pydantic import BaseModel, Field


class InferenceConfig(BaseModel):
    checkpoint_path: Path
    t_span: tuple[float, float] = (0.0, 10.0)
    n_steps: int = Field(1000, gt=0)
    input_cols: list[str] = ["t"]
    target_cols: list[str]
    device: str = "cpu"