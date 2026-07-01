from pathlib import Path
from typing import Protocol
import pandas as pd


class DatasetWriterPort(Protocol):
    def save(self, df: pd.DataFrame, path: Path) -> None:
        ...