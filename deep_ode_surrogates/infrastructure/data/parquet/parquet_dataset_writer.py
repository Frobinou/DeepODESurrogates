from pathlib import Path
import pandas as pd


class ParquetDatasetWriter:
    def save(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)