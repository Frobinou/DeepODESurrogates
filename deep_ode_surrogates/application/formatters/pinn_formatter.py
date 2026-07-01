import numpy as np
import pandas as pd


class PINNFormatter:
    @staticmethod
    def transform(
        df: pd.DataFrame,
        state_cols: list[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        x_cols = (
            ["t"]
            + [col for col in df.columns if col in state_cols]
            + [col for col in df.columns if col not in state_cols + ["t", "run_id"]]
        )

        x = df[x_cols].values.astype(np.float32)
        y = df[state_cols].values.astype(np.float32)

        return x, y
