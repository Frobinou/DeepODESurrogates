# tests/infrastructure/data/test_lotka_volterra_parquet.py
from deep_ode_surrogates.infrastructure.logging.logger import setup_logger

logger = setup_logger()


def test_time_is_sorted(generated_dataframe):
    df = generated_dataframe

    assert df["t"].is_monotonic_increasing
    assert df["t"].is_unique


def test_parquet_has_multiple_complete_trajectories(generated_dataframe):
    df = generated_dataframe
    logger.info(df)
    counts = df.groupby("run_id")["t"].count()

    assert counts.nunique() == 1
    assert counts.iloc[0] > 1
