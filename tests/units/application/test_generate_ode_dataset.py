import numpy as np
import pandas as pd
from pydantic import BaseModel

from deep_ode_surrogates.application.generate_ode_dataset import (
    GenerateODEDatasetUseCase,
)


class DummyParams(BaseModel):
    alpha: float = 1.0
    beta: float = 0.1


class DummyODE:
    def __init__(self, params: DummyParams):
        self.params = params

    def update_params(self, new_params: DummyParams):
        self.params = new_params


class DummyTrajectory:
    def __init__(self, t, y, run_id=None):
        self.t = t
        self.y = y
        self.run_id = run_id


class DummySimulator:
    def execute(self, ode, x0, t_span, n_steps, run_id=None):
        t = np.linspace(t_span[0], t_span[1], n_steps)

        y = np.column_stack(
            [
                x0[0] + ode.params.alpha * t,
                x0[1] + ode.params.beta * t,
            ]
        )

        return DummyTrajectory(t=t, y=y, run_id=run_id)


def make_generator():
    return GenerateODEDatasetUseCase(
        simulator=DummySimulator(),
        target_cols=["x", "y"],
    )


def test_generate_dataset_with_fixed_params_and_fixed_x0():
    ode = DummyODE(params=DummyParams(alpha=1.0, beta=0.1))
    generator = make_generator()

    df = generator.execute(
        ode=ode,
        x0=[1.0, 2.0],
        t_span=(0, 1),
        n_steps=3,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "run_id" not in df.columns

    assert list(df.columns) == ["x", "y", "t", "alpha", "beta"]

    np.testing.assert_allclose(df["t"].values, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(df["x"].values, [1.0, 1.5, 2.0])
    np.testing.assert_allclose(df["y"].values, [2.0, 2.05, 2.1])

    assert (df["alpha"] == 1.0).all()
    assert (df["beta"] == 0.1).all()


def test_generate_dataset_with_fixed_params_and_sampled_x0():
    ode = DummyODE(params=DummyParams(alpha=1.0, beta=0.1))
    generator = make_generator()

    def x0_sampler(rng):
        return np.array([rng.uniform(0, 1), rng.uniform(1, 2)])

    df = generator.execute(
        ode=ode,
        n_sims=4,
        x0_sampler=x0_sampler,
        t_span=(0, 1),
        n_steps=3,
        seed=123,
    )

    assert len(df) == 12
    assert "run_id" in df.columns
    assert sorted(df["run_id"].unique()) == [0, 1, 2, 3]

    assert (df["alpha"] == 1.0).all()
    assert (df["beta"] == 0.1).all()

    starts = df.groupby("run_id")[["x", "y"]].first()
    assert len(starts.drop_duplicates()) == 4


def test_generate_dataset_with_sampled_params_and_fixed_x0():
    ode = DummyODE(params=DummyParams(alpha=1.0, beta=0.1))
    generator = make_generator()

    def param_sampler(rng):
        return DummyParams(
            alpha=float(rng.uniform(1.0, 2.0)),
            beta=float(rng.uniform(0.1, 0.2)),
        )

    df = generator.execute(
        ode=ode,
        x0=[1.0, 2.0],
        n_sims=4,
        param_sampler=param_sampler,
        t_span=(0, 1),
        n_steps=3,
        seed=123,
    )

    assert len(df) == 12
    assert "run_id" in df.columns
    assert sorted(df["run_id"].unique()) == [0, 1, 2, 3]

    starts = df.groupby("run_id")[["x", "y"]].first()
    np.testing.assert_allclose(starts["x"].values, np.ones(4))
    np.testing.assert_allclose(starts["y"].values, np.full(4, 2.0))

    sampled_params = df.groupby("run_id")[["alpha", "beta"]].first()
    assert len(sampled_params.drop_duplicates()) == 4


def test_generate_dataset_with_sampled_params_and_sampled_x0():
    ode = DummyODE(params=DummyParams(alpha=1.0, beta=0.1))
    generator = make_generator()

    def x0_sampler(rng):
        return np.array(
            [
                rng.uniform(0.5, 2.0),
                rng.uniform(0.5, 2.0),
            ]
        )

    def param_sampler(rng):
        return DummyParams(
            alpha=float(rng.uniform(1.0, 2.0)),
            beta=float(rng.uniform(0.1, 0.2)),
        )

    df = generator.execute(
        ode=ode,
        n_sims=4,
        x0_sampler=x0_sampler,
        param_sampler=param_sampler,
        t_span=(0, 1),
        n_steps=3,
        seed=123,
    )

    assert len(df) == 12
    assert "run_id" in df.columns
    assert sorted(df["run_id"].unique()) == [0, 1, 2, 3]

    starts = df.groupby("run_id")[["x", "y"]].first()
    params = df.groupby("run_id")[["alpha", "beta"]].first()

    assert len(starts.drop_duplicates()) == 4
    assert len(params.drop_duplicates()) == 4


def test_generate_dataset_is_reproducible_with_same_seed():
    ode_1 = DummyODE(params=DummyParams(alpha=1.0, beta=0.1))
    ode_2 = DummyODE(params=DummyParams(alpha=1.0, beta=0.1))

    generator = make_generator()

    def x0_sampler(rng):
        return rng.uniform(0.5, 2.0, size=2)

    def param_sampler(rng):
        return DummyParams(
            alpha=float(rng.uniform(1.0, 2.0)),
            beta=float(rng.uniform(0.1, 0.2)),
        )

    df_1 = generator.execute(
        ode=ode_1,
        n_sims=5,
        x0_sampler=x0_sampler,
        param_sampler=param_sampler,
        seed=42,
    )

    df_2 = generator.execute(
        ode=ode_2,
        n_sims=5,
        x0_sampler=x0_sampler,
        param_sampler=param_sampler,
        seed=42,
    )

    pd.testing.assert_frame_equal(df_1, df_2)
