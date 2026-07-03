import numpy as np
import plotly.express as px
import plotly.graph_objects as go

GT_COLOR = px.colors.qualitative.Plotly[0]  # bleu
PRED_COLOR = px.colors.qualitative.Plotly[1]  # orange


def _as_2d_y(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)

    if y.ndim == 1:
        return y.reshape(-1, 1)

    if y.ndim != 2:
        raise ValueError("y must have shape (n_steps,) or (n_steps, n_states).")

    return y


def plot_trajectory(
    t: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray | None = None,
    state_names: list[str] | None = None,
    train_t: np.ndarray | None = None,  # <-- ce paramètre manque chez toi
    title: str = "Trajectory",
) -> go.Figure:
    t = np.asarray(t)
    y = _as_2d_y(y)

    if len(t) != y.shape[0]:
        raise ValueError("t and y must have the same number of time steps.")

    n_states = y.shape[1]
    state_names = state_names or [f"x{i}" for i in range(n_states)]

    fig = go.Figure()

    for i, name in enumerate(state_names):
        color = px.colors.qualitative.Plotly[i]
        fig.add_trace(
            go.Scatter(
                x=t,
                y=y[:, i],
                mode="lines+markers",
                name=name,
                line={"color": color},
                marker={"size": 6, "color": color},
            )
        )

        if y_pred is not None:
            y_pred = _as_2d_y(y_pred)
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=y_pred[:, i],
                    mode="lines+markers",
                    name=f"{name} pred",
                    line={"dash": "dash", "color": color},
                    marker={"size": 6, "symbol": "x", "color": color},
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="t",
        yaxis_title="state value",
        template="plotly_white",
        hovermode="x unified",
    )

    return fig


def plot_phase_space(
    y: np.ndarray,
    y_pred: np.ndarray | None = None,
    state_names: list[str] | None = None,
    x_idx: int = 0,
    y_idx: int = 1,
    title: str = "Phase space",
) -> go.Figure:
    y = _as_2d_y(y)

    if y.shape[1] < 2:
        raise ValueError("Phase-space plot requires at least 2 states.")

    n_states = y.shape[1]
    state_names = state_names or [f"x{i}" for i in range(n_states)]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=y[:, x_idx],
            y=y[:, y_idx],
            mode="lines",
            name="ground truth",
            line={"color": GT_COLOR},
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[y[0, x_idx]],
            y=[y[0, y_idx]],
            mode="markers",
            name="start",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[y[-1, x_idx]],
            y=[y[-1, y_idx]],
            mode="markers",
            name="end",
        )
    )

    if y_pred is not None:
        y_pred = _as_2d_y(y_pred)

        fig.add_trace(
            go.Scatter(
                x=y_pred[:, x_idx],
                y=y_pred[:, y_idx],
                mode="lines",
                name="prediction",
                line={"dash": "dash", "color": PRED_COLOR},
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=state_names[x_idx],
        yaxis_title=state_names[y_idx],
        template="plotly_white",
    )

    return fig
