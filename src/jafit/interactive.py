# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>
import numpy as np
import numpy.typing as npt
from .ja import Solution, Coef, solve, SolverError, Model
from .mag import HysteresisLoop, hm_to_hb


def run(
    ref: HysteresisLoop | None,
    *,
    model: Model,
    initial_coef: Coef,
    H_amp: tuple[float, float],
) -> None:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output, State
    import plotly.graph_objects as go

    app = dash.Dash(__name__)
    # TODO: add model selection
    # TODO: add H min-max selection
    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Label("c_r"),
                    dcc.Input(id="input-c_r", type="number", value=initial_coef.c_r),
                ],
                style={"margin": "5px"},
            ),
            html.Div(
                [
                    html.Label("M_s"),
                    dcc.Input(id="input-M_s", type="number", value=initial_coef.M_s),
                ],
                style={"margin": "5px"},
            ),
            html.Div(
                [
                    html.Label("a"),
                    dcc.Input(id="input-a", type="number", value=initial_coef.a),
                ],
                style={"margin": "5px"},
            ),
            html.Div(
                [
                    html.Label("k_p"),
                    dcc.Input(id="input-k_p", type="number", value=initial_coef.k_p),
                ],
                style={"margin": "5px"},
            ),
            html.Div(
                [
                    html.Label("α"),
                    dcc.Input(id="input-alpha", type="number", value=initial_coef.alpha),
                ],
                style={"margin": "5px"},
            ),
            html.Button("Submit", id="submit-button", n_clicks=0),
            html.Div(
                id="message",
                style={
                    "color": "white",
                    "backgroundColor": "red",
                    "padding": "10px",
                    "marginTop": "10px",
                    "display": "none",  # hidden by default
                },
            ),
            dcc.Loading(id="loading-indicator", type="default", children=dcc.Graph(id="scatter-plot")),
        ]
    )

    @app.callback(
        [Output("scatter-plot", "figure"), Output("message", "children"), Output("message", "style")],
        Input("submit-button", "n_clicks"),
        [
            State("input-c_r", "value"),
            State("input-M_s", "value"),
            State("input-a", "value"),
            State("input-k_p", "value"),
            State("input-alpha", "value"),
        ],
        prevent_initial_call=True,
    )
    def update_graph(n_clicks: int, c_r: float, M_s: float, a: float, k_p: float, alpha: float) -> None:
        _ = n_clicks

        # Solve the system.
        exception: Exception | None = None
        try:
            sol = solve(
                model=model,
                coef=Coef(c_r=c_r, M_s=M_s, a=a, k_p=k_p, alpha=alpha),
                H_stop=H_amp,
                fast=True,
            )
            branches = sol.branches
        except SolverError as ex:
            exception, branches = ex, ex.partial_curves
        except Exception as ex:
            exception, branches = ex, []

        # Make the plot.
        fig = go.Figure()
        for idx, br in enumerate(branches):
            hb = hm_to_hb(br)
            fig.add_trace(go.Scattergl(x=hb[:, 0], y=hb[:, 1], mode="lines", name=f"J(H) JA #{idx}"))
        if ref is not None:
            if len(ref.descending):
                hb = hm_to_hb(ref.descending)
                fig.add_trace(go.Scattergl(x=hb[:, 0], y=hb[:, 1], mode="markers", name="J(H) reference descending"))
            if len(ref.ascending):
                hb = hm_to_hb(ref.ascending)
                fig.add_trace(go.Scattergl(x=hb[:, 0], y=hb[:, 1], mode="markers", name="J(H) reference ascending"))
        title = f"model={model.name}, c_r={c_r}, M_s={M_s}, a={a}, k_p={k_p}, α={alpha}"
        fig.update_layout(title=title, xaxis_title="H [A/m]", yaxis_title="B [T]")

        # Generate output messages.
        if exception:
            msg = f"{type(exception).__name__}: {exception}"
            msg_style = {
                "color": "white",
                "backgroundColor": "red",
                "padding": "10px",
                "marginTop": "10px",
                "display": "block",
            }
        else:
            msg = title
            msg_style = {"display": "none"}  # hide

        # noinspection PyTypeChecker
        return fig, msg, msg_style

    app.run_server(debug=True)
