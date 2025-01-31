# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>

import time
from logging import getLogger
from typing import Any
from .ja import Coef, solve, SolverError, Model
from .mag import HysteresisLoop, hm_to_hb


def run(
    ref: HysteresisLoop | None,
    model: Model,
    initial_coef: Coef,
    initial_H_amp: tuple[float, float],
) -> None:
    """
    Runs the interactive GUI blockingly until stopped.
    """
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output, State
    import plotly.graph_objects as go

    app = dash.Dash(__name__)
    font_family = "'Ubuntu Mono', 'Consolas', monospace"
    app.layout = html.Div(
        style={
            "display": "flex",
            "flexDirection": "row",
            "margin": "0",
            "padding": "0",
            "boxSizing": "border-box",
            "fontFamily": font_family,
        },
        children=[
            # LEFT PANEL (fixed width = 400px)
            html.Div(
                style={
                    "width": "400px",
                    "padding": "10px",
                    "boxSizing": "border-box",
                    "overflowY": "auto",  # scroll if content exceeds
                },
                children=[
                    html.Div([html.H3("JA coefficients")]),
                    html.Table(
                        [
                            html.Tr(
                                [
                                    html.Td("c_r"),
                                    html.Td(
                                        dcc.Input(
                                            id="input-c_r",
                                            type="number",
                                            value=initial_coef.c_r,
                                            min=1e-12,
                                            max=1 - 1e-12,
                                        )
                                    ),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("M_s"),
                                    html.Td(
                                        dcc.Input(
                                            id="input-M_s",
                                            type="number",
                                            value=initial_coef.M_s,
                                            min=1e-3,
                                            max=9e6,
                                        )
                                    ),
                                    html.Td("A/m"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("a"),
                                    html.Td(
                                        dcc.Input(
                                            id="input-a",
                                            type="number",
                                            value=initial_coef.a,
                                            min=1e-3,
                                            max=3e6,
                                        )
                                    ),
                                    html.Td("A/m"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("k_p"),
                                    html.Td(
                                        dcc.Input(
                                            id="input-k_p",
                                            type="number",
                                            value=initial_coef.k_p,
                                            min=1e-3,
                                            max=3e6,
                                        )
                                    ),
                                    html.Td("A/m"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("α"),
                                    html.Td(
                                        dcc.Input(
                                            id="input-alpha",
                                            type="number",
                                            value=initial_coef.alpha,
                                            min=0,
                                            max=10,
                                        )
                                    ),
                                ]
                            ),
                        ]
                    ),
                    html.Div([html.H3("Solver parameters")]),
                    html.Table(
                        [
                            html.Tr(
                                [
                                    html.Td("H amplitude min"),
                                    html.Td(
                                        dcc.Input(
                                            id="input-H_amp_min",
                                            type="number",
                                            value=initial_H_amp[0],
                                            min=1,
                                            max=9e6,
                                        )
                                    ),
                                    html.Td("A/m"),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("H amplitude max"),
                                    html.Td(
                                        dcc.Input(
                                            id="input-H_amp_max",
                                            type="number",
                                            value=initial_H_amp[1],
                                            min=1,
                                            max=9e6,
                                        )
                                    ),
                                    html.Td("A/m"),
                                ]
                            ),
                        ]
                    ),
                    html.Div([html.H3("Controls")]),
                    html.Button("Evaluate ⟳", id="submit-button", n_clicks=0, style={"font-family": font_family}),
                    html.Div([html.H3("Invocation arguments")]),
                    html.Div(
                        id="command-text",
                        style={
                            "marginTop": "10px",
                            "whiteSpace": "pre-wrap",
                            "wordWrap": "break-word",
                        },
                    ),
                    html.Div([html.H3("Last evaluation status")]),
                    html.Div(
                        id="message",
                        style={
                            "marginTop": "10px",
                            "whiteSpace": "pre-wrap",
                            "wordWrap": "break-word",
                        },
                    ),
                ],
            ),
            # Main area
            html.Div(
                style={
                    "flex": "1",  # fill remaining horizontal space
                    "display": "flex",  # flex container so children can expand
                    "flexDirection": "column",  # ensure child can fill vertical space
                    "padding": "10px",
                    "boxSizing": "border-box",
                },
                children=[
                    dcc.Loading(
                        id="loading-indicator",
                        type="default",
                        style={"flex": "1", "display": "flex", "flexDirection": "column"},
                        children=dcc.Graph(
                            id="scatter-plot",
                            style={"flex": "1", "width": "100%"},
                            config={"scrollZoom": False},
                        ),
                    ),
                ],
            ),
        ],
    )

    @app.callback(
        [
            Output("scatter-plot", "figure"),
            Output("command-text", "children"),
            Output("message", "children"),
            Output("message", "style"),
        ],
        Input("submit-button", "n_clicks"),
        [
            State("message", "style"),
            State("input-c_r", "value"),
            State("input-M_s", "value"),
            State("input-a", "value"),
            State("input-k_p", "value"),
            State("input-alpha", "value"),
            State("input-H_amp_min", "value"),
            State("input-H_amp_max", "value"),
        ],
        prevent_initial_call=False,
    )
    def update(
        n_clicks: int,
        msg_style: dict[str, Any],
        c_r: float,
        M_s: float,
        a: float,
        k_p: float,
        alpha: float,
        H_amp_min: float,
        H_amp_max: float,
    ) -> None:
        _ = n_clicks
        started_at = time.monotonic()
        H_amp = (min(H_amp_min, H_amp_max), H_amp_max)
        _logger.info("Recomputing: c_r=%f, M_s=%f, a=%f, k_p=%f, α=%f, H_amp=%s", c_r, M_s, a, k_p, alpha, H_amp)

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
        fig.update_layout(title="J(H)", xaxis_title="H [A/m]", yaxis_title="B [T]", height=1000)

        # Generate output messages.
        command_text = (
            f"model={model.name.lower()} c_r={c_r} M_s={M_s} a={a} k_p={k_p} alpha={alpha} "
            f"H_amp_min={H_amp[0]} H_amp_max={H_amp[1]}"
        )
        if exception:
            msg = f"{type(exception).__name__}: {exception}"
            msg_style["color"] = "red"
        else:
            msg = f"Computed successfully in {time.monotonic() - started_at:.3f} s"
            msg_style["color"] = "green"

        _logger.info("Recomputing done in %.3f s", time.monotonic() - started_at)
        # noinspection PyTypeChecker
        return fig, command_text, msg, msg_style

    app.run_server(debug=True, use_reloader=False)


_logger = getLogger(__name__)
