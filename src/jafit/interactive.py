# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>

import time
from logging import getLogger
from typing import Any
from .ja import Coef, solve, SolverError, Model
from .mag import HysteresisLoop, hm_to_hj, hm_to_hb
from . import loss


_PLOT_MODES = {
    "J(H)": (hm_to_hj, "J [T]"),
    "B(H)": (hm_to_hb, "B [T]"),
    "M(H)": (lambda x: x, "M [A/m]"),
}


def run(
    ref: HysteresisLoop | None,
    initial_model: Model,
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

    app = dash.Dash(__name__, title="Jiles-Atherton interactive solver")
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
            # SIDE PANEL
            html.Div(
                style={
                    "width": "300px",
                    "padding": "10px",
                    "boxSizing": "border-box",
                    "overflowY": "auto",
                    "font-size": "12px",
                },
                children=[
                    html.Div([html.H3("Jiles-Atherton parameters")]),
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
                                            min=1e-12,
                                            max=100,
                                        )
                                    ),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Model"),
                                    html.Td(
                                        dcc.RadioItems(
                                            id="ja-model",
                                            options=[{"label": k.name.lower(), "value": k.name} for k in Model],
                                            value=initial_model.name,
                                        ),
                                    ),
                                ]
                            ),
                        ],
                    ),
                    html.Button(
                        "Solve ⟳",
                        id="solve-button",
                        type="submit",
                        n_clicks=0,
                        style={"font-family": font_family, "width": "100%"},
                    ),
                    html.Div([html.H3("Solver and plot parameters")]),
                    html.Table(
                        [
                            html.Tr(
                                [
                                    html.Td("|H| min"),
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
                                    html.Td("|H| max"),
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
                            html.Tr(
                                [
                                    html.Td("Plot"),
                                    html.Td(
                                        dcc.RadioItems(
                                            id="plot-mode",
                                            options=[{"label": k, "value": k} for k in _PLOT_MODES],
                                            value=list(_PLOT_MODES)[0],
                                            labelStyle={"display": "inline-block", "margin-right": "0px"},
                                        ),
                                    ),
                                ]
                            ),
                            html.Tr(
                                [
                                    html.Td("Quality"),
                                    html.Td(
                                        dcc.RadioItems(
                                            id="quality",
                                            options=[
                                                {"label": "Fast", "value": 0},
                                                {"label": "Precise", "value": 1},
                                            ],
                                            value=0,
                                            labelStyle={"display": "inline-block", "margin-right": "0px"},
                                        ),
                                    ),
                                ]
                            ),
                        ]
                    ),
                    html.Div([html.H3("Command line")]),
                    html.Div(
                        id="command-text",
                        style={"marginTop": "10px", "whiteSpace": "pre-wrap", "wordWrap": "break-word"},
                    ),
                    html.Div([html.H3("Last solution status")]),
                    html.Div(
                        id="message",
                        style={"marginTop": "10px", "whiteSpace": "pre-wrap", "wordWrap": "break-word"},
                    ),
                ],
            ),
            # MAIN AREA
            html.Div(
                style={
                    "flex": "1",  # fill remaining horizontal space
                    "display": "flex",
                    "flexDirection": "column",
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
        [
            Input("solve-button", "n_clicks"),
            Input("input-c_r", "n_submit"),
            Input("input-M_s", "n_submit"),
            Input("input-a", "n_submit"),
            Input("input-k_p", "n_submit"),
            Input("input-alpha", "n_submit"),
        ],
        [
            State("message", "style"),
            State("input-c_r", "value"),
            State("input-M_s", "value"),
            State("input-a", "value"),
            State("input-k_p", "value"),
            State("input-alpha", "value"),
            State("ja-model", "value"),
            State("input-H_amp_min", "value"),
            State("input-H_amp_max", "value"),
            State("plot-mode", "value"),
            State("quality", "value"),
        ],
        prevent_initial_call=False,
    )
    def update(
        _btn_clicks: int,
        _c_r_submit: int,
        _M_s_submit: int,
        _a_submit: int,
        _k_p_submit: int,
        _alpha_submit: int,
        msg_style: dict[str, Any],
        c_r: float,
        M_s: float,
        a: float,
        k_p: float,
        alpha: float,
        ja_model_name: str,
        H_amp_min: float,
        H_amp_max: float,
        plot_mode_name: str,
        quality: int,
    ) -> Any:
        started_at = time.monotonic()
        H_amp = (min(H_amp_min, H_amp_max), H_amp_max)
        hm_to_plot, y_label = _PLOT_MODES[plot_mode_name]
        model = Model[ja_model_name.upper().strip()]
        assert isinstance(model, Model)
        _logger.info("Solving: c_r=%f, M_s=%f, a=%f, k_p=%f, α=%f, H_amp=%s", c_r, M_s, a, k_p, alpha, H_amp)

        # Solve the system.
        sol: HysteresisLoop | None = None
        exception: Exception | None = None
        try:
            solution = solve(
                model=model,
                coef=Coef(c_r=c_r, M_s=M_s, a=a, k_p=k_p, alpha=alpha),
                H_stop=H_amp,
                fast=quality < 1,
            )
            branches = solution.branches
            sol = HysteresisLoop(descending=solution.last_descending[::-1], ascending=solution.last_ascending)
        except SolverError as ex:
            exception, branches = ex, ex.partial_curves
        except Exception as ex:
            exception, branches = ex, []

        # Compute losses
        losses: dict[str, float] = {}
        if ref is not None and sol is not None:
            losses["nearest"] = loss.make_nearest(ref)(sol)
            losses["key_points"] = loss.make_demag_key_points(ref)(sol)
            losses["magnetization"] = loss.make_magnetization(ref)(sol)

        # Make the plot.
        fig = go.Figure()
        for idx, br in enumerate(branches):
            hb = hm_to_plot(br)
            fig.add_trace(go.Scattergl(x=hb[:, 0], y=hb[:, 1], mode="lines", name=f"JA #{idx}", line=dict(width=2)))
        if ref is not None:
            ref_params = dict(mode="markers", marker=dict(size=3))
            if len(ref.descending):
                hb = hm_to_plot(ref.descending)
                fig.add_trace(go.Scattergl(x=hb[:, 0], y=hb[:, 1], name="reference descending", **ref_params))
            if len(ref.ascending):
                hb = hm_to_plot(ref.ascending)
                fig.add_trace(go.Scattergl(x=hb[:, 0], y=hb[:, 1], name="reference ascending", **ref_params))
        fig.update_layout(
            xaxis_title="H [A/m]",
            yaxis_title=y_label,
            height=1000,
            margin=dict(l=20, r=20, t=20, b=20),
            hovermode="x",
        )

        # Generate output messages.
        command_text = (
            f"jafit model={model.name.lower()}"
            f" c_r={c_r} M_s={M_s} a={a} k_p={k_p} alpha={alpha}"
            f" H_amp_min={H_amp[0]} H_amp_max={H_amp[1]}"
        )
        if exception:
            msg_style["color"] = "#800"
            msg = f"{type(exception).__name__}: {exception}"
        else:
            msg_style["color"] = "#080"
            msg = f"Solved in {time.monotonic() - started_at:.3f} s"
        if losses:
            msg += "\nLosses:"
            for k, v in losses.items():
                k = k.ljust(max(map(len, losses.keys())))
                msg += f"\n{k}= {v}"

        _logger.info("Solved in %.3f s", time.monotonic() - started_at)
        # noinspection PyTypeChecker
        return fig, command_text, msg, msg_style

    app.run_server(debug=True, use_reloader=False)


_logger = getLogger(__name__)
