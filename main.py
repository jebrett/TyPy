#!/usr/bin/env python3
"""Dash application for exploring three-phase tie-simplexes."""

from __future__ import annotations

from dash import Dash, Input, Output, Patch, ctx, dcc, html, no_update
from dash.exceptions import MissingCallbackContextException

from geometry import generate_simplex_family, make_ellipsoid, nearest_simplex
from visualization import (
    REGIONS,
    REGION_TRACE_INDICES,
    SELECTED_FILL_TRACE_INDEX,
    SELECTED_LINE_TRACE_INDEX,
    build_figure,
    empty_figure,
)

DEFAULTS = {
    "length": 3.0,
    "width": 2.0,
    "height": 2.0,
    "ellipsoid_theta": 90.0,
    "ellipsoid_phi": 45.0,
    "rho": 5.0,
    "simplex_theta": 90.0,
    "simplex_phi": 45.0,
    "maximum_alpha": 45.0,
}
FAMILY_SIZE = 500
REGION_COLOR_IDS = {
    f"{key}-color": (index, key)
    for index, (key, _, _) in enumerate(REGIONS)
}


def slider(label: str, identifier: str, minimum: float, maximum: float, value: float, step: float = 1.0):
    return html.Div(
        [
            html.Label(label, htmlFor=identifier),
            dcc.Slider(
                id=identifier,
                min=minimum,
                max=maximum,
                value=value,
                step=step,
                marks={minimum: f"{minimum:g}", maximum: f"{maximum:g}"},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ],
        className="control",
    )


def control_group(title: str, children, *, expanded: bool = True):
    return html.Details(
        [
            html.Summary(title),
            html.Div(children, className="control-group-body"),
        ],
        open=expanded,
        className="control-group",
    )


def color_control(label: str, identifier: str, value: str):
    return html.Div(
        [
            html.Label(label, htmlFor=identifier),
            dcc.Input(id=identifier, type="color", value=value),
        ],
        className="color-control",
    )


app = Dash(__name__)
app.title = "Three-phase simplex viewer"
server = app.server

app.layout = html.Div(
    [
        html.Header(
            [
                html.H1("Three-phase equilibrium viewer"),
                html.P(
                    "Explore tie-simplexes and the associative, segregative, "
                    "and three-phase regions inside an ellipsoidal phase boundary."
                ),
            ]
        ),
        html.Main(
            [
                html.Aside(
                    [
                        control_group(
                            "Ellipsoid controls",
                            [
                                slider(
                                    "Length (semi-axis)",
                                    "length",
                                    0,
                                    10,
                                    DEFAULTS["length"],
                                    0.1,
                                ),
                                slider(
                                    "Width (semi-axis)",
                                    "width",
                                    0,
                                    10,
                                    DEFAULTS["width"],
                                    0.1,
                                ),
                                slider(
                                    "Height (semi-axis)",
                                    "height",
                                    0,
                                    10,
                                    DEFAULTS["height"],
                                    0.1,
                                ),
                                slider(
                                    "Direction theta (zenith)",
                                    "ellipsoid-theta",
                                    0,
                                    180,
                                    DEFAULTS["ellipsoid_theta"],
                                ),
                                slider(
                                    "Direction phi (azimuth)",
                                    "ellipsoid-phi",
                                    0,
                                    360,
                                    DEFAULTS["ellipsoid_phi"],
                                ),
                                slider(
                                    "Center rho",
                                    "rho",
                                    0,
                                    10,
                                    DEFAULTS["rho"],
                                    0.1,
                                ),
                            ],
                        ),
                        control_group(
                            "Region appearance",
                            [
                                dcc.Checklist(
                                    id="region-visibility",
                                    options=[
                                        {"label": name, "value": key}
                                        for key, name, _ in REGIONS
                                    ],
                                    value=[key for key, _, _ in REGIONS],
                                    className="region-checklist",
                                ),
                                html.Div(
                                    [
                                        color_control(name, f"{key}-color", color)
                                        for key, name, color in REGIONS
                                    ],
                                    className="color-grid",
                                ),
                            ],
                        ),
                        control_group(
                            "Simplex controls",
                            [
                                slider(
                                    "Initial theta (zenith)",
                                    "simplex-theta",
                                    0,
                                    180,
                                    DEFAULTS["simplex_theta"],
                                ),
                                slider(
                                    "Phi (azimuth)",
                                    "simplex-phi",
                                    0,
                                    360,
                                    DEFAULTS["simplex_phi"],
                                ),
                                slider(
                                    "Maximum alpha",
                                    "maximum-alpha",
                                    0,
                                    89.5,
                                    DEFAULTS["maximum_alpha"],
                                    0.5,
                                ),
                                html.Div(
                                    [
                                        html.Label("Alpha window (z1 to z2)"),
                                        dcc.RangeSlider(
                                            id="z-window",
                                            min=-1,
                                            max=1,
                                            value=[-0.5, 0.5],
                                            step=0.01,
                                            allowCross=False,
                                            tooltip={"placement": "bottom"},
                                        ),
                                    ],
                                    className="control",
                                ),
                                html.Div(
                                    [
                                        html.Label("Orange slice: main-vertex z"),
                                        dcc.Slider(
                                            id="slice-z",
                                            min=-1,
                                            max=1,
                                            value=0,
                                            step=0.01,
                                            tooltip={
                                                "placement": "bottom",
                                                "always_visible": True,
                                            },
                                        ),
                                    ],
                                    className="control",
                                ),
                            ],
                        ),
                    ],
                    className="controls",
                ),
                html.Section(
                    [
                        dcc.Loading(dcc.Graph(id="phase-graph", className="graph")),
                        html.Div(id="geometry-status", className="status"),
                    ],
                    className="viewer",
                ),
            ]
        ),
    ],
    className="page",
)


def _central_family(
    length: float,
    width: float,
    height: float,
    ellipsoid_theta: float,
    ellipsoid_phi: float,
    rho: float,
    simplex_theta: float,
    simplex_phi: float,
):
    ellipsoid = make_ellipsoid(
        [length, width, height],
        rho,
        ellipsoid_theta,
        ellipsoid_phi,
    )
    family = generate_simplex_family(
        ellipsoid,
        simplex_phi,
        0,
        0,
        0,
        initial_theta=simplex_theta,
        target_count=FAMILY_SIZE,
    )
    return ellipsoid, family


@app.callback(
    Output("z-window", "min"),
    Output("z-window", "max"),
    Output("z-window", "step"),
    Output("z-window", "value"),
    Output("z-window", "marks"),
    Output("slice-z", "min"),
    Output("slice-z", "max"),
    Output("slice-z", "step"),
    Output("slice-z", "value"),
    Output("slice-z", "marks"),
    Input("length", "value"),
    Input("width", "value"),
    Input("height", "value"),
    Input("ellipsoid-theta", "value"),
    Input("ellipsoid-phi", "value"),
    Input("rho", "value"),
    Input("simplex-theta", "value"),
    Input("simplex-phi", "value"),
)
def update_z_controls(
    length,
    width,
    height,
    ellipsoid_theta,
    ellipsoid_phi,
    rho,
    simplex_theta,
    simplex_phi,
):
    _, family = _central_family(
        length,
        width,
        height,
        ellipsoid_theta,
        ellipsoid_phi,
        rho,
        simplex_theta,
        simplex_phi,
    )
    if not family:
        marks = {-1: "-1", 1: "1"}
        return -1, 1, 0.01, [-0.5, 0.5], marks, -1, 1, 0.01, 0, marks

    minimum = family[0].vertex_z
    maximum = family[-1].vertex_z
    span = maximum - minimum
    if span < 1e-7:
        minimum -= 0.01
        maximum += 0.01
        span = maximum - minimum
    step = span / max(len(family) - 1, 1)
    z1 = minimum + 0.25 * span
    z2 = minimum + 0.75 * span
    selected = nearest_simplex(family, theta=simplex_theta).vertex_z
    marks = {
        minimum: f"{minimum:.2f}",
        maximum: f"{maximum:.2f}",
    }
    return (
        minimum,
        maximum,
        step,
        [z1, z2],
        marks,
        minimum,
        maximum,
        step,
        selected,
        marks,
    )


@app.callback(
    Output("phase-graph", "figure"),
    Output("geometry-status", "children"),
    Input("length", "value"),
    Input("width", "value"),
    Input("height", "value"),
    Input("ellipsoid-theta", "value"),
    Input("ellipsoid-phi", "value"),
    Input("rho", "value"),
    Input("simplex-theta", "value"),
    Input("simplex-phi", "value"),
    Input("maximum-alpha", "value"),
    Input("z-window", "value"),
    Input("slice-z", "value"),
    Input("region-visibility", "value"),
    Input("three-phase-color", "value"),
    Input("segregative-color", "value"),
    Input("associative-left-color", "value"),
    Input("associative-right-color", "value"),
)
def update_figure(
    length,
    width,
    height,
    ellipsoid_theta,
    ellipsoid_phi,
    rho,
    simplex_theta,
    simplex_phi,
    maximum_alpha,
    z_window,
    selected_z,
    visible_regions=None,
    three_phase_color=None,
    segregative_color=None,
    associative_left_color=None,
    associative_right_color=None,
):
    try:
        triggered_id = ctx.triggered_id
    except MissingCallbackContextException:
        triggered_id = None
    return update_figure_for_trigger(
        triggered_id,
        length,
        width,
        height,
        ellipsoid_theta,
        ellipsoid_phi,
        rho,
        simplex_theta,
        simplex_phi,
        maximum_alpha,
        z_window,
        selected_z,
        visible_regions,
        three_phase_color,
        segregative_color,
        associative_left_color,
        associative_right_color,
    )


def selected_simplex_patch(selected):
    patch = Patch()
    vertices = selected.vertices
    closed = vertices[[0, 1, 2, 0]]
    for axis, coordinate in zip(("x", "y", "z"), vertices.T):
        patch["data"][SELECTED_FILL_TRACE_INDEX][axis] = coordinate.tolist()
    patch["data"][SELECTED_FILL_TRACE_INDEX]["visible"] = selected.alpha > 1e-8
    for axis, coordinate in zip(("x", "y", "z"), closed.T):
        patch["data"][SELECTED_LINE_TRACE_INDEX][axis] = coordinate.tolist()
    return patch


def region_visibility_patch(visible_regions):
    visible = set(visible_regions or [])
    patch = Patch()
    for index, (key, _, _) in zip(REGION_TRACE_INDICES, REGIONS):
        patch["data"][index]["visible"] = key in visible
    return patch


def region_color_patch(color_input_id, color):
    patch = Patch()
    trace_index, _ = REGION_COLOR_IDS[color_input_id]
    patch["data"][trace_index]["color"] = color
    return patch


def _selection_status(selected, family):
    simplex_kind = "1-simplex (line)" if selected.alpha <= 1e-8 else "2-simplex (triangle)"
    return (
        f"Selected {simplex_kind}: vertex z = {selected.vertex_z:.3f}, "
        f"theta = {selected.theta:.1f}°, alpha = {selected.alpha:.2f}°. "
        f"Rendered {len(family)} reachable simplexes."
    )


def update_figure_for_trigger(
    triggered_id,
    length,
    width,
    height,
    ellipsoid_theta,
    ellipsoid_phi,
    rho,
    simplex_theta,
    simplex_phi,
    maximum_alpha,
    z_window,
    selected_z,
    visible_regions,
    three_phase_color=None,
    segregative_color=None,
    associative_left_color=None,
    associative_right_color=None,
):
    if triggered_id == "region-visibility":
        return region_visibility_patch(visible_regions), no_update
    region_colors = {
        "three-phase": three_phase_color or REGIONS[0][2],
        "segregative": segregative_color or REGIONS[1][2],
        "associative-left": associative_left_color or REGIONS[2][2],
        "associative-right": associative_right_color or REGIONS[3][2],
    }
    if triggered_id in REGION_COLOR_IDS:
        _, region_key = REGION_COLOR_IDS[triggered_id]
        return region_color_patch(triggered_id, region_colors[region_key]), no_update

    ellipsoid = make_ellipsoid(
        [length, width, height],
        rho,
        ellipsoid_theta,
        ellipsoid_phi,
    )
    family = generate_simplex_family(
        ellipsoid,
        simplex_phi,
        z_window[0],
        z_window[1],
        maximum_alpha,
        initial_theta=simplex_theta,
        target_count=FAMILY_SIZE,
    )
    if not family:
        message = (
            "No origin-based entry simplex is reachable for these controls. "
            "Move the ellipsoid away from the origin or adjust its direction."
        )
        return empty_figure(message), message

    selected = nearest_simplex(family, vertex_z=selected_z)
    status = _selection_status(selected, family)
    if triggered_id == "slice-z":
        return selected_simplex_patch(selected), status

    region_keys = (
        set(visible_regions)
        if visible_regions is not None
        else {key for key, _, _ in REGIONS}
    )
    figure, selected = build_figure(
        ellipsoid,
        family,
        selected_z,
        visible_regions=region_keys,
        region_colors=region_colors,
    )
    return figure, status


if __name__ == "__main__":
    app.run(debug=True)
