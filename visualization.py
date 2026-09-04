"""Plotly rendering helpers for the simplex visualizer."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from skimage import filters, measure

from geometry import (
    Ellipsoid,
    Simplex,
    clip_points_to_ellipsoid,
    ellipsoid_bounds,
    nearest_simplex,
    phase_grid,
)

REGIONS = (
    ("three-phase", "Three-phase coexistence", "#2ca02c"),
    ("segregative", "Segregative two-phase", "#d62728"),
    ("associative-left", "Associative two-phase (left)", "#1f77b4"),
    ("associative-right", "Associative two-phase (right)", "#17becf"),
)
REGION_TRACE_INDICES = tuple(range(len(REGIONS)))
SELECTED_FILL_TRACE_INDEX = 6
SELECTED_LINE_TRACE_INDEX = 7


def smooth_region_mesh(
    mask: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    smoothing: float = 0.65,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a closed triangular surface from a sampled region mask."""
    if not np.any(mask):
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)

    steps = np.array([axis[1] - axis[0] for axis in axes])
    padded = np.pad(mask.astype(float), 1)
    scalar_field = filters.gaussian(
        padded,
        sigma=smoothing,
        preserve_range=True,
        mode="constant",
    )
    if np.max(scalar_field) <= 0.5:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)

    vertices, faces, _, _ = measure.marching_cubes(
        scalar_field,
        level=0.5,
        spacing=tuple(steps),
        allow_degenerate=False,
    )
    origin = np.array([axis[0] for axis in axes]) - steps
    return vertices + origin, faces


def ellipsoid_surface(
    ellipsoid: Ellipsoid,
    longitude_samples: int = 48,
    latitude_samples: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    longitude = np.linspace(0.0, 2.0 * np.pi, longitude_samples)
    latitude = np.linspace(0.0, np.pi, latitude_samples)
    longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)
    local = np.column_stack(
        (
            ellipsoid.axes[0]
            * np.sin(latitude_grid).ravel()
            * np.cos(longitude_grid).ravel(),
            ellipsoid.axes[1]
            * np.sin(latitude_grid).ravel()
            * np.sin(longitude_grid).ravel(),
            ellipsoid.axes[2] * np.cos(latitude_grid).ravel(),
        )
    )
    global_points = local @ ellipsoid.basis.T + ellipsoid.center
    shape = latitude_grid.shape
    return (
        global_points[:, 0].reshape(shape),
        global_points[:, 1].reshape(shape),
        global_points[:, 2].reshape(shape),
    )


def _family_coordinates(family: list[Simplex]) -> tuple[list[float | None], ...]:
    coordinates: tuple[list[float | None], ...] = ([], [], [])
    for simplex in family:
        closed = simplex.vertices[[0, 1, 2, 0]]
        for axis in range(3):
            coordinates[axis].extend(closed[:, axis].tolist())
            coordinates[axis].append(None)
    return coordinates


def build_figure(
    ellipsoid: Ellipsoid,
    family: list[Simplex],
    selected_z: float,
    *,
    grid_resolution: int = 30,
    visible_regions: set[str] | None = None,
    region_colors: dict[str, str] | None = None,
) -> tuple[go.Figure, Simplex]:
    """Build the complete phase-region figure and return the selected simplex."""
    if not family:
        raise ValueError("No valid simplex family can be rendered.")

    selected = nearest_simplex(family, vertex_z=selected_z)
    figure = go.Figure()
    axes, labels = phase_grid(ellipsoid, family, resolution=grid_resolution)
    if visible_regions is None:
        visible_regions = {key for key, _, _ in REGIONS}
    region_colors = region_colors or {}
    for label, (key, name, color) in enumerate(REGIONS):
        vertices, triangles = smooth_region_mesh(labels == label, axes)
        if len(vertices):
            vertices = clip_points_to_ellipsoid(vertices, ellipsoid)
        figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0] if len(vertices) else [],
                y=vertices[:, 1] if len(vertices) else [],
                z=vertices[:, 2] if len(vertices) else [],
                i=triangles[:, 0] if len(triangles) else [],
                j=triangles[:, 1] if len(triangles) else [],
                k=triangles[:, 2] if len(triangles) else [],
                color=region_colors.get(key, color),
                opacity=0.18,
                flatshading=False,
                lighting={
                    "ambient": 0.7,
                    "diffuse": 0.75,
                    "fresnel": 0.15,
                    "roughness": 0.65,
                    "specular": 0.2,
                },
                name=name,
                hoverinfo="name",
                visible=key in visible_regions,
            )
        )

    surface_x, surface_y, surface_z = ellipsoid_surface(ellipsoid)
    figure.add_trace(
        go.Surface(
            x=surface_x,
            y=surface_y,
            z=surface_z,
            colorscale=[[0, "#67748e"], [1, "#67748e"]],
            opacity=0.12,
            showscale=False,
            name="Ellipsoidal phase boundary",
            hoverinfo="skip",
            contours={
                "x": {"show": True, "color": "#67748e", "width": 1},
                "y": {"show": True, "color": "#67748e", "width": 1},
                "z": {"show": True, "color": "#67748e", "width": 1},
            },
        )
    )

    family_x, family_y, family_z = _family_coordinates(family)
    figure.add_trace(
        go.Scatter3d(
            x=family_x,
            y=family_y,
            z=family_z,
            mode="lines",
            line={"color": "rgba(65, 65, 65, 0.28)", "width": 2},
            name="Simplex family",
            hoverinfo="skip",
        )
    )

    selected_vertices = selected.vertices
    figure.add_trace(
        go.Mesh3d(
            x=selected_vertices[:, 0],
            y=selected_vertices[:, 1],
            z=selected_vertices[:, 2],
            i=[0],
            j=[1],
            k=[2],
            color="#ff7f0e",
            opacity=0.65,
            name="Selected 2-simplex",
            visible=selected.alpha > 1e-8,
        )
    )
    closed = selected_vertices[[0, 1, 2, 0]]
    figure.add_trace(
        go.Scatter3d(
            x=closed[:, 0],
            y=closed[:, 1],
            z=closed[:, 2],
            mode="lines+markers",
            line={"color": "#ff7f0e", "width": 8},
            marker={"color": "#ff7f0e", "size": 4},
            name="Selected simplex",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode="markers",
            marker={"color": "black", "size": 4},
            name="Origin",
        )
    )

    lower, upper = ellipsoid_bounds(ellipsoid)
    padding = max(np.max(upper - lower) * 0.08, 0.2)
    figure.update_layout(
        title="Three-phase equilibrium simplex viewer",
        scene={
            "xaxis": {
                "title": "Component 1",
                "range": [lower[0] - padding, upper[0] + padding],
            },
            "yaxis": {
                "title": "Component 2",
                "range": [lower[1] - padding, upper[1] + padding],
            },
            "zaxis": {
                "title": "Component 3",
                "range": [lower[2] - padding, upper[2] + padding],
            },
            "aspectmode": "data",
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01},
        margin={"l": 0, "r": 0, "b": 0, "t": 80},
        uirevision="simplex-view",
    )
    return figure, selected


def empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 18, "color": "#a61b1b"},
    )
    figure.update_layout(
        scene={"aspectmode": "data"},
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return figure
