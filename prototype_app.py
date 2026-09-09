#!/usr/bin/env python3
"""Side-by-side Trame prototype for comparing smooth region mesh methods."""

from __future__ import annotations

import json
from time import perf_counter

import psutil
import pyvista as pv
import vtk
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import html, vtk as vtk_widgets, vuetify3

from geometry import generate_simplex_family, make_ellipsoid
from prototype_meshes import (
    REGION_KEYS,
    direct_swept_meshes,
    mesh_metrics,
    partition_agreement,
    surface_nets_meshes,
)

COLORS = {
    "three-phase": "#087830",
    "segregative": "#d62728",
    "associative-left": "#1f77b4",
    "associative-right": "#17becf",
}
PRESETS = {
    "Default triangle": {
        "axes": [3, 2, 2],
        "rho": 5,
        "ellipsoid_theta": 90,
        "ellipsoid_phi": 45,
        "simplex_theta": 90,
        "simplex_phi": 45,
        "maximum_alpha": 45,
    },
    "Alpha zero transition": {
        "axes": [3, 2, 2],
        "rho": 5,
        "ellipsoid_theta": 90,
        "ellipsoid_phi": 45,
        "simplex_theta": 90,
        "simplex_phi": 45,
        "maximum_alpha": 0,
    },
    "Rotated ellipsoid": {
        "axes": [4, 2.2, 1.5],
        "rho": 6,
        "ellipsoid_theta": 55,
        "ellipsoid_phi": 125,
        "simplex_theta": 70,
        "simplex_phi": 120,
        "maximum_alpha": 42,
    },
    "Narrow axis": {
        "axes": [4, 1, 2.5],
        "rho": 6,
        "ellipsoid_theta": 80,
        "ellipsoid_phi": 20,
        "simplex_theta": 85,
        "simplex_phi": 20,
        "maximum_alpha": 35,
    },
    "High alpha": {
        "axes": [3.5, 2.5, 2.5],
        "rho": 6,
        "ellipsoid_theta": 90,
        "ellipsoid_phi": 45,
        "simplex_theta": 90,
        "simplex_phi": 45,
        "maximum_alpha": 85,
    },
}

server = get_server(client_type="vue3")
state, ctrl = server.state, server.controller
plotter = pv.Plotter(shape=(1, 2), off_screen=True, border=False)
shared_camera = vtk.vtkCamera()
for renderer in plotter.renderers:
    renderer.SetActiveCamera(shared_camera)
    renderer.SetBackground(0.96, 0.97, 0.99)

actors: dict[str, dict[str, object]] = {"direct": {}, "surface_nets": {}}
state.preset = "Default triangle"
state.surface_resolution = 64
state.opacity = 0.34
state.wireframe = False
state.metrics = "Generating initial comparison..."
state.busy = False
for key, color in COLORS.items():
    state[f"color_{key.replace('-', '_')}"] = color
    state[f"visible_{key.replace('-', '_')}"] = True


def build_geometry(preset_name: str):
    preset = PRESETS[preset_name]
    ellipsoid = make_ellipsoid(
        preset["axes"],
        preset["rho"],
        preset["ellipsoid_theta"],
        preset["ellipsoid_phi"],
    )
    base = generate_simplex_family(
        ellipsoid,
        preset["simplex_phi"],
        0,
        0,
        0,
        initial_theta=preset["simplex_theta"],
        target_count=500,
    )
    if len(base) != 500:
        raise ValueError("This preset cannot produce 500 reachable simplexes.")
    z1, z2 = base[125].vertex_z, base[375].vertex_z
    family = generate_simplex_family(
        ellipsoid,
        preset["simplex_phi"],
        z1,
        z2,
        preset["maximum_alpha"],
        initial_theta=preset["simplex_theta"],
        target_count=500,
    )
    if len(family) != 500:
        raise ValueError("This preset cannot produce 500 valid region simplexes.")
    return ellipsoid, family


def _clear_renderer(renderer_index: int):
    plotter.subplot(0, renderer_index)
    plotter.clear()
    actors["direct" if renderer_index == 0 else "surface_nets"].clear()


def _add_result(method: str, renderer_index: int, result):
    plotter.subplot(0, renderer_index)
    for key, mesh in result.meshes.items():
        if not mesh.n_points:
            continue
        actor = plotter.add_mesh(
            mesh,
            name=f"{method}-{key}",
            color=COLORS[key],
            opacity=state.opacity,
            smooth_shading=True,
            show_edges=state.wireframe,
            edge_color="#252525",
        )
        actor.SetVisibility(bool(state[f"visible_{key.replace('-', '_')}"]))
        actors[method][key] = actor
    plotter.add_text(
        "Direct analytical loft" if method == "direct" else "VTK multi-label Surface Nets",
        position="upper_left",
        font_size=11,
        color="#182230",
        name=f"{method}-title",
    )


def _method_metrics(result, ellipsoid):
    return {
        "generation_seconds": round(result.generation_seconds, 3),
        "grid_resolution": result.grid_resolution or "analytical",
        "regions": {
            key: mesh_metrics(mesh, ellipsoid)
            for key, mesh in result.meshes.items()
        },
    }


@ctrl.add("rebuild")
def rebuild():
    state.busy = True
    started_memory = psutil.Process().memory_info().rss
    started = perf_counter()
    try:
        ellipsoid, family = build_geometry(state.preset)
        direct = direct_swept_meshes(ellipsoid, family)
        surface_nets = surface_nets_meshes(
            ellipsoid,
            family,
            resolution=int(state.surface_resolution),
        )
        _clear_renderer(0)
        _clear_renderer(1)
        _add_result("direct", 0, direct)
        _add_result("surface_nets", 1, surface_nets)
        plotter.renderers[0].ResetCamera()
        memory_delta = max(psutil.Process().memory_info().rss - started_memory, 0)
        state.metrics = json.dumps(
            {
                "total_seconds": round(perf_counter() - started, 3),
                "memory_delta_mb": round(memory_delta / 1024**2, 2),
                "direct": {
                    **_method_metrics(direct, ellipsoid),
                    **partition_agreement(direct, ellipsoid, family),
                },
                "surface_nets": {
                    **_method_metrics(surface_nets, ellipsoid),
                    **partition_agreement(surface_nets, ellipsoid, family),
                },
            },
            indent=2,
        )
        ctrl.view_update()
    except Exception as error:
        state.metrics = f"Prototype generation failed explicitly:\n{error}"
    finally:
        state.busy = False


def initialize_client(**_):
    """Populate the scene for the first browser and refresh later clients."""
    if not actors["direct"] and not actors["surface_nets"]:
        rebuild()
    elif hasattr(ctrl, "view_update"):
        ctrl.view_update()


ctrl.on_client_connected.add(initialize_client)


@state.change("opacity", "wireframe")
def update_appearance(opacity, wireframe, **_):
    for method_actors in actors.values():
        for actor in method_actors.values():
            actor.prop.opacity = opacity
            actor.prop.show_edges = wireframe
    if hasattr(ctrl, "view_update"):
        ctrl.view_update()


def apply_region_appearance(region_key: str, color: str, visible: bool):
    """Apply one region's appearance to both comparison renderers."""
    for method_actors in actors.values():
        if region_key in method_actors:
            method_actors[region_key].prop.color = color
            method_actors[region_key].SetVisibility(bool(visible))
    if hasattr(ctrl, "view_update"):
        ctrl.view_update()


def _register_region_callbacks():
    for key in REGION_KEYS:
        suffix = key.replace("-", "_")
        color_name = f"color_{suffix}"
        visible_name = f"visible_{suffix}"

        def register_region_callback(
            region_key=key,
            region_color_name=color_name,
            region_visible_name=visible_name,
        ):
            @state.change(region_color_name, region_visible_name)
            def update_region(**kwargs):
                color = kwargs.get(region_color_name, state[region_color_name])
                visible = kwargs.get(region_visible_name, state[region_visible_name])
                apply_region_appearance(region_key, color, visible)

            return update_region

        register_region_callback()


_register_region_callbacks()

with SinglePageLayout(server) as layout:
    layout.title.set_text("Smooth phase-region rendering comparison")
    with layout.toolbar:
        vuetify3.VSelect(
            v_model=("preset", state.preset),
            items=("preset_names", list(PRESETS)),
            density="compact",
            hide_details=True,
            style="max-width: 240px",
        )
        vuetify3.VSelect(
            v_model=("surface_resolution", state.surface_resolution),
            items=("surface_resolutions", [64, 96, 128]),
            label="Surface Nets grid",
            density="compact",
            hide_details=True,
            style="max-width: 180px",
        )
        vuetify3.VBtn(
            "Rebuild comparison",
            click=ctrl.rebuild,
            loading=("busy", False),
            color="primary",
        )
    with layout.content:
        with vuetify3.VContainer(fluid=True, classes="pa-2"):
            with vuetify3.VRow(no_gutters=True):
                with vuetify3.VCol(cols=12, md=9):
                    view = vtk_widgets.VtkLocalView(
                        plotter.ren_win,
                        ref="view",
                        style="height: calc(100vh - 78px); width: 100%;",
                    )
                    ctrl.view_update = view.update
                with vuetify3.VCol(cols=12, md=3, classes="pa-3"):
                    html.H3("Shared appearance")
                    vuetify3.VSlider(
                        v_model=("opacity", state.opacity),
                        min=0.05,
                        max=1,
                        step=0.05,
                        label="Opacity",
                    )
                    vuetify3.VSwitch(
                        v_model=("wireframe", state.wireframe),
                        label="Show mesh edges",
                        density="compact",
                    )
                    for key in REGION_KEYS:
                        suffix = key.replace("-", "_")
                        with vuetify3.VCard(classes="pa-2 mb-2", variant="outlined"):
                            vuetify3.VSwitch(
                                v_model=(f"visible_{suffix}", True),
                                label=key.replace("-", " ").title(),
                                density="compact",
                                hide_details=True,
                            )
                            vuetify3.VTextField(
                                v_model=(f"color_{suffix}", COLORS[key]),
                                type="color",
                                density="compact",
                                hide_details=True,
                            )
                    html.H3("Metrics")
                    html.Pre(
                        "{{ metrics }}",
                        style="font-size: 11px; max-height: 38vh; overflow: auto;",
                    )


if __name__ == "__main__":
    rebuild()
    server.start()
