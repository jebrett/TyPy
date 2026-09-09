#!/usr/bin/env python3
"""Production VTK/Trame viewer for three-phase tie-simplexes."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from time import time_ns

import numpy as np
import pyvista as pv
import vtk
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import html, vtk as vtk_widgets, vuetify3

from geometry import Ellipsoid, Simplex, generate_simplex_family, make_ellipsoid, nearest_simplex
from prototype_meshes import REGION_KEYS, direct_swept_meshes, partition_agreement

FAMILY_SIZE = 500
REGION_NAMES = {
    "three-phase": "Three-phase coexistence",
    "segregative": "Segregative two-phase",
    "associative-left": "Associative two-phase (left)",
    "associative-right": "Associative two-phase (right)",
}
REGION_COLORS = {
    "three-phase": "#087830",
    "segregative": "#d62728",
    "associative-left": "#1f77b4",
    "associative-right": "#1f77b4",
}
REGION_OPACITIES = {
    "three-phase": 0.42,
    "segregative": 0.18,
    "associative-left": 0.18,
    "associative-right": 0.18,
}
EXPORT_TEMP_DIRECTORY = TemporaryDirectory(prefix="typy-viewer-exports-")
EXPORT_DIRECTORY = Path(EXPORT_TEMP_DIRECTORY.name)

server = get_server(client_type="vue3")
server.serve["exports"] = str(EXPORT_DIRECTORY)
state, ctrl = server.state, server.controller
plotter = pv.Plotter(off_screen=True, border=False)
plotter.renderer.SetBackground(0.96, 0.97, 0.99)
plotter.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0.0)

actors: dict[str, object] = {}
datasets: dict[str, pv.PolyData] = {}
current_ellipsoid: Ellipsoid | None = None
current_family: list[Simplex] = []
rebuilding = False
current_export_path: Path | None = None

state.ellipsoid_length = 3.0
state.ellipsoid_width = 2.0
state.ellipsoid_height = 2.0
state.ellipsoid_theta = 90.0
state.ellipsoid_phi = 45.0
state.rho = 5.0
state.z_offset = 0.0
state.simplex_theta = 90.0
state.simplex_phi = 45.0
state.maximum_alpha = 45.0
state.z_min = -1.0
state.z_max = 1.0
state.z_step = 0.01
state.z_window = [-0.5, 0.5]
state.slice_z = 0.0
state.busy = False
state.error_message = ""
state.status = "Preparing the initial analytical mesh..."
state.export_format = "svg"
state.export_url = ""
state.export_filename = ""
state.export_error = ""
for key in ("three-phase", "segregative"):
    suffix = key.replace("-", "_")
    state[f"color_{suffix}"] = REGION_COLORS[key]
    state[f"visible_{suffix}"] = True
state.color_associative = REGION_COLORS["associative-left"]
state.visible_associative = True


def _ellipsoid_from_state() -> Ellipsoid:
    return make_ellipsoid(
        [state.ellipsoid_length, state.ellipsoid_width, state.ellipsoid_height],
        state.rho,
        state.ellipsoid_theta,
        state.ellipsoid_phi,
        state.z_offset,
    )


def _central_family(ellipsoid: Ellipsoid) -> list[Simplex]:
    return generate_simplex_family(
        ellipsoid,
        state.simplex_phi,
        0,
        0,
        0,
        initial_theta=state.simplex_theta,
        target_count=FAMILY_SIZE,
    )


def _reachable_range(family: list[Simplex]) -> dict[str, object]:
    minimum = family[0].vertex_z
    maximum = family[-1].vertex_z
    span = maximum - minimum
    if span <= 1e-8:
        raise ValueError("Reachable simplex z range is degenerate.")
    return {
        "z_min": minimum,
        "z_max": maximum,
        "z_step": span / (FAMILY_SIZE - 1),
        "z_window": [minimum + 0.25 * span, minimum + 0.75 * span],
        "slice_z": nearest_simplex(family, theta=state.simplex_theta).vertex_z,
    }


def _ellipsoid_mesh(ellipsoid: Ellipsoid) -> pv.PolyData:
    longitude = np.linspace(0, 2 * np.pi, 64)
    latitude = np.linspace(0, np.pi, 33)
    u, v = np.meshgrid(longitude, latitude, indexing="ij")
    local = np.stack(
        (
            ellipsoid.axes[0] * np.cos(u) * np.sin(v),
            ellipsoid.axes[1] * np.sin(u) * np.sin(v),
            ellipsoid.axes[2] * np.cos(v),
        ),
        axis=-1,
    )
    points = local @ ellipsoid.basis.T + ellipsoid.center
    grid = pv.StructuredGrid(points[:, :, 0], points[:, :, 1], points[:, :, 2])
    return grid.extract_surface(algorithm="dataset_surface").triangulate()


def _family_lines(family: list[Simplex]) -> pv.PolyData:
    point_groups = [simplex.vertices[[0, 1, 2, 0]] for simplex in family]
    points = np.vstack(point_groups)
    cells = np.asarray(
        [[4, *range(index * 4, index * 4 + 4)] for index in range(len(family))]
    )
    mesh = pv.PolyData(points)
    mesh.lines = cells.ravel()
    return mesh


def _selected_meshes(simplex: Simplex) -> tuple[pv.PolyData, pv.PolyData]:
    vertices = simplex.vertices
    fill = pv.PolyData(vertices, np.array([3, 0, 1, 2]))
    outline = pv.PolyData(vertices[[0, 1, 2, 0]])
    outline.lines = np.array([4, 0, 1, 2, 3])
    return fill, outline


def _build_geometry(reset_range: bool):
    ellipsoid = _ellipsoid_from_state()
    base_family = _central_family(ellipsoid)
    if len(base_family) != FAMILY_SIZE:
        raise ValueError("No complete 500-simplex branch is reachable.")
    range_values: dict[str, object] = {}
    if reset_range:
        range_values = _reachable_range(base_family)
    z1, z2 = range_values.get("z_window", state.z_window)
    selected_z = range_values.get("slice_z", state.slice_z)
    family = generate_simplex_family(
        ellipsoid,
        state.simplex_phi,
        z1,
        z2,
        state.maximum_alpha,
        initial_theta=state.simplex_theta,
        target_count=FAMILY_SIZE,
    )
    if len(family) != FAMILY_SIZE:
        raise ValueError(
            "Direct lofting cannot produce four valid 500-section regions "
            "for this control configuration."
        )
    regions = direct_swept_meshes(ellipsoid, family)
    if any(not mesh.n_cells for mesh in regions.meshes.values()):
        raise ValueError("At least one phase region has no valid analytical mesh.")
    invalid_topology = [
        key
        for key, mesh in regions.meshes.items()
        if mesh.n_open_edges or not mesh.is_manifold
    ]
    if invalid_topology:
        raise ValueError(
            "Analytical region meshes are not closed and manifold: "
            + ", ".join(invalid_topology)
            + "."
        )
    quality = partition_agreement(
        regions,
        ellipsoid,
        family,
        sample_count=400,
    )
    if (
        quality["classifier_agreement"] < 0.94
        or quality["gap_fraction"] > 0.03
        or quality["overlap_fraction"] > 0.03
    ):
        raise ValueError(
            "Analytical partition coverage is incomplete "
            f"(agreement {quality['classifier_agreement']:.1%}, "
            f"gaps {quality['gap_fraction']:.1%}, "
            f"overlaps {quality['overlap_fraction']:.1%})."
        )
    selected = nearest_simplex(family, vertex_z=selected_z)
    fill, outline = _selected_meshes(selected)
    return ellipsoid, family, regions, selected, fill, outline, range_values


def _add_scene(ellipsoid, family, regions, selected, fill, outline, range_values):
    global current_ellipsoid, current_family
    camera_position = plotter.camera_position if actors else None
    plotter.clear()
    actors.clear()
    datasets.clear()

    for key, mesh in regions.meshes.items():
        appearance_key = (
            "associative" if key.startswith("associative-") else key.replace("-", "_")
        )
        datasets[key] = mesh
        actors[key] = plotter.add_mesh(
            mesh,
            name=key,
            color=state[f"color_{appearance_key}"],
            opacity=REGION_OPACITIES[key],
            smooth_shading=True,
        )
        actors[key].SetVisibility(bool(state[f"visible_{appearance_key}"]))

    hull = _ellipsoid_mesh(ellipsoid)
    family_mesh = _family_lines(family)
    datasets["ellipsoid"] = hull
    datasets["family"] = family_mesh
    datasets["selected-fill"] = fill
    datasets["selected-outline"] = outline
    actors["ellipsoid"] = plotter.add_mesh(
        hull,
        color="#67748e",
        opacity=0.10,
        smooth_shading=True,
        name="Ellipsoidal phase boundary",
    )
    actors["family"] = plotter.add_mesh(
        family_mesh,
        color="#414141",
        opacity=0.18,
        line_width=1,
        name="Simplex family",
    )
    actors["selected-fill"] = plotter.add_mesh(
        fill,
        color="#ff7f0e",
        opacity=0.65,
        name="Selected 2-simplex",
    )
    actors["selected-fill"].SetVisibility(selected.alpha > 1e-8)
    actors["selected-outline"] = plotter.add_mesh(
        outline,
        color="#ff7f0e",
        line_width=6,
        render_lines_as_tubes=True,
        name="Selected simplex",
    )
    plotter.add_mesh(
        pv.Sphere(radius=0.05, center=(0, 0, 0)),
        color="black",
        name="Origin",
    )
    actors["axes"] = plotter.show_bounds(
        grid="front",
        location="outer",
        all_edges=True,
        ticks="outside",
        xtitle="Component 1",
        ytitle="Component 2",
        ztitle="Component 3",
        color="#364152",
        font_size=11,
    )

    if camera_position is None:
        plotter.reset_camera()
    else:
        plotter.camera_position = camera_position
    current_ellipsoid = ellipsoid
    current_family = family
    for key, value in range_values.items():
        state[key] = value
    state.status = (
        f"Selected {'1-simplex' if selected.alpha <= 1e-8 else '2-simplex'}: "
        f"vertex z = {selected.vertex_z:.3f}, theta = {selected.theta:.1f}°, "
        f"alpha = {selected.alpha:.2f}°. Rendered {len(family)} analytical sections."
    )


def rebuild_scene(*, reset_range: bool = False):
    """Build off-screen first; commit only a complete valid analytical scene."""
    global rebuilding
    if rebuilding:
        return False
    rebuilding = True
    state.busy = True
    state.flush()
    try:
        geometry = _build_geometry(reset_range)
        _add_scene(*geometry)
        state.error_message = ""
        if hasattr(ctrl, "view_update"):
            ctrl.view_update()
        return True
    except Exception as error:
        state.error_message = (
            f"{error} The last valid analytical render remains displayed."
        )
        return False
    finally:
        state.busy = False
        rebuilding = False


def update_selected_slice(value: float | None = None):
    if not current_family or "selected-fill" not in datasets:
        return
    selected = nearest_simplex(
        current_family,
        vertex_z=state.slice_z if value is None else value,
    )
    fill, outline = _selected_meshes(selected)
    datasets["selected-fill"].copy_from(fill)
    datasets["selected-outline"].copy_from(outline)
    actors["selected-fill"].SetVisibility(selected.alpha > 1e-8)
    state.status = (
        f"Selected {'1-simplex' if selected.alpha <= 1e-8 else '2-simplex'}: "
        f"vertex z = {selected.vertex_z:.3f}, theta = {selected.theta:.1f}°, "
        f"alpha = {selected.alpha:.2f}°. Rendered {len(current_family)} analytical sections."
    )
    if hasattr(ctrl, "view_update"):
        ctrl.view_update()


def rebuild_for_geometry(*_, **__):
    rebuild_scene(reset_range=True)


def rebuild_for_simplex_shape(*_, **__):
    rebuild_scene(reset_range=False)


def update_slice(slice_z=None, **_):
    if not rebuilding:
        update_selected_slice(slice_z)


def apply_region_appearance(region_key: str, color: str, visible: bool):
    if region_key not in actors:
        return
    actors[region_key].prop.color = color
    actors[region_key].SetVisibility(bool(visible))
    if hasattr(ctrl, "view_update"):
        ctrl.view_update()


def apply_associative_appearance(color: str, visible: bool):
    for key in ("associative-left", "associative-right"):
        apply_region_appearance(key, color, visible)


def export_graphic(export_format: str | None = None, *_):
    """Save the current server-synchronized camera view as SVG or PDF."""
    global current_export_path
    selected_format = (export_format or state.export_format).lower()
    if selected_format not in {"svg", "pdf"}:
        state.export_error = f"Unsupported export format: {selected_format}"
        return False
    try:
        filename = f"simplex-view-{time_ns()}.{selected_format}"
        destination = EXPORT_DIRECTORY / filename
        plotter.render()
        plotter.save_graphic(destination)
        if current_export_path is not None and current_export_path != destination:
            current_export_path.unlink(missing_ok=True)
        current_export_path = destination
        state.export_filename = filename
        state.export_url = f"/exports/{filename}"
        state.export_error = ""
        return True
    except Exception as error:
        state.export_error = f"Export failed: {error}"
        return False


def _register_region_callbacks():
    for key in ("three-phase", "segregative"):
        suffix = key.replace("-", "_")
        color_name = f"color_{suffix}"
        visible_name = f"visible_{suffix}"

        def register(
            region_key=key,
            region_color_name=color_name,
            region_visible_name=visible_name,
        ):
            @state.change(region_color_name, region_visible_name)
            def update_region(**kwargs):
                if region_key not in actors:
                    return
                color = kwargs.get(region_color_name, state[region_color_name])
                visible = kwargs.get(region_visible_name, state[region_visible_name])
                apply_region_appearance(region_key, color, visible)

            return update_region

        register()

    @state.change("color_associative", "visible_associative")
    def update_associative_regions(**kwargs):
        color = kwargs.get("color_associative", state.color_associative)
        visible = kwargs.get("visible_associative", state.visible_associative)
        apply_associative_appearance(color, visible)


_register_region_callbacks()
ctrl.rebuild_scene = lambda: rebuild_scene(reset_range=False)
ctrl.rebuild_geometry = rebuild_for_geometry
ctrl.rebuild_simplex_shape = rebuild_for_simplex_shape
ctrl.update_slice = update_slice
ctrl.export_graphic = export_graphic


def _slider(label, model, minimum, maximum, step, on_end):
    return vuetify3.VSlider(
        v_model=(model, state[model]),
        min=minimum,
        max=maximum,
        step=step,
        label=label,
        density="compact",
        hide_details=True,
        thumb_label=True,
        end=on_end,
    )


def _control_group(title):
    details = html.Details(open=True, classes="mb-3")
    details.add_child(html.Summary(title, classes="text-subtitle-1 font-weight-bold pa-2"))
    return details


with SinglePageLayout(server) as layout:
    layout.title.set_text("Three-phase equilibrium viewer")
    with layout.toolbar:
        vuetify3.VBtn(
            "Rebuild",
            click=ctrl.rebuild_scene,
            loading=("busy", False),
            color="primary",
            size="small",
        )
        vuetify3.VSelect(
            v_model=("export_format", state.export_format),
            items=("export_formats", ["svg", "pdf"]),
            label="Export format",
            density="compact",
            hide_details=True,
            style="max-width: 120px",
            classes="ml-3",
        )
        vuetify3.VBtn(
            "Export picture",
            click=(ctrl.export_graphic, "[export_format]"),
            variant="outlined",
            size="small",
            classes="ml-2",
        )
        html.A(
            "Download",
            v_if=("export_url",),
            href=("export_url",),
            download=("export_filename",),
            classes="v-btn v-btn--size-small bg-success ml-2 px-4 py-2",
        )
        vuetify3.VSpacer()
        html.Span("Direct analytical loft renderer", classes="text-caption")
    with layout.content:
        with vuetify3.VContainer(fluid=True, classes="pa-0"):
            with vuetify3.VRow(no_gutters=True):
                with vuetify3.VCol(cols=12, md=3, classes="pa-3 overflow-y-auto"):
                    with _control_group("Ellipsoid controls"):
                        with html.Div(classes="pa-2"):
                            _slider("Length (semi-axis)", "ellipsoid_length", 0, 10, 0.1, ctrl.rebuild_geometry)
                            _slider("Width (semi-axis)", "ellipsoid_width", 0, 10, 0.1, ctrl.rebuild_geometry)
                            _slider("Height (semi-axis)", "ellipsoid_height", 0, 10, 0.1, ctrl.rebuild_geometry)
                            _slider("Direction theta (zenith)", "ellipsoid_theta", 0, 180, 1, ctrl.rebuild_geometry)
                            _slider("Direction phi (azimuth)", "ellipsoid_phi", 0, 360, 1, ctrl.rebuild_geometry)
                            _slider("Center rho", "rho", 0, 10, 0.1, ctrl.rebuild_geometry)
                            _slider("Z offset", "z_offset", -10, 10, 0.1, ctrl.rebuild_geometry)

                    with _control_group("Region appearance"):
                        with html.Div(classes="pa-2"):
                            for key in ("three-phase", "segregative"):
                                suffix = key.replace("-", "_")
                                vuetify3.VSwitch(
                                    v_model=(f"visible_{suffix}", True),
                                    label=REGION_NAMES[key],
                                    density="compact",
                                    hide_details=True,
                                )
                                vuetify3.VTextField(
                                    v_model=(f"color_{suffix}", REGION_COLORS[key]),
                                    type="color",
                                    label=f"{REGION_NAMES[key]} color",
                                    density="compact",
                                    hide_details=True,
                                    classes="mb-2",
                                )
                            vuetify3.VSwitch(
                                v_model=("visible_associative", True),
                                label="Associative two-phase (left and right)",
                                density="compact",
                                hide_details=True,
                            )
                            vuetify3.VTextField(
                                v_model=("color_associative", REGION_COLORS["associative-left"]),
                                type="color",
                                label="Associative two-phase color",
                                density="compact",
                                hide_details=True,
                                classes="mb-2",
                            )

                    with _control_group("Simplex controls"):
                        with html.Div(classes="pa-2"):
                            _slider("Initial theta (zenith)", "simplex_theta", 0, 180, 1, ctrl.rebuild_geometry)
                            _slider("Phi (azimuth)", "simplex_phi", 0, 360, 1, ctrl.rebuild_geometry)
                            _slider("Maximum alpha", "maximum_alpha", 0, 89.5, 0.5, ctrl.rebuild_simplex_shape)
                            vuetify3.VRangeSlider(
                                v_model=("z_window", state.z_window),
                                min=("z_min", state.z_min),
                                max=("z_max", state.z_max),
                                step=("z_step", state.z_step),
                                label="Alpha window (z1 to z2)",
                                density="compact",
                                hide_details=True,
                                thumb_label=True,
                                end=ctrl.rebuild_simplex_shape,
                            )
                            vuetify3.VSlider(
                                v_model=("slice_z", state.slice_z),
                                min=("z_min", state.z_min),
                                max=("z_max", state.z_max),
                                step=("z_step", state.z_step),
                                label="Orange slice: main-vertex z",
                                density="compact",
                                hide_details=True,
                                thumb_label=True,
                                end=ctrl.update_slice,
                            )

                    vuetify3.VAlert(
                        "{{ error_message }}",
                        v_if=("error_message",),
                        type="error",
                        variant="tonal",
                        classes="mb-2",
                    )
                    vuetify3.VAlert(
                        "{{ export_error }}",
                        v_if=("export_error",),
                        type="error",
                        variant="tonal",
                        classes="mb-2",
                    )
                    vuetify3.VAlert(
                        "{{ status }}",
                        type="info",
                        variant="tonal",
                    )

                with vuetify3.VCol(cols=12, md=9):
                    view = vtk_widgets.VtkRemoteView(
                        plotter.ren_win,
                        ref="view",
                        style="height: calc(100vh - 64px); width: 100%;",
                        interactive_quality=70,
                        still_quality=100,
                        interactive_ratio=0.75,
                    )
                    ctrl.view_update = view.update


def initialize_client(**_):
    if not actors:
        rebuild_scene(reset_range=True)
    elif hasattr(ctrl, "view_update"):
        ctrl.view_update()


ctrl.on_client_connected.add(initialize_client)

if __name__ == "__main__":
    rebuild_scene(reset_range=True)
    server.start()
