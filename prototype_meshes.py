"""Experimental smooth mesh generators used by the renderer comparison app."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pyvista as pv
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from geometry import Ellipsoid, Simplex, ellipsoid_bounds, ellipsoid_value, phase_grid

REGION_KEYS = ("three-phase", "segregative", "associative-left", "associative-right")


@dataclass(frozen=True)
class MethodResult:
    meshes: dict[str, pv.PolyData]
    generation_seconds: float
    grid_resolution: int | None


def plane_ellipsoid_ellipse(
    ellipsoid: Ellipsoid,
    origin: np.ndarray,
    normal: np.ndarray,
    samples: int = 192,
) -> np.ndarray:
    """Sample the exact ellipse where a plane intersects an ellipsoid."""
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    scaled_rotation = ellipsoid.basis @ np.diag(ellipsoid.axes)
    sphere_normal = scaled_rotation.T @ normal
    sphere_offset = float(np.dot(normal, origin - ellipsoid.center))
    normal_squared = float(np.dot(sphere_normal, sphere_normal))
    circle_center = sphere_offset * sphere_normal / normal_squared
    radius_squared = 1.0 - sphere_offset**2 / normal_squared
    if radius_squared <= 1e-12:
        return np.empty((0, 3))

    reference = np.array([0.0, 0.0, 1.0])
    sphere_normal_unit = sphere_normal / np.sqrt(normal_squared)
    if abs(np.dot(reference, sphere_normal_unit)) > 0.95:
        reference = np.array([0.0, 1.0, 0.0])
    first = np.cross(sphere_normal_unit, reference)
    first /= np.linalg.norm(first)
    second = np.cross(sphere_normal_unit, first)
    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    circle = (
        circle_center
        + np.sqrt(radius_squared)
        * (
            np.cos(angles)[:, None] * first
            + np.sin(angles)[:, None] * second
        )
    )
    return ellipsoid.center + circle @ scaled_rotation.T


def _plane_basis(triangle: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = triangle[0]
    first = triangle[1] - origin
    first_length = np.linalg.norm(first)
    if first_length <= 1e-10:
        raise ValueError("The simplex plane is degenerate.")
    first /= first_length
    normal = np.cross(triangle[1] - origin, triangle[2] - origin)
    normal_length = np.linalg.norm(normal)
    if normal_length <= 1e-10:
        raise ValueError("The simplex plane is degenerate.")
    normal /= normal_length
    second = np.cross(normal, first)
    return origin, first, second


def _barycentric_affine(
    triangle: np.ndarray,
    origin: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.column_stack(
        (
            (triangle - origin) @ first,
            (triangle - origin) @ second,
        )
    )
    matrix = np.vstack((coordinates.T, np.ones(3)))
    inverse = np.linalg.inv(matrix)
    constants = inverse[:, 2]
    gradients = inverse[:, :2]
    return constants, gradients


def clip_polygon_halfplane(
    polygon: np.ndarray,
    constant: float,
    gradient: np.ndarray,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Clip a convex 2D polygon to constant + gradient dot point >= 0."""
    if not len(polygon):
        return polygon
    output: list[np.ndarray] = []
    previous = polygon[-1]
    previous_value = constant + float(np.dot(gradient, previous))
    for current in polygon:
        current_value = constant + float(np.dot(gradient, current))
        previous_inside = previous_value >= -tolerance
        current_inside = current_value >= -tolerance
        if previous_inside != current_inside:
            fraction = previous_value / (previous_value - current_value)
            output.append(previous + fraction * (current - previous))
        if current_inside:
            output.append(current)
        previous = current
        previous_value = current_value
    return np.asarray(output)


def _region_constraints(
    label: int,
    constants: np.ndarray,
    gradients: np.ndarray,
) -> list[tuple[float, np.ndarray]]:
    if label == 0:
        return [(constants[index], gradients[index]) for index in range(3)]
    minimum_index = label - 1
    constraints = [(-constants[minimum_index], -gradients[minimum_index])]
    for index in range(3):
        if index == minimum_index:
            continue
        constraints.append(
            (
                constants[index] - constants[minimum_index],
                gradients[index] - gradients[minimum_index],
            )
        )
    return constraints


def simplex_region_polygon(
    ellipsoid: Ellipsoid,
    simplex: Simplex,
    label: int,
    ellipse_samples: int = 192,
) -> np.ndarray:
    """Return one region's ordered cross-section polygon in world coordinates."""
    triangle = simplex.classification_vertices
    plane_origin, first, second = _plane_basis(triangle)
    normal = np.cross(first, second)
    ellipse = plane_ellipsoid_ellipse(
        ellipsoid,
        plane_origin,
        normal,
        samples=ellipse_samples,
    )
    if not len(ellipse):
        return np.empty((0, 3))
    polygon = np.column_stack(
        (
            (ellipse - plane_origin) @ first,
            (ellipse - plane_origin) @ second,
        )
    )
    constants, gradients = _barycentric_affine(
        triangle,
        plane_origin,
        first,
        second,
    )
    for constant, gradient in _region_constraints(label, constants, gradients):
        polygon = clip_polygon_halfplane(polygon, constant, gradient)
        if len(polygon) < 3:
            return np.empty((0, 3))
    return plane_origin + polygon[:, :1] * first + polygon[:, 1:] * second


def resample_closed_polygon(points: np.ndarray, count: int = 64) -> np.ndarray:
    """Resample an ordered closed polygon uniformly by perimeter distance."""
    if len(points) < 3:
        return np.empty((0, 3))
    consecutive_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[
        np.concatenate(([True], consecutive_lengths > 1e-12))
    ]
    if len(points) > 1 and np.linalg.norm(points[-1] - points[0]) <= 1e-12:
        points = points[:-1]
    if len(points) < 3:
        return np.empty((0, 3))
    closed = np.vstack((points, points[0]))
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(np.sum(segment_lengths))
    if perimeter <= 1e-12:
        return np.empty((0, 3))
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    targets = np.linspace(0.0, perimeter, count, endpoint=False)
    indices = np.searchsorted(cumulative, targets, side="right") - 1
    indices = np.minimum(indices, len(points) - 1)
    fractions = (targets - cumulative[indices]) / segment_lengths[indices]
    return closed[indices] + fractions[:, None] * (
        closed[indices + 1] - closed[indices]
    )


def _align_ring(reference: np.ndarray, ring: np.ndarray) -> np.ndarray:
    candidates = []
    for candidate in (ring, ring[::-1]):
        distances = np.sum(
            (candidate[None, :, :] - reference[0][None, None, :]) ** 2,
            axis=2,
        ).ravel()
        shift = int(np.argmin(distances))
        aligned = np.roll(candidate, -shift, axis=0)
        candidates.append(aligned)
    return min(candidates, key=lambda candidate: np.sum((candidate - reference) ** 2))


def loft_polygon_run(
    rings: list[np.ndarray],
    *,
    start_tip: np.ndarray | None = None,
    end_tip: np.ndarray | None = None,
) -> pv.PolyData:
    """Create a closed indexed triangular surface through corresponding rings."""
    if len(rings) < 2:
        return pv.PolyData()
    aligned = [rings[0]]
    for ring in rings[1:]:
        aligned.append(_align_ring(aligned[-1], ring))
    ring_size = len(aligned[0])
    points = np.vstack(aligned)
    faces: list[list[int]] = []
    for layer in range(len(aligned) - 1):
        start = layer * ring_size
        following = (layer + 1) * ring_size
        for index in range(ring_size):
            next_index = (index + 1) % ring_size
            faces.append([3, start + index, following + index, following + next_index])
            faces.append([3, start + index, following + next_index, start + next_index])

    for layer, reverse, tip in (
        (0, True, start_tip),
        (len(aligned) - 1, False, end_tip),
    ):
        center_index = len(points)
        cap_point = np.mean(aligned[layer], axis=0) if tip is None else tip
        points = np.vstack((points, cap_point))
        start = layer * ring_size
        for index in range(ring_size):
            next_index = (index + 1) % ring_size
            edge = (next_index, index) if reverse else (index, next_index)
            faces.append([3, center_index, start + edge[0], start + edge[1]])
    return pv.PolyData(points, np.asarray(faces).ravel()).clean()


def _terminal_support_point(
    ellipsoid: Ellipsoid,
    endpoint: Simplex,
    adjacent: Simplex,
) -> np.ndarray:
    triangle = endpoint.classification_vertices
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal /= np.linalg.norm(normal)
    interior_side = np.dot(normal, adjacent.vertices[0] - triangle[0])
    outward = -np.sign(interior_side or 1.0) * normal
    shape_matrix = (
        ellipsoid.basis
        @ np.diag(ellipsoid.axes**2)
        @ ellipsoid.basis.T
    )
    return ellipsoid.center + shape_matrix @ outward / np.sqrt(
        outward @ shape_matrix @ outward
    )


def direct_swept_meshes(
    ellipsoid: Ellipsoid,
    family: list[Simplex],
    *,
    perimeter_samples: int = 64,
    ellipse_samples: int = 192,
) -> MethodResult:
    """Build four direct meshes by lofting analytical region cross-sections."""
    started = perf_counter()
    meshes: dict[str, pv.PolyData] = {}
    for label, key in enumerate(REGION_KEYS):
        runs: list[tuple[int, int, list[np.ndarray]]] = []
        current: list[np.ndarray] = []
        run_start = 0
        for family_index, simplex in enumerate(family):
            polygon = simplex_region_polygon(
                ellipsoid,
                simplex,
                label,
                ellipse_samples=ellipse_samples,
            )
            ring = resample_closed_polygon(polygon, perimeter_samples)
            if len(ring):
                if not current:
                    run_start = family_index
                current.append(ring)
            elif current:
                runs.append((run_start, family_index - 1, current))
                current = []
        if current:
            runs.append((run_start, len(family) - 1, current))
        run_meshes = []
        for start, stop, run in runs:
            if len(run) < 2:
                continue
            start_tip = (
                _terminal_support_point(ellipsoid, family[0], family[1])
                if start == 0
                else None
            )
            end_tip = (
                _terminal_support_point(ellipsoid, family[-1], family[-2])
                if stop == len(family) - 1
                else None
            )
            run_meshes.append(
                loft_polygon_run(run, start_tip=start_tip, end_tip=end_tip)
            )
        meshes[key] = (
            run_meshes[0].merge(run_meshes[1:], merge_points=False)
            if run_meshes
            else pv.PolyData()
        )
    return MethodResult(meshes, perf_counter() - started, None)


def _vtk_label_image(
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    labels: np.ndarray,
) -> vtk.vtkImageData:
    image = vtk.vtkImageData()
    image.SetDimensions(labels.shape)
    image.SetOrigin(*(axis[0] for axis in axes))
    image.SetSpacing(*(axis[1] - axis[0] for axis in axes))
    scalars = numpy_to_vtk(
        labels.astype(np.int16).ravel(order="F"),
        deep=True,
        array_type=vtk.VTK_SHORT,
    )
    scalars.SetName("RegionLabels")
    image.GetPointData().SetScalars(scalars)
    return image


def surface_nets_meshes(
    ellipsoid: Ellipsoid,
    family: list[Simplex],
    *,
    resolution: int = 64,
    smoothing_iterations: int = 20,
) -> MethodResult:
    """Contour all four labels jointly with vtkSurfaceNets3D."""
    started = perf_counter()
    axes, labels = phase_grid(ellipsoid, family, resolution=resolution)
    image = _vtk_label_image(axes, labels + 1)
    surface_nets = vtk.vtkSurfaceNets3D()
    surface_nets.SetInputData(image)
    surface_nets.SetBackgroundLabel(0)
    surface_nets.SetNumberOfLabels(4)
    for index in range(4):
        surface_nets.SetLabel(index, index + 1)
    surface_nets.SmoothingOn()
    surface_nets.SetNumberOfIterations(smoothing_iterations)
    surface_nets.SetConstraintScale(1.0)
    surface_nets.Update()

    output = pv.wrap(surface_nets.GetOutput())
    boundary_labels = vtk_to_numpy(
        surface_nets.GetOutput().GetCellData().GetArray("BoundaryLabels")
    )
    meshes = {}
    for label, key in enumerate(REGION_KEYS, start=1):
        cell_ids = np.flatnonzero(np.any(boundary_labels == label, axis=1))
        mesh = (
            output.extract_cells(cell_ids)
            .extract_surface(algorithm="dataset_surface")
            .triangulate()
            .clean()
        )
        if mesh.n_points:
            values = ellipsoid_value(mesh.points, ellipsoid)
            exterior = values > 1.0
            if np.any(exterior):
                local = (mesh.points[exterior] - ellipsoid.center) @ ellipsoid.basis
                local /= np.sqrt(values[exterior])[:, None]
                mesh.points[exterior] = local @ ellipsoid.basis.T + ellipsoid.center
        meshes[key] = mesh
    return MethodResult(meshes, perf_counter() - started, resolution)


def mesh_metrics(mesh: pv.PolyData, ellipsoid: Ellipsoid) -> dict[str, float | int]:
    """Calculate comparable topology and hull metrics for a region mesh."""
    if not mesh.n_points:
        return {
            "points": 0,
            "triangles": 0,
            "boundary_edges": 0,
            "non_manifold_edges": 0,
            "hull_error": 0.0,
            "bytes": 0,
        }
    boundary = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    non_manifold = mesh.extract_feature_edges(
        boundary_edges=False,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=True,
    )
    hull_error = float(np.maximum(ellipsoid_value(mesh.points, ellipsoid) - 1.0, 0).max())
    return {
        "points": mesh.n_points,
        "triangles": mesh.n_cells,
        "boundary_edges": boundary.n_cells,
        "non_manifold_edges": non_manifold.n_cells,
        "hull_error": hull_error,
        "bytes": int(mesh.points.nbytes + mesh.faces.nbytes),
    }


def partition_agreement(
    result: MethodResult,
    ellipsoid: Ellipsoid,
    family: list[Simplex],
    *,
    sample_count: int = 1200,
    seed: int = 7,
) -> dict[str, float | int]:
    """Compare mesh containment with independent classifier samples."""
    from geometry import classify_points

    lower, upper = ellipsoid_bounds(ellipsoid)
    generator = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    while sum(len(batch) for batch in accepted) < sample_count:
        candidates = generator.uniform(lower, upper, size=(sample_count, 3))
        accepted.append(candidates[ellipsoid_value(candidates, ellipsoid) <= 1.0])
    points = np.vstack(accepted)[:sample_count]
    expected = classify_points(points, family)
    containment = np.zeros((sample_count, 4), dtype=bool)
    point_cloud = pv.PolyData(points)
    for label, key in enumerate(REGION_KEYS):
        mesh = result.meshes[key]
        if not mesh.n_cells:
            continue
        selected = point_cloud.select_interior_points(
            mesh,
            method="cell_locator",
            check_surface=False,
        )
        containment[:, label] = selected["selected_points"].astype(bool)
    occupied_counts = containment.sum(axis=1)
    predictions = containment.argmax(axis=1)
    agreement = (predictions == expected) & (occupied_counts == 1)
    return {
        "classifier_agreement": float(np.mean(agreement)),
        "gap_fraction": float(np.mean(occupied_counts == 0)),
        "overlap_fraction": float(np.mean(occupied_counts > 1)),
        "samples": sample_count,
    }


def default_prototype_geometry() -> tuple[Ellipsoid, list[Simplex]]:
    """Build the production viewer's default 500-simplex geometry."""
    from geometry import generate_simplex_family, make_ellipsoid

    ellipsoid = make_ellipsoid([3, 2, 2], 5, 90, 45)
    base = generate_simplex_family(
        ellipsoid,
        45,
        0,
        0,
        0,
        initial_theta=90,
        target_count=500,
    )
    z1 = base[125].vertex_z
    z2 = base[375].vertex_z
    family = generate_simplex_family(
        ellipsoid,
        45,
        z1,
        z2,
        45,
        initial_theta=90,
        target_count=500,
    )
    return ellipsoid, family
