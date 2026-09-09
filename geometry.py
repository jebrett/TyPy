"""Geometry primitives for the three-phase simplex visualizer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

AXIS_EPSILON = 0.05
INTERSECTION_EPSILON = 1e-8

Vector = NDArray[np.float64]


@dataclass(frozen=True)
class Ellipsoid:
    center: Vector
    axes: Vector
    basis: Vector


@dataclass(frozen=True)
class Simplex:
    theta: float
    vertex_z: float
    alpha: float
    vertices: Vector
    classification_vertices: Vector


def spherical_direction(theta: float, phi: float) -> Vector:
    """Return a unit vector from zenith theta and azimuth phi, in degrees."""
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    return np.array(
        [
            np.sin(theta_rad) * np.cos(phi_rad),
            np.sin(theta_rad) * np.sin(phi_rad),
            np.cos(theta_rad),
        ],
        dtype=float,
    )


def make_ellipsoid(
    axes: tuple[float, float, float] | list[float] | Vector,
    rho: float,
    theta: float,
    phi: float,
    z_offset: float = 0.0,
) -> Ellipsoid:
    """Construct an ellipsoid along a directed ray from a z-axis intercept."""
    primary = spherical_direction(theta, phi)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(primary, reference)) > 0.95:
        reference = np.array([0.0, 1.0, 0.0])

    secondary = np.cross(reference, primary)
    secondary /= np.linalg.norm(secondary)
    tertiary = np.cross(primary, secondary)
    tertiary /= np.linalg.norm(tertiary)

    return Ellipsoid(
        center=float(rho) * primary + np.array([0.0, 0.0, float(z_offset)]),
        axes=np.maximum(np.asarray(axes, dtype=float), AXIS_EPSILON),
        basis=np.column_stack((primary, secondary, tertiary)),
    )


def ellipsoid_value(points: Vector, ellipsoid: Ellipsoid) -> Vector:
    """Evaluate the ellipsoid's normalized implicit quadratic."""
    points_2d = np.atleast_2d(points)
    local = (points_2d - ellipsoid.center) @ ellipsoid.basis
    return np.sum((local / ellipsoid.axes) ** 2, axis=1)


def ellipsoid_bounds(ellipsoid: Ellipsoid) -> tuple[Vector, Vector]:
    """Return the exact axis-aligned bounds of a rotated ellipsoid."""
    extents = np.sqrt(np.sum((ellipsoid.basis * ellipsoid.axes) ** 2, axis=1))
    return ellipsoid.center - extents, ellipsoid.center + extents


def clip_points_to_ellipsoid(points: Vector, ellipsoid: Ellipsoid) -> Vector:
    """Project exterior points radially onto the ellipsoid surface."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    local = (points - ellipsoid.center) @ ellipsoid.basis
    normalized_radius = np.sqrt(np.sum((local / ellipsoid.axes) ** 2, axis=1))
    exterior = normalized_radius > 1.0
    local[exterior] /= normalized_radius[exterior, None]
    return local @ ellipsoid.basis.T + ellipsoid.center


def ray_ellipsoid_intersections(
    origin: Vector,
    direction: Vector,
    ellipsoid: Ellipsoid,
) -> Vector:
    """Return sorted signed distances where a ray's supporting line meets a hull."""
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm <= INTERSECTION_EPSILON:
        return np.empty(0)
    direction = direction / norm

    local_origin = (origin - ellipsoid.center) @ ellipsoid.basis
    local_direction = direction @ ellipsoid.basis
    inverse_axes_squared = 1.0 / ellipsoid.axes**2

    a = np.sum(local_direction**2 * inverse_axes_squared)
    b = 2.0 * np.sum(local_origin * local_direction * inverse_axes_squared)
    c = np.sum(local_origin**2 * inverse_axes_squared) - 1.0
    discriminant = b * b - 4.0 * a * c

    if discriminant < -INTERSECTION_EPSILON:
        return np.empty(0)
    if abs(discriminant) <= INTERSECTION_EPSILON:
        return np.array([-b / (2.0 * a)])

    root = np.sqrt(discriminant)
    return np.sort(np.array([(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]))


def alpha_for_z(z: float, z1: float, z2: float, maximum: float) -> float:
    """Return the sine-windowed simplex opening angle in degrees."""
    if z2 - z1 <= INTERSECTION_EPSILON or z <= z1 or z >= z2:
        return 0.0
    phase = (z - z1) / (z2 - z1)
    return float(maximum * np.sin(np.pi * phase))


def _exit_point(vertex: Vector, direction: Vector, ellipsoid: Ellipsoid) -> Vector | None:
    roots = ray_ellipsoid_intersections(vertex, direction, ellipsoid)
    positive = roots[roots > INTERSECTION_EPSILON]
    if not len(positive):
        return None
    return vertex + positive[-1] * direction


def _simplex_vertices(
    vertex: Vector,
    theta: float,
    phi: float,
    alpha: float,
    ellipsoid: Ellipsoid,
) -> Vector | None:
    endpoints = []
    for azimuth in (phi - alpha, phi + alpha):
        endpoint = _exit_point(
            vertex,
            spherical_direction(theta, azimuth),
            ellipsoid,
        )
        if endpoint is None:
            return None
        endpoints.append(endpoint)
    return np.vstack((vertex, endpoints[0], endpoints[1]))


def generate_simplex_family(
    ellipsoid: Ellipsoid,
    phi: float,
    z1: float,
    z2: float,
    maximum_alpha: float,
    *,
    initial_theta: float | None = None,
    samples: int = 121,
    target_count: int | None = None,
    classification_alpha: float = 0.25,
) -> list[Simplex]:
    """Generate a monotonic, origin-based entry family ordered by vertex z."""
    theta_values = np.linspace(0.0, 180.0, samples)
    if initial_theta is not None:
        theta_values = np.unique(np.append(theta_values, initial_theta))

    origin = np.zeros(3)

    def simplex_at_theta(theta: float) -> Simplex | None:
        direction = spherical_direction(float(theta), phi)
        roots = ray_ellipsoid_intersections(origin, direction, ellipsoid)
        if len(roots) < 2 or roots[0] <= INTERSECTION_EPSILON:
            return None

        vertex = roots[0] * direction
        alpha = alpha_for_z(vertex[2], z1, z2, maximum_alpha)
        vertices = _simplex_vertices(vertex, float(theta), phi, alpha, ellipsoid)
        if vertices is None:
            return None

        effective_alpha = max(alpha, classification_alpha)
        classification_vertices = _simplex_vertices(
            vertex,
            float(theta),
            phi,
            effective_alpha,
            ellipsoid,
        )
        if classification_vertices is None:
            classification_vertices = vertices

        return Simplex(
            theta=float(theta),
            vertex_z=float(vertex[2]),
            alpha=alpha,
            vertices=vertices,
            classification_vertices=classification_vertices,
        )

    candidates = [
        simplex
        for theta in theta_values
        if (simplex := simplex_at_theta(float(theta))) is not None
    ]
    if not candidates:
        return []

    candidates.sort(key=lambda simplex: simplex.theta)
    nominal_step = 180.0 / max(samples - 1, 1)
    groups: list[list[Simplex]] = [[]]
    for simplex in candidates:
        if groups[-1] and simplex.theta - groups[-1][-1].theta > 1.5 * nominal_step:
            groups.append([])
        groups[-1].append(simplex)

    anchor_theta = initial_theta if initial_theta is not None else 90.0
    reachable_group = min(
        groups,
        key=lambda group: min(abs(simplex.theta - anchor_theta) for simplex in group),
    )
    def split_monotonic(simplexes: list[Simplex]) -> list[list[Simplex]]:
        segments: list[list[Simplex]] = []
        segment_start = 0
        direction = 0
        for index in range(1, len(simplexes)):
            difference = simplexes[index].vertex_z - simplexes[index - 1].vertex_z
            next_direction = 0 if abs(difference) < 1e-9 else int(np.sign(difference))
            if direction and next_direction and next_direction != direction:
                segments.append(simplexes[segment_start:index])
                segment_start = index - 1
            if next_direction:
                direction = next_direction
        segments.append(simplexes[segment_start:])
        return segments

    monotonic_segments = split_monotonic(reachable_group)

    def segment_score(segment: list[Simplex]) -> tuple[float, float]:
        lower_theta = segment[0].theta
        upper_theta = segment[-1].theta
        distance = max(lower_theta - anchor_theta, anchor_theta - upper_theta, 0.0)
        z_span = abs(segment[-1].vertex_z - segment[0].vertex_z)
        return distance, -z_span

    family = min(monotonic_segments, key=segment_score)
    if target_count is not None and target_count >= 2:
        if len(family) >= 2:
            theta_start, theta_stop = family[0].theta, family[-1].theta
        else:
            theta_start = max(0.0, family[0].theta - nominal_step)
            theta_stop = min(180.0, family[0].theta + nominal_step)
        for multiplier in (4, 8, 16, 32):
            dense_theta = np.linspace(
                theta_start,
                theta_stop,
                target_count * multiplier,
            )
            valid_runs: list[list[Simplex]] = []
            current_run: list[Simplex] = []
            for theta in dense_theta:
                simplex = simplex_at_theta(float(theta))
                if simplex is None:
                    if current_run:
                        valid_runs.append(current_run)
                        current_run = []
                    continue
                current_run.append(simplex)
            if current_run:
                valid_runs.append(current_run)
            if not valid_runs:
                continue

            dense_segments = [
                segment
                for run in valid_runs
                for segment in split_monotonic(run)
            ]
            dense_family = min(dense_segments, key=segment_score)
            dense_family.sort(key=lambda simplex: simplex.vertex_z)
            distinct = [
                simplex
                for index, simplex in enumerate(dense_family)
                if index == 0
                or abs(simplex.vertex_z - dense_family[index - 1].vertex_z) >= 1e-9
            ]
            if len(distinct) < target_count:
                continue

            selected_indices = np.linspace(
                0,
                len(distinct) - 1,
                target_count,
                dtype=int,
            )
            return [distinct[index] for index in selected_indices]
        return []

    family.sort(key=lambda simplex: simplex.vertex_z)
    deduplicated: list[Simplex] = []
    for simplex in family:
        if deduplicated and abs(simplex.vertex_z - deduplicated[-1].vertex_z) < 1e-7:
            continue
        deduplicated.append(simplex)
    return deduplicated


def nearest_simplex(
    family: list[Simplex],
    *,
    vertex_z: float | None = None,
    theta: float | None = None,
) -> Simplex:
    """Select the family member nearest a requested z or zenith."""
    if not family:
        raise ValueError("The simplex family is empty.")
    if vertex_z is not None:
        return min(family, key=lambda simplex: abs(simplex.vertex_z - vertex_z))
    if theta is not None:
        return min(family, key=lambda simplex: abs(simplex.theta - theta))
    raise ValueError("Either vertex_z or theta must be provided.")


def classify_points(
    points: Vector,
    family: list[Simplex],
    *,
    chunk_size: int = 2048,
) -> NDArray[np.int_]:
    """Classify points into green, red, blue-left, and blue-right regions.

    Labels are 0 through 3 respectively. Each point uses the simplex whose
    plane is nearest to the point.
    """
    if not family:
        raise ValueError("The simplex family is empty.")

    points = np.atleast_2d(np.asarray(points, dtype=float))
    origins = np.array([simplex.classification_vertices[0] for simplex in family])
    normals = np.array(
        [
            np.cross(
                simplex.classification_vertices[1] - simplex.classification_vertices[0],
                simplex.classification_vertices[2] - simplex.classification_vertices[0],
            )
            for simplex in family
        ]
    )
    normal_lengths = np.linalg.norm(normals, axis=1)
    valid_normals = normal_lengths > INTERSECTION_EPSILON
    normals[valid_normals] /= normal_lengths[valid_normals, None]
    nearest_indices = np.empty(len(points), dtype=int)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        plane_distances = np.abs(
            np.einsum(
                "pfi,fi->pf",
                points[start:stop, None, :] - origins[None, :, :],
                normals,
            )
        )
        plane_distances[:, ~valid_normals] = np.inf
        nearest_indices[start:stop] = plane_distances.argmin(axis=1)
    labels = np.empty(len(points), dtype=int)

    for family_index in np.unique(nearest_indices):
        point_indices = np.flatnonzero(nearest_indices == family_index)
        triangle = family[family_index].classification_vertices
        origin, second, third = triangle
        edge_matrix = np.column_stack((second - origin, third - origin))
        coefficients = np.linalg.lstsq(
            edge_matrix,
            (points[point_indices] - origin).T,
            rcond=None,
        )[0].T
        barycentric = np.column_stack(
            (1.0 - coefficients[:, 0] - coefficients[:, 1], coefficients)
        )

        inside = np.all(barycentric >= -1e-9, axis=1)
        local_labels = np.argmin(barycentric, axis=1)
        local_labels = np.where(inside, 0, local_labels + 1)
        labels[point_indices] = local_labels

    return labels


def phase_grid(
    ellipsoid: Ellipsoid,
    family: list[Simplex],
    resolution: int = 18,
) -> tuple[tuple[Vector, Vector, Vector], NDArray[np.int_]]:
    """Sample the ellipsoid and return a discrete four-region label grid."""
    lower, upper = ellipsoid_bounds(ellipsoid)
    axes = tuple(
        np.linspace(lower[index], upper[index], resolution) for index in range(3)
    )
    x_grid, y_grid, z_grid = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))
    inside = ellipsoid_value(points, ellipsoid) <= 1.0

    labels = np.full(len(points), -1, dtype=int)
    labels[inside] = classify_points(points[inside], family)
    return axes, labels.reshape((resolution, resolution, resolution))
