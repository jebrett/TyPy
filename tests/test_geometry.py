import unittest

import numpy as np

from geometry import (
    AXIS_EPSILON,
    alpha_for_z,
    classify_points,
    clip_points_to_ellipsoid,
    ellipsoid_value,
    generate_simplex_family,
    make_ellipsoid,
    nearest_simplex,
    ray_ellipsoid_intersections,
    spherical_direction,
)


class GeometryTests(unittest.TestCase):
    def test_spherical_direction_uses_zenith_and_azimuth(self):
        np.testing.assert_allclose(spherical_direction(0, 123), [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(spherical_direction(90, 90), [0, 1, 0], atol=1e-12)

    def test_ellipsoid_axes_are_clamped_and_basis_is_orthonormal(self):
        ellipsoid = make_ellipsoid([0, 2, 3], rho=4, theta=0, phi=0)

        self.assertEqual(ellipsoid.axes[0], AXIS_EPSILON)
        np.testing.assert_allclose(ellipsoid.basis.T @ ellipsoid.basis, np.eye(3))
        np.testing.assert_allclose(ellipsoid.center, [0, 0, 4])

    def test_ellipsoid_center_includes_z_axis_intercept(self):
        ellipsoid = make_ellipsoid(
            [2, 1, 1],
            rho=5,
            theta=90,
            phi=0,
            z_offset=2.5,
        )

        np.testing.assert_allclose(ellipsoid.center, [5, 0, 2.5], atol=1e-12)

    def test_ray_intersections_work_for_rotated_shifted_ellipsoid(self):
        ellipsoid = make_ellipsoid([2, 1, 1], rho=5, theta=90, phi=0)
        roots = ray_ellipsoid_intersections(
            np.zeros(3),
            spherical_direction(90, 0),
            ellipsoid,
        )

        np.testing.assert_allclose(roots, [3, 7], atol=1e-10)

    def test_exterior_points_are_clipped_to_ellipsoid(self):
        ellipsoid = make_ellipsoid([2, 1, 1], rho=5, theta=90, phi=0)
        points = np.array([[8, 0, 0], [5, 0.5, 0]])
        clipped = clip_points_to_ellipsoid(points, ellipsoid)

        np.testing.assert_allclose(ellipsoid_value(clipped, ellipsoid), [1, 0.25])

    def test_alpha_follows_sine_window(self):
        self.assertEqual(alpha_for_z(-1, 0, 10, 60), 0)
        self.assertEqual(alpha_for_z(0, 0, 10, 60), 0)
        self.assertAlmostEqual(alpha_for_z(5, 0, 10, 60), 60)
        self.assertEqual(alpha_for_z(10, 0, 10, 60), 0)
        self.assertEqual(alpha_for_z(5, 5, 5, 60), 0)

    def test_family_is_reachable_ordered_and_on_surface(self):
        ellipsoid = make_ellipsoid([2, 2, 2], rho=5, theta=90, phi=0)
        family = generate_simplex_family(
            ellipsoid,
            phi=0,
            z1=-1,
            z2=1,
            maximum_alpha=30,
            initial_theta=90,
            samples=181,
        )

        self.assertGreater(len(family), 10)
        self.assertTrue(
            all(left.vertex_z < right.vertex_z for left, right in zip(family, family[1:]))
        )
        theta_differences = np.diff([simplex.theta for simplex in family])
        self.assertTrue(np.all(theta_differences > 0) or np.all(theta_differences < 0))
        theta_differences = np.diff([simplex.theta for simplex in family])
        self.assertTrue(np.all(theta_differences > 0) or np.all(theta_differences < 0))
        for simplex in family:
            np.testing.assert_allclose(
                ellipsoid_value(simplex.vertices, ellipsoid),
                np.ones(3),
                atol=1e-6,
            )

    def test_leg_azimuths_are_symmetric(self):
        ellipsoid = make_ellipsoid([2, 2, 2], rho=5, theta=90, phi=0)
        family = generate_simplex_family(
            ellipsoid,
            phi=0,
            z1=-2,
            z2=2,
            maximum_alpha=20,
            initial_theta=90,
            samples=181,
        )
        simplex = nearest_simplex(family, theta=90)
        relative = simplex.vertices[1:] - simplex.vertices[0]

        self.assertGreater(simplex.alpha, 19.9)
        self.assertAlmostEqual(relative[0, 1], -relative[1, 1], places=7)
        self.assertAlmostEqual(relative[0, 0], relative[1, 0], places=7)

    def test_zero_alpha_renders_as_line_but_can_still_classify(self):
        ellipsoid = make_ellipsoid([2, 2, 2], rho=5, theta=90, phi=0)
        family = generate_simplex_family(
            ellipsoid,
            phi=0,
            z1=10,
            z2=11,
            maximum_alpha=30,
            initial_theta=90,
            samples=181,
        )
        simplex = nearest_simplex(family, theta=90)

        self.assertEqual(simplex.alpha, 0)
        np.testing.assert_allclose(simplex.vertices[1], simplex.vertices[2])
        self.assertGreater(
            np.linalg.norm(
                simplex.classification_vertices[1]
                - simplex.classification_vertices[2]
            ),
            0,
        )

    def test_four_regions_follow_triangle_half_planes(self):
        ellipsoid = make_ellipsoid([2, 2, 2], rho=5, theta=90, phi=0)
        family = generate_simplex_family(
            ellipsoid,
            phi=0,
            z1=-2,
            z2=2,
            maximum_alpha=25,
            initial_theta=90,
            samples=181,
        )
        simplex = nearest_simplex(family, theta=90)
        a, b, c = simplex.classification_vertices
        centroid = (a + b + c) / 3
        beyond_opposite = -0.5 * a + 0.75 * b + 0.75 * c
        beyond_b = 0.75 * a - 0.5 * b + 0.75 * c
        beyond_c = 0.75 * a + 0.75 * b - 0.5 * c

        labels = classify_points(
            np.vstack((centroid, beyond_opposite, beyond_b, beyond_c)),
            [simplex],
        )
        np.testing.assert_array_equal(labels, [0, 1, 2, 3])

    def test_every_rendered_simplex_centroid_is_in_green_region(self):
        ellipsoid = make_ellipsoid([3, 2, 2], rho=5, theta=90, phi=45)
        family = generate_simplex_family(
            ellipsoid,
            phi=45,
            z1=-1,
            z2=1,
            maximum_alpha=45,
            initial_theta=90,
        )
        centroids = np.array(
            [np.mean(simplex.classification_vertices, axis=0) for simplex in family]
        )

        np.testing.assert_array_equal(
            classify_points(centroids, family),
            np.zeros(len(family), dtype=int),
        )

    def test_family_uses_one_monotonic_z_branch(self):
        ellipsoid = make_ellipsoid([5.1, 8.5, 6.4], 7.4, 16, 195)
        family = generate_simplex_family(
            ellipsoid,
            phi=183,
            z1=-10,
            z2=10,
            maximum_alpha=20,
            initial_theta=60,
        )
        theta_values = np.array([simplex.theta for simplex in family])

        self.assertGreater(len(family), 2)
        self.assertTrue(np.all(np.diff(theta_values) > 0) or np.all(np.diff(theta_values) < 0))

    def test_family_can_resample_reachable_branch_to_target_count(self):
        ellipsoid = make_ellipsoid([3, 2, 2], rho=5, theta=90, phi=45)
        family = generate_simplex_family(
            ellipsoid,
            phi=45,
            z1=-1,
            z2=1,
            maximum_alpha=45,
            initial_theta=90,
            target_count=500,
        )

        self.assertEqual(len(family), 500)
        theta_differences = np.diff([simplex.theta for simplex in family])
        self.assertTrue(np.all(theta_differences > 0) or np.all(theta_differences < 0))
        self.assertTrue(
            all(left.vertex_z < right.vertex_z for left, right in zip(family, family[1:]))
        )

    def test_target_count_replenishes_filtered_leg_directions(self):
        ellipsoid = make_ellipsoid([3, 2, 2], rho=5, theta=1, phi=0)
        family = generate_simplex_family(
            ellipsoid,
            phi=45,
            z1=-1,
            z2=1,
            maximum_alpha=89.5,
            initial_theta=0,
            target_count=500,
        )

        self.assertEqual(len(family), 500)

    def test_target_count_expands_single_discovery_candidate(self):
        ellipsoid = make_ellipsoid([2, 2, 2], rho=5, theta=90, phi=0)
        family = generate_simplex_family(
            ellipsoid,
            phi=0,
            z1=-1,
            z2=1,
            maximum_alpha=30,
            initial_theta=90,
            samples=5,
            target_count=500,
        )

        self.assertEqual(len(family), 500)

    def test_chunked_classification_matches_single_chunk(self):
        ellipsoid = make_ellipsoid([3, 2, 2], rho=5, theta=90, phi=45)
        family = generate_simplex_family(
            ellipsoid,
            phi=45,
            z1=-1,
            z2=1,
            maximum_alpha=45,
            initial_theta=90,
            target_count=50,
        )
        points = np.array(
            [np.mean(simplex.classification_vertices, axis=0) for simplex in family]
        )

        np.testing.assert_array_equal(
            classify_points(points, family, chunk_size=7),
            classify_points(points, family, chunk_size=len(points)),
        )


if __name__ == "__main__":
    unittest.main()
