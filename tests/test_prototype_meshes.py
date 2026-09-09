import unittest

import numpy as np

from geometry import ellipsoid_value
from prototype_meshes import (
    clip_polygon_halfplane,
    default_prototype_geometry,
    direct_swept_meshes,
    loft_polygon_run,
    mesh_metrics,
    partition_agreement,
    plane_ellipsoid_ellipse,
    resample_closed_polygon,
    simplex_region_polygon,
    surface_nets_meshes,
)


class PrototypeMeshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ellipsoid, cls.family = default_prototype_geometry()

    def test_plane_ellipsoid_ellipse_lies_on_plane_and_hull(self):
        simplex = self.family[len(self.family) // 2]
        triangle = simplex.classification_vertices
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        normal /= np.linalg.norm(normal)
        ellipse = plane_ellipsoid_ellipse(
            self.ellipsoid,
            triangle[0],
            normal,
            samples=80,
        )

        np.testing.assert_allclose(
            ellipsoid_value(ellipse, self.ellipsoid),
            np.ones(len(ellipse)),
            atol=1e-10,
        )
        np.testing.assert_allclose((ellipse - triangle[0]) @ normal, 0, atol=1e-10)

    def test_halfplane_polygon_clipping(self):
        square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
        clipped = clip_polygon_halfplane(square, 0, np.array([1, 0]))

        self.assertTrue(np.all(clipped[:, 0] >= -1e-10))
        self.assertEqual(len(clipped), 4)

    def test_all_direct_region_cross_sections_are_valid(self):
        simplex = self.family[len(self.family) // 2]
        for label in range(4):
            polygon = simplex_region_polygon(self.ellipsoid, simplex, label)
            self.assertGreaterEqual(len(polygon), 3)
            np.testing.assert_array_less(
                ellipsoid_value(polygon, self.ellipsoid),
                np.ones(len(polygon)) + 1e-9,
            )

    def test_degenerate_simplex_plane_is_rejected(self):
        degenerate = self.family[len(self.family) // 2]
        vertices = degenerate.classification_vertices.copy()
        vertices[1:] = vertices[0]
        invalid = type(degenerate)(
            theta=degenerate.theta,
            vertex_z=degenerate.vertex_z,
            alpha=0,
            vertices=vertices,
            classification_vertices=vertices,
        )

        with self.assertRaisesRegex(ValueError, "degenerate"):
            simplex_region_polygon(self.ellipsoid, invalid, 0)

    def test_perimeter_resampling_is_uniform(self):
        square = np.array(
            [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
            dtype=float,
        )
        ring = resample_closed_polygon(square, count=16)
        closed = np.vstack((ring, ring[0]))
        lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)

        self.assertEqual(ring.shape, (16, 3))
        np.testing.assert_allclose(lengths, np.full(16, 0.5))

    def test_perimeter_resampling_ignores_duplicate_vertices(self):
        square_with_duplicates = np.array(
            [
                [-1, -1, 0],
                [1, -1, 0],
                [1, -1, 0],
                [1, 1, 0],
                [-1, 1, 0],
                [-1, -1, 0],
            ],
            dtype=float,
        )

        ring = resample_closed_polygon(square_with_duplicates, count=16)

        self.assertEqual(ring.shape, (16, 3))
        self.assertTrue(np.isfinite(ring).all())

    def test_loft_is_watertight(self):
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        rings = [
            np.column_stack((np.cos(angles), np.sin(angles), np.full(24, z)))
            for z in (0, 1, 2)
        ]
        mesh = loft_polygon_run(rings)
        metrics = mesh_metrics(mesh, self.ellipsoid)

        self.assertGreater(mesh.n_cells, 0)
        self.assertEqual(metrics["boundary_edges"], 0)
        self.assertEqual(metrics["non_manifold_edges"], 0)

    def test_direct_meshes_extend_to_ellipsoid_terminal_tips(self):
        direct = direct_swept_meshes(
            self.ellipsoid,
            self.family,
            perimeter_samples=16,
            ellipse_samples=64,
        )
        all_points = np.vstack([mesh.points for mesh in direct.meshes.values()])
        family_z = np.array([simplex.vertex_z for simplex in self.family])

        self.assertLess(np.min(all_points[:, 2]), np.min(family_z))
        self.assertGreater(np.max(all_points[:, 2]), np.max(family_z))
        self.assertTrue(
            all(mesh.n_open_edges == 0 and mesh.is_manifold for mesh in direct.meshes.values())
        )

    def test_direct_and_surface_nets_generate_four_regions(self):
        direct = direct_swept_meshes(
            self.ellipsoid,
            self.family,
            perimeter_samples=16,
            ellipse_samples=64,
        )
        surface_nets = surface_nets_meshes(
            self.ellipsoid,
            self.family,
            resolution=20,
            smoothing_iterations=4,
        )

        self.assertTrue(all(mesh.n_cells for mesh in direct.meshes.values()))
        self.assertTrue(all(mesh.n_cells for mesh in surface_nets.meshes.values()))
        self.assertTrue(
            all(
                mesh_metrics(mesh, self.ellipsoid)["hull_error"] < 1e-5
                for mesh in surface_nets.meshes.values()
            )
        )

    def test_partition_agreement_reports_bounded_metrics(self):
        direct = direct_swept_meshes(
            self.ellipsoid,
            self.family,
            perimeter_samples=12,
            ellipse_samples=48,
        )
        metrics = partition_agreement(
            direct,
            self.ellipsoid,
            self.family,
            sample_count=60,
        )

        for key in ("classifier_agreement", "gap_fraction", "overlap_fraction"):
            self.assertGreaterEqual(metrics[key], 0)
            self.assertLessEqual(metrics[key], 1)
        self.assertLessEqual(
            metrics["classifier_agreement"],
            1 - metrics["gap_fraction"] - metrics["overlap_fraction"] + 1e-12,
        )


if __name__ == "__main__":
    unittest.main()
