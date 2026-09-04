import unittest

import numpy as np

from geometry import (
    ellipsoid_value,
    generate_simplex_family,
    make_ellipsoid,
    phase_grid,
)
from visualization import REGIONS, build_figure, smooth_region_mesh


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        self.ellipsoid = make_ellipsoid([3, 2, 2], 5, 90, 45)
        central_family = generate_simplex_family(
            self.ellipsoid,
            phi=45,
            z1=0,
            z2=0,
            maximum_alpha=0,
            initial_theta=90,
        )
        z1 = central_family[len(central_family) // 4].vertex_z
        z2 = central_family[3 * len(central_family) // 4].vertex_z
        self.family = generate_simplex_family(
            self.ellipsoid,
            phi=45,
            z1=z1,
            z2=z2,
            maximum_alpha=45,
            initial_theta=90,
        )

    def test_phase_grid_contains_all_four_regions(self):
        axes, labels = phase_grid(self.ellipsoid, self.family, resolution=14)

        self.assertEqual(labels.shape, (14, 14, 14))
        self.assertEqual(len(axes), 3)
        self.assertTrue(set(range(4)).issubset(set(np.unique(labels))))

    def test_smooth_region_mesh_is_closed_and_triangular(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[1:4, 1:4, 1:4] = True
        axes = tuple(np.arange(5, dtype=float) for _ in range(3))

        vertices, triangles = smooth_region_mesh(mask, axes)

        self.assertGreater(len(vertices), 0)
        self.assertGreater(len(triangles), 0)
        self.assertEqual(vertices.shape[1], 3)
        self.assertEqual(triangles.shape[1], 3)
        edge_counts = {}
        for triangle in triangles:
            for edge in (
                tuple(sorted((triangle[0], triangle[1]))),
                tuple(sorted((triangle[1], triangle[2]))),
                tuple(sorted((triangle[2], triangle[0]))),
            ):
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        self.assertTrue(all(count == 2 for count in edge_counts.values()))

    def test_figure_has_regions_family_and_selected_simplex(self):
        figure, selected = build_figure(
            self.ellipsoid,
            self.family,
            selected_z=0,
            grid_resolution=12,
        )
        trace_names = {trace.name for trace in figure.data}

        self.assertEqual(len(figure.data), 9)
        self.assertTrue({name for _, name, _ in REGIONS}.issubset(trace_names))
        self.assertIn("Simplex family", trace_names)
        self.assertIn("Selected simplex", trace_names)
        self.assertAlmostEqual(selected.theta, 90)
        for trace in figure.data[:4]:
            vertices = np.column_stack((trace.x, trace.y, trace.z))
            self.assertLessEqual(
                np.max(ellipsoid_value(vertices, self.ellipsoid)),
                1.0 + 1e-10,
            )

    def test_figure_retains_hidden_region_trace_slots(self):
        figure, _ = build_figure(
            self.ellipsoid,
            self.family,
            selected_z=0,
            grid_resolution=12,
            visible_regions={"three-phase"},
        )

        self.assertEqual(len(figure.data), 9)
        self.assertTrue(figure.data[0].visible)
        self.assertTrue(all(trace.visible is False for trace in figure.data[1:4]))

    def test_figure_applies_custom_region_colors(self):
        custom_colors = {
            "three-phase": "#010203",
            "segregative": "#111213",
            "associative-left": "#212223",
            "associative-right": "#313233",
        }
        figure, _ = build_figure(
            self.ellipsoid,
            self.family,
            selected_z=0,
            grid_resolution=12,
            region_colors=custom_colors,
        )

        self.assertEqual(
            [trace.color for trace in figure.data[:4]],
            list(custom_colors.values()),
        )


if __name__ == "__main__":
    unittest.main()
