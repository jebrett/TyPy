import unittest

from dash import no_update

from geometry import generate_simplex_family, make_ellipsoid, nearest_simplex
from main import (
    DEFAULTS,
    FAMILY_SIZE,
    REGION_COLOR_IDS,
    app,
    color_control,
    region_color_patch,
    region_visibility_patch,
    selected_simplex_patch,
    update_figure,
    update_figure_for_trigger,
    update_z_controls,
)


class AppCallbackTests(unittest.TestCase):
    def test_default_callbacks_build_interactive_figure(self):
        z_controls = update_z_controls(
            DEFAULTS["length"],
            DEFAULTS["width"],
            DEFAULTS["height"],
            DEFAULTS["ellipsoid_theta"],
            DEFAULTS["ellipsoid_phi"],
            DEFAULTS["rho"],
            DEFAULTS["simplex_theta"],
            DEFAULTS["simplex_phi"],
        )
        figure, status = update_figure(
            DEFAULTS["length"],
            DEFAULTS["width"],
            DEFAULTS["height"],
            DEFAULTS["ellipsoid_theta"],
            DEFAULTS["ellipsoid_phi"],
            DEFAULTS["rho"],
            DEFAULTS["simplex_theta"],
            DEFAULTS["simplex_phi"],
            DEFAULTS["maximum_alpha"],
            z_controls[3],
            z_controls[8],
        )

        self.assertGreaterEqual(len(figure.data), 8)
        self.assertIn("Selected 2-simplex", status)
        self.assertIn(f"Rendered {FAMILY_SIZE} reachable simplexes", status)

    def test_origin_inside_hull_returns_clear_empty_state(self):
        figure, status = update_figure(
            5,
            5,
            5,
            90,
            0,
            0,
            90,
            0,
            45,
            [-1, 1],
            0,
        )

        self.assertEqual(len(figure.data), 0)
        self.assertIn("No origin-based entry simplex", status)

    def test_region_toggle_patch_only_changes_four_region_traces(self):
        patch = region_visibility_patch(["three-phase", "associative-left"])
        operations = patch.to_plotly_json()["operations"]

        self.assertEqual(len(operations), 4)
        self.assertEqual(
            [operation["location"] for operation in operations],
            [
                ["data", 0, "visible"],
                ["data", 1, "visible"],
                ["data", 2, "visible"],
                ["data", 3, "visible"],
            ],
        )
        self.assertEqual(
            [operation["params"]["value"] for operation in operations],
            [True, False, True, False],
        )

    def test_region_color_patch_only_changes_selected_trace(self):
        for color_input_id, (trace_index, _) in REGION_COLOR_IDS.items():
            patch = region_color_patch(color_input_id, "#abcdef")
            operations = patch.to_plotly_json()["operations"]

            self.assertEqual(len(operations), 1)
            self.assertEqual(
                operations[0]["location"],
                ["data", trace_index, "color"],
            )
            self.assertEqual(operations[0]["params"]["value"], "#abcdef")

    def test_slice_patch_only_changes_orange_traces(self):
        ellipsoid = make_ellipsoid([3, 2, 2], 5, 90, 45)
        family = generate_simplex_family(
            ellipsoid,
            phi=45,
            z1=-1,
            z2=1,
            maximum_alpha=45,
            initial_theta=90,
        )
        patch = selected_simplex_patch(nearest_simplex(family, theta=90))
        operations = patch.to_plotly_json()["operations"]

        self.assertTrue(operations)
        self.assertEqual(
            {operation["location"][1] for operation in operations},
            {6, 7},
        )
        self.assertTrue(all(operation["location"][0] == "data" for operation in operations))

    def test_slice_trigger_returns_patch_without_layout_changes(self):
        z_controls = update_z_controls(
            DEFAULTS["length"],
            DEFAULTS["width"],
            DEFAULTS["height"],
            DEFAULTS["ellipsoid_theta"],
            DEFAULTS["ellipsoid_phi"],
            DEFAULTS["rho"],
            DEFAULTS["simplex_theta"],
            DEFAULTS["simplex_phi"],
        )
        patch, status = update_figure_for_trigger(
            "slice-z",
            DEFAULTS["length"],
            DEFAULTS["width"],
            DEFAULTS["height"],
            DEFAULTS["ellipsoid_theta"],
            DEFAULTS["ellipsoid_phi"],
            DEFAULTS["rho"],
            DEFAULTS["simplex_theta"],
            DEFAULTS["simplex_phi"],
            DEFAULTS["maximum_alpha"],
            z_controls[3],
            z_controls[8],
            ["three-phase", "segregative"],
        )
        operations = patch.to_plotly_json()["operations"]

        self.assertIn("Selected 2-simplex", status)
        self.assertTrue(all(operation["location"][0] == "data" for operation in operations))
        self.assertNotIn("layout", {operation["location"][0] for operation in operations})

    def test_visibility_trigger_skips_geometry_rebuild(self):
        patch, status = update_figure_for_trigger(
            "region-visibility",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            ["three-phase"],
        )

        self.assertEqual(len(patch.to_plotly_json()["operations"]), 4)
        self.assertIs(status, no_update)

    def test_collapsed_simplex_patch_hides_orange_fill(self):
        ellipsoid = make_ellipsoid([3, 2, 2], 5, 90, 45)
        family = generate_simplex_family(
            ellipsoid,
            phi=45,
            z1=10,
            z2=11,
            maximum_alpha=45,
            initial_theta=90,
        )
        patch = selected_simplex_patch(nearest_simplex(family, theta=90))
        operations = patch.to_plotly_json()["operations"]
        fill_visibility = [
            operation["params"]["value"]
            for operation in operations
            if operation["location"] == ["data", 6, "visible"]
        ]

        self.assertEqual(fill_visibility, [False])

    def test_layout_has_three_collapsible_control_groups(self):
        aside = app.layout.children[1].children[0]
        groups = aside.children

        self.assertEqual(len(groups), 3)
        self.assertTrue(all(group.__class__.__name__ == "Details" for group in groups))
        self.assertEqual(
            [group.children[0].children for group in groups],
            ["Ellipsoid controls", "Region appearance", "Simplex controls"],
        )

    def test_color_picker_uses_supported_dropdown(self):
        picker = color_control("Region", "region-color", "#2ca02c").children[1]

        self.assertEqual(picker._namespace, "dash_core_components")
        self.assertFalse(picker.clearable)
        self.assertGreaterEqual(len(picker.options), 10)


if __name__ == "__main__":
    unittest.main()
