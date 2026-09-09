import unittest
from pathlib import Path

import numpy as np

import main
from geometry import spherical_direction


class ProductionTrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        main.state.ellipsoid_length = 3.0
        main.state.ellipsoid_width = 2.0
        main.state.ellipsoid_height = 2.0
        main.state.ellipsoid_theta = 90.0
        main.state.ellipsoid_phi = 45.0
        main.state.rho = 5.0
        main.state.z_offset = 0.0
        main.state.simplex_theta = 90.0
        main.state.simplex_phi = 45.0
        main.state.maximum_alpha = 45.0
        if not main.actors:
            main.rebuild_scene(reset_range=True)

    def test_production_scene_contains_all_layers(self):
        self.assertEqual(len(main.current_family), main.FAMILY_SIZE)
        self.assertTrue(
            {
                *main.REGION_KEYS,
                "ellipsoid",
                "family",
                "selected-fill",
                "selected-outline",
                "axes",
            }.issubset(main.actors)
        )

    def test_all_legacy_controls_have_trame_state(self):
        expected = {
            "ellipsoid_length",
            "ellipsoid_width",
            "ellipsoid_height",
            "ellipsoid_theta",
            "ellipsoid_phi",
            "rho",
            "z_offset",
            "simplex_theta",
            "simplex_phi",
            "maximum_alpha",
            "z_window",
            "slice_z",
        }
        for key in ("three-phase", "segregative"):
            suffix = key.replace("-", "_")
            expected.update({f"color_{suffix}", f"visible_{suffix}"})
        expected.update({"color_associative", "visible_associative"})

        self.assertTrue(all(key in main.state for key in expected))

    def test_ellipsoid_z_offset_translates_center_intercept(self):
        original_offset = main.state.z_offset
        try:
            main.state.z_offset = 1.75
            ellipsoid = main._ellipsoid_from_state()
            directed_center = (
                main.state.rho
                * spherical_direction(
                    main.state.ellipsoid_theta,
                    main.state.ellipsoid_phi,
                )
            )

            np.testing.assert_allclose(
                ellipsoid.center,
                directed_center + [0, 0, 1.75],
            )
        finally:
            main.state.z_offset = original_offset

    def test_slice_update_changes_only_selected_datasets_and_preserves_camera(self):
        original_regions = {key: id(main.datasets[key]) for key in main.REGION_KEYS}
        original_fill = main.datasets["selected-fill"].points.copy()
        camera = [
            (10.0, 10.0, 10.0),
            (3.0, 3.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        main.plotter.camera_position = camera

        main.update_selected_slice(main.current_family[100].vertex_z)

        self.assertFalse(
            np.allclose(original_fill, main.datasets["selected-fill"].points)
        )
        self.assertEqual(
            original_regions,
            {key: id(main.datasets[key]) for key in main.REGION_KEYS},
        )
        np.testing.assert_allclose(list(main.plotter.camera_position), camera)

    def test_invalid_geometry_keeps_last_valid_scene(self):
        original_actors = dict(main.actors)
        original_rho = main.state.rho
        try:
            main.state.rho = 0.0
            success = main.rebuild_scene(reset_range=True)

            self.assertFalse(success)
            self.assertEqual(original_actors, main.actors)
            self.assertIn("last valid analytical render", main.state.error_message)
        finally:
            main.state.rho = original_rho
            self.assertTrue(main.rebuild_scene(reset_range=True))

    def test_valid_rebuild_preserves_camera(self):
        camera = [
            (10.0, 9.0, 8.0),
            (3.0, 3.0, 0.0),
            (0.0, 0.0, 1.0),
        ]
        main.plotter.camera_position = camera

        self.assertTrue(main.rebuild_scene(reset_range=False))

        np.testing.assert_allclose(list(main.plotter.camera_position), camera)

    def test_incomplete_polar_partition_keeps_last_valid_scene(self):
        original_actors = dict(main.actors)
        original_theta = main.state.ellipsoid_theta
        original_range = (
            main.state.z_min,
            main.state.z_max,
            list(main.state.z_window),
            main.state.slice_z,
        )
        try:
            main.state.ellipsoid_theta = 0.1
            success = main.rebuild_scene(reset_range=True)

            self.assertFalse(success)
            self.assertEqual(original_actors, main.actors)
            self.assertIn("last valid analytical render", main.state.error_message)
            self.assertEqual(
                (
                    main.state.z_min,
                    main.state.z_max,
                    list(main.state.z_window),
                    main.state.slice_z,
                ),
                original_range,
            )
        finally:
            main.state.ellipsoid_theta = original_theta
            self.assertTrue(main.rebuild_scene(reset_range=True))

    def test_region_appearance_updates_actor_without_rebuild(self):
        actor = main.actors["three-phase"]
        original_id = id(actor)

        main.apply_region_appearance("three-phase", "#123456", False)

        self.assertEqual(id(main.actors["three-phase"]), original_id)
        self.assertFalse(actor.GetVisibility())
        expected = tuple(channel / 255 for channel in (0x12, 0x34, 0x56))
        for actual, target in zip(actor.prop.color, expected):
            self.assertAlmostEqual(actual, target, places=5)
        main.apply_region_appearance(
            "three-phase",
            main.REGION_COLORS["three-phase"],
            True,
        )

    def test_arbitrary_region_hex_color_is_supported(self):
        main.apply_region_appearance("segregative", "#4a17c9", True)
        expected = tuple(channel / 255 for channel in (0x4A, 0x17, 0xC9))

        for actual, target in zip(main.actors["segregative"].prop.color, expected):
            self.assertAlmostEqual(actual, target, places=5)
        main.apply_region_appearance(
            "segregative",
            main.REGION_COLORS["segregative"],
            True,
        )

    def test_associative_appearance_control_updates_both_regions(self):
        main.apply_associative_appearance("#8a2be2", False)

        expected = tuple(channel / 255 for channel in (0x8A, 0x2B, 0xE2))
        for key in ("associative-left", "associative-right"):
            self.assertFalse(main.actors[key].GetVisibility())
            for actual, target in zip(main.actors[key].prop.color, expected):
                self.assertAlmostEqual(actual, target, places=5)
        main.apply_associative_appearance(
            main.REGION_COLORS["associative-left"], True
        )

    def test_current_view_exports_svg_and_pdf(self):
        generated = []
        try:
            for export_format, signature in (("svg", b"<svg"), ("pdf", b"%PDF")):
                self.assertTrue(main.export_graphic(export_format))
                destination = main.EXPORT_DIRECTORY / main.state.export_filename
                generated.append(destination)
                self.assertTrue(destination.exists())
                content = destination.read_bytes()
                self.assertIn(signature, content[:500])
                self.assertEqual(main.state.export_url, f"/exports/{destination.name}")
                if len(generated) > 1:
                    self.assertFalse(generated[-2].exists())
        finally:
            for destination in generated:
                destination.unlink(missing_ok=True)

    def test_unsupported_export_format_surfaces_error(self):
        self.assertFalse(main.export_graphic("png"))
        self.assertIn("Unsupported export format", main.state.export_error)

    def test_production_view_uses_server_synchronized_camera(self):
        self.assertEqual(main.view.__class__.__name__, "VtkRemoteView")

    def test_download_uses_native_anchor_attribute(self):
        markup = main.layout.html

        self.assertIn('download="export_filename"', markup)


if __name__ == "__main__":
    unittest.main()
