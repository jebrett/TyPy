import unittest

import prototype_app


class PrototypeAppTests(unittest.TestCase):
    def test_all_presets_generate_exact_family(self):
        for preset_name in prototype_app.PRESETS:
            with self.subTest(preset=preset_name):
                _, family = prototype_app.build_geometry(preset_name)
                self.assertEqual(len(family), 500)

    def test_region_appearance_updates_both_renderers(self):
        prototype_app.state.surface_resolution = 20
        prototype_app.rebuild()
        prototype_app.apply_region_appearance("three-phase", "#123456", False)

        for method in ("direct", "surface_nets"):
            actor = prototype_app.actors[method]["three-phase"]
            self.assertFalse(actor.GetVisibility())
            expected = tuple(channel / 255 for channel in (0x12, 0x34, 0x56))
            for actual, target in zip(actor.prop.color, expected):
                self.assertAlmostEqual(actual, target, places=5)

    def test_client_initialization_populates_both_views(self):
        for method_actors in prototype_app.actors.values():
            method_actors.clear()
        prototype_app.state.surface_resolution = 20

        prototype_app.initialize_client()

        self.assertTrue(prototype_app.actors["direct"])
        self.assertTrue(prototype_app.actors["surface_nets"])


if __name__ == "__main__":
    unittest.main()
