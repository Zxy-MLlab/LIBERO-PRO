from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_40task_robustness_bddls.py"
SPEC = importlib.util.spec_from_file_location("pr1_generator", SCRIPT)
GENERATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GENERATOR
assert SPEC and SPEC.loader
SPEC.loader.exec_module(GENERATOR)


def base_task() -> dict[str, object]:
    return {
        "objects": {
            "akita_black_bowl_1": "akita_black_bowl",
            "plate_1": "plate",
        },
        "target": "wooden_cabinet_1",
        "appearance_targets": ["flat_stove_1"],
    }


class GeneratorOverrideTests(unittest.TestCase):
    def test_fixture_task_uses_movable_inactive_target(self):
        task = GENERATOR.apply_task_overrides(
            base_task(), "libero_goal", "open_the_middle_drawer_of_the_cabinet"
        )
        self.assertEqual(task["inactive_target"], "akita_black_bowl_1")

    def test_texture_override_does_not_change_default_appearance_target(self):
        task = base_task()
        task["target"] = "plate_1"
        task = GENERATOR.apply_task_overrides(
            task, "libero_goal", "push_the_plate_to_the_front_of_the_stove"
        )
        self.assertEqual(task["appearance_targets"], ["flat_stove_1"])
        self.assertEqual(task["active_appearance_targets"], ["plate_1"])

    def test_wine_bottle_occlusion_override_preserves_published_format(self):
        task = base_task()
        task["target"] = "akita_black_bowl_1"
        task = GENERATOR.apply_task_overrides(
            task, "libero_goal", "put_the_wine_bottle_on_the_rack"
        )
        self.assertEqual(task["occlusion_between_object"], "akita_black_bowl_1")
        self.assertEqual(task["occlusion_min_between_distance"], "0.18")
        self.assertEqual(task["occlusion_between_delta_xy"], ["-0.18", "0.10"])
        self.assertTrue(task["occlusion_between_after_avoid"])


if __name__ == "__main__":
    unittest.main()
