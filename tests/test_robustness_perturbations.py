from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ENVS_ROOT = ROOT / "libero" / "libero" / "envs"


def load_modules():
    package_name = "libero_pro_pr1_envs"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ENVS_ROOT)]
    sys.modules[package_name] = package

    parser_name = f"{package_name}.perturbation_config"
    parser_spec = importlib.util.spec_from_file_location(
        parser_name, ENVS_ROOT / "perturbation_config.py"
    )
    parser_module = importlib.util.module_from_spec(parser_spec)
    sys.modules[parser_name] = parser_module
    assert parser_spec and parser_spec.loader
    parser_spec.loader.exec_module(parser_module)

    runtime_name = f"{package_name}.robustness_perturbations"
    runtime_spec = importlib.util.spec_from_file_location(
        runtime_name, ENVS_ROOT / "robustness_perturbations.py"
    )
    runtime_module = importlib.util.module_from_spec(runtime_spec)
    sys.modules[runtime_name] = runtime_module
    assert runtime_spec and runtime_spec.loader
    runtime_spec.loader.exec_module(runtime_module)
    return parser_module, runtime_module


PARSER, RUNTIME = load_modules()


class PerturbationConfigTests(unittest.TestCase):
    def test_parses_near_grasp_runtime_config(self):
        text = """
(define (problem test)
  (:domain robosuite)
  (:perturbation_config
    (:case runtime_object_move)
    (:runtime_move
      (:enabled true)
      (:mode qpos_delta_xy)
      (:object bowl_1)
      (:step 80)
      (:trigger near_grasp)
      (:distance_threshold 0.09)
      (:fallback_step 160)
      (:delta_xy 0.17 0.06)
      (:min_delta_norm 0.18)
    )
  )
)
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "task.bddl"
            path.write_text(text, encoding="utf-8")
            config = PARSER.parse_bddl_perturbation_config(path)

        runtime = config["runtime_move"]
        self.assertEqual(runtime["trigger"], "near_grasp")
        self.assertEqual(runtime["distance_threshold"], 0.09)
        self.assertEqual(runtime["fallback_step"], 160)
        self.assertEqual(runtime["delta_xy"], [0.17, 0.06])

    def test_infers_active_case_spec(self):
        config = {
            "case": "visual_noise_glare",
            "observation": {
                "enabled": True,
                "mode": "dark_noise",
                "brightness": 0.5,
                "sigma": 0.0,
            },
        }
        spec = RUNTIME.infer_spec_from_config(
            "visual_noise_glare", "libero_goal", "task", config
        )
        self.assertEqual(spec.observation_perturbation, "dark_noise")
        self.assertEqual(spec.perturbation_layer, "bddl_config_policy_observation_pixels")


class RuntimeTriggerTests(unittest.TestCase):
    def setUp(self):
        self.settings = {
            "trigger": "near_grasp",
            "configured_step": 80,
            "fallback_step": 160,
            "distance_threshold": 0.09,
            "distance_metric": "xyz",
        }

    def test_triggers_at_nine_centimeters(self):
        result = RUNTIME.runtime_move_trigger_decision(
            42, [0.0, 0.0, 0.0], [0.09, 0.0, 0.0], self.settings
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["reason"], "distance_threshold")

    def test_waits_when_target_is_far(self):
        result = RUNTIME.runtime_move_trigger_decision(
            159, [0.0, 0.0, 0.0], [0.3, 0.0, 0.0], self.settings
        )
        self.assertIsNone(result)

    def test_uses_fallback_step(self):
        result = RUNTIME.runtime_move_trigger_decision(
            160, [0.0, 0.0, 0.0], [0.3, 0.0, 0.0], self.settings
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["reason"], "fallback_step")


class ObservationTests(unittest.TestCase):
    def test_dark_noise_uses_bddl_values(self):
        config = {
            "observation": {
                "enabled": True,
                "mode": "dark_noise",
                "brightness": 0.5,
                "sigma": 0.0,
            }
        }
        spec = RUNTIME.infer_spec_from_config(
            "visual_noise_glare", "libero_goal", "task", config
        )
        original = np.full((1, 4, 4, 3), 200, dtype=np.uint8)
        observation = {"pixels": {"agentview": original}}
        transformed = RUNTIME.apply_case_to_observation(
            observation, spec, seed=1, args=RUNTIME.PerturbationRuntimeOptions()
        )

        self.assertTrue(np.all(transformed["pixels"]["agentview"] == 100))
        self.assertTrue(np.all(original == 200))


if __name__ == "__main__":
    unittest.main()
