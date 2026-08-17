from __future__ import annotations

import unittest

import numpy as np

from libero.evaluation import ActionSpec, PolicyRequest, RawObservation
from libero.evaluation.adapters import OpenVLAAdapter
from libero.evaluation.clients import available_clients
from libero.evaluation.clients.openvla_client import OpenVLAClient


def observation() -> RawObservation:
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    return RawObservation(
        agentview_rgb=image,
        wrist_rgb=np.zeros_like(image),
        eef_pos=np.zeros(3, dtype=np.float32),
        eef_quat=np.asarray([0, 0, 0, 1], dtype=np.float32),
        gripper_qpos=np.zeros(2, dtype=np.float32),
    )


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, action):
        self.action = action
        self.get_calls = []
        self.post_calls = []

    def get(self, url, timeout):
        self.get_calls.append((url, timeout))
        return FakeResponse({"paths": {"/act": {"post": {}}}})

    def post(self, url, json, timeout):
        self.post_calls.append((url, json, timeout))
        return FakeResponse(self.action)

    def close(self):
        pass


class OpenVLAAdapterTest(unittest.TestCase):
    def test_observation_uses_only_rotated_main_rgb_image(self):
        raw = observation()
        payload = OpenVLAAdapter(image_preprocess="server").adapt_observation(
            raw,
            "pick up the object",
            "libero_10",
        )

        np.testing.assert_array_equal(
            payload["image"],
            raw.agentview_rgb[::-1, ::-1],
        )
        self.assertTrue(payload["image"].flags.c_contiguous)
        self.assertEqual(payload["image"].shape, (2, 3, 3))
        self.assertEqual(payload["image"].dtype, np.uint8)
        self.assertEqual(payload["instruction"], "pick up the object")
        self.assertEqual(payload["unnorm_key"], "libero_10")
        self.assertEqual(set(payload), {"image", "instruction", "unnorm_key"})

    def test_official_libero_image_preprocessing_is_default(self):
        adapter = OpenVLAAdapter()
        payload = adapter.adapt_observation(
            observation(),
            "pick up the object",
            "libero_10",
        )

        self.assertEqual(payload["image"].shape, (224, 224, 3))
        self.assertEqual(payload["image"].dtype, np.uint8)
        self.assertTrue(payload["image"].flags.c_contiguous)
        self.assertEqual(adapter.image_preprocess, "official_libero")
        self.assertFalse(adapter.center_crop)

    def test_official_center_crop_is_optional(self):
        payload = OpenVLAAdapter(center_crop=True).adapt_observation(
            observation(),
            "pick up the object",
            "libero_10",
        )

        self.assertEqual(payload["image"].shape, (224, 224, 3))
        self.assertEqual(payload["image"].dtype, np.uint8)

    def test_center_crop_requires_official_image_preprocessing(self):
        with self.assertRaisesRegex(ValueError, "center_crop"):
            OpenVLAAdapter(image_preprocess="server", center_crop=True)

    def test_unknown_image_preprocessing_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "image_preprocess"):
            OpenVLAAdapter(image_preprocess="unknown")

    def test_non_libero_unnorm_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "starting with 'libero_'"):
            OpenVLAAdapter().adapt_observation(
                observation(),
                "pick up the object",
                "bridge_orig",
            )

    def test_official_action_preserves_first_six_dimensions(self):
        raw = np.asarray(
            [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 1.0],
            dtype=np.float32,
        )

        action = OpenVLAAdapter().actions_from_model_output(raw)

        np.testing.assert_allclose(action[0, :6], raw[:6])
        self.assertEqual(float(action[0, -1]), -1.0)

    def test_native_gripper_is_mapped_once_for_libero(self):
        adapter = OpenVLAAdapter()
        opened = adapter.actions_from_model_output(
            np.asarray([0.1, -0.1, 0, 0, 0, 0, 1.0], dtype=np.float32)
        )
        closed = adapter.actions_from_model_output(
            np.asarray([0, 0, 0, 0, 0, 0, 0.0], dtype=np.float32)
        )

        self.assertEqual(opened.shape, (1, 7))
        self.assertEqual(opened.dtype, np.float32)
        self.assertEqual(float(opened[0, -1]), -1.0)
        self.assertEqual(float(closed[0, -1]), 1.0)

    def test_official_error_string_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "server inference failed"):
            OpenVLAAdapter().actions_from_model_output("error")


class OpenVLAClientTest(unittest.TestCase):
    def test_registry_contains_openvla(self):
        self.assertIn("openvla", available_clients())

    def test_official_act_request_and_direct_numpy_response(self):
        session = FakeSession(
            np.asarray([0.1, -0.1, 0, 0, 0, 0, 1.0], dtype=np.float32)
        )
        client = OpenVLAClient(
            "http://gpu-server:8000",
            timeout_seconds=120,
            unnorm_key="libero_10",
            session=session,
        )

        self.assertTrue(client.check().ready)
        response = client.infer(
            PolicyRequest(
                instruction="pick up the object",
                observation=observation(),
            )
        ).validate(ActionSpec())

        self.assertEqual(session.get_calls, [("http://gpu-server:8000/openapi.json", 120.0)])
        self.assertEqual(len(session.post_calls), 1)
        url, payload, timeout = session.post_calls[0]
        self.assertEqual(url, "http://gpu-server:8000/act")
        self.assertEqual(timeout, 120.0)
        self.assertEqual(payload["unnorm_key"], "libero_10")
        self.assertNotIn("state", payload)
        self.assertNotIn("wrist_image", payload)
        self.assertEqual(response.actions.shape, (1, 7))
        self.assertEqual(float(response.actions[0, -1]), -1.0)
        np.testing.assert_allclose(
            response.metadata["raw_model_action"],
            [0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 1.0],
        )
        self.assertEqual(response.metadata["adapted_action_chunk"][0][-1], -1.0)
        self.assertEqual(response.metadata["model_input_image_shape"], [224, 224, 3])
        self.assertEqual(response.metadata["unnorm_key"], "libero_10")
        self.assertFalse(response.metadata["center_crop"])

    def test_non_libero_unnorm_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "starting with 'libero_'"):
            OpenVLAClient(
                "http://gpu-server:8000",
                unnorm_key="bridge_orig",
                session=FakeSession(np.zeros(7, dtype=np.float32)),
            )

    def test_config_selects_official_libero_image_preprocessing(self):
        import unittest.mock

        session = FakeSession(
            np.asarray([0, 0, 0, 0, 0, 0, 1.0], dtype=np.float32)
        )
        with unittest.mock.patch(
            "libero.evaluation.clients.openvla_client._create_http_session",
            return_value=session,
        ):
            client = OpenVLAClient.from_config(
                {
                    "connection": {"base_url": "http://gpu-server:8000"},
                    "inference": {"unnorm_key": "libero_10"},
                    "adapter": {
                        "image_preprocess": "official_libero",
                        "center_crop": True,
                    },
                }
            )

        self.assertEqual(client.adapter.image_preprocess, "official_libero")
        self.assertTrue(client.adapter.center_crop)
        self.assertEqual(
            client.check().metadata["image_preprocess"],
            "official_libero",
        )
        self.assertEqual(client.check().metadata["unnorm_key"], "libero_10")
        self.assertTrue(client.check().metadata["center_crop"])
        client.close()


if __name__ == "__main__":
    unittest.main()
