from __future__ import annotations

import base64
import json
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import urlopen

from policy_eval.eval_one_task import latency_stats, parse_init_state_ids, resolve_data_paths
from policy_eval.live_preview import LivePreviewServer, encode_rgb_png
from policy_eval.mock_policy_server import MockPolicy, build_server
from policy_eval.protocol import (
    PROTOCOL_VERSION,
    PolicyClient,
    ProtocolError,
    validate_action_request,
)


def encoded_test_image() -> dict:
    raw = bytes(range(12))
    return {
        "encoding": "base64",
        "dtype": "uint8",
        "shape": [2, 2, 3],
        "color_space": "RGB",
        "orientation": "upright",
        "data": base64.b64encode(raw).decode("ascii"),
    }


def valid_request(horizon: int = 3) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": "test-request-1",
        "task": {
            "suite": "libero_object",
            "task_name": "pick_up_the_cream_cheese_and_place_it_in_the_basket",
            "language_instruction": "pick up the cream cheese and place it in the basket",
        },
        "episode": {"episode_index": 0, "init_state_id": 0, "step": 0},
        "observation": {
            "agentview_rgb": encoded_test_image(),
            "wrist_rgb": encoded_test_image(),
            "proprio": {
                "eef_position": [0.0, 0.0, 0.0],
                "eef_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "gripper_qpos": [0.0, 0.0],
            },
            "state_vector": [0.0] * 8,
        },
        "parameters": {"requested_action_horizon": horizon},
    }


class FakeRGBImage:
    shape = (2, 2, 3)

    class DType:
        name = "uint8"

    dtype = DType()

    def __init__(self) -> None:
        self.raw = bytes(range(12))

    def tobytes(self, order: str = "C") -> bytes:
        self.order = order
        return self.raw


class ProtocolValidationTest(unittest.TestCase):
    def test_valid_request(self) -> None:
        request = valid_request()
        self.assertIs(validate_action_request(request), request)

    def test_rejects_incorrect_image_byte_count(self) -> None:
        request = valid_request()
        request["observation"]["agentview_rgb"]["data"] = base64.b64encode(b"too short").decode()
        with self.assertRaisesRegex(ProtocolError, "expected 12"):
            validate_action_request(request)


class EvaluatorUtilityTest(unittest.TestCase):
    def test_init_state_range_parser(self) -> None:
        self.assertEqual(parse_init_state_ids("0,2-3", 3), [0, 2, 3])

    def test_latency_stats(self) -> None:
        stats = latency_stats([1.0, 2.0, 10.0])
        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["median_ms"], 2.0)
        self.assertEqual(stats["max_ms"], 10.0)

    def test_bddl_and_init_roots_are_selected_independently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / "repo"
            package = repo / "libero" / "libero"
            external = root / "libero_data"
            (package / "assets").mkdir(parents=True)
            (package / "bddl_files" / "libero_object").mkdir(parents=True)
            (package / "init_files" / "different_suite").mkdir(parents=True)
            (external / "bddl_files" / "different_suite").mkdir(parents=True)
            (external / "init_files" / "libero_object").mkdir(parents=True)

            paths = resolve_data_paths(repo, external, "libero_object")

            self.assertEqual(paths["bddl_files"], package / "bddl_files")
            self.assertEqual(paths["init_states"], external / "init_files")

    def test_repository_data_is_the_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / "repo"
            package = repo / "libero" / "libero"
            (package / "assets").mkdir(parents=True)
            (package / "bddl_files" / "libero_object").mkdir(parents=True)
            (package / "init_files" / "libero_object").mkdir(parents=True)

            paths = resolve_data_paths(repo, None, "libero_object")

            self.assertEqual(paths["bddl_files"], package / "bddl_files")
            self.assertEqual(paths["init_states"], package / "init_files")


class MockServerIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = build_server(
            host="127.0.0.1",
            port=0,
            policy=MockPolicy(mode="noop", default_horizon=5),
            quiet=True,
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        host, port = cls.server.server_address[:2]
        cls.client = PolicyClient(f"http://{host}:{port}", timeout_seconds=2.0)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2.0)

    def test_health_and_metadata(self) -> None:
        self.assertEqual(self.client.health()["status"], "ok")
        self.assertEqual(self.client.metadata()["model_name"], "mock/noop")

    def test_noop_action_chunk(self) -> None:
        response, actions, round_trip_ms = self.client.infer(valid_request(horizon=3))
        self.assertEqual(response["request_id"], "test-request-1")
        self.assertEqual(actions, [[0.0] * 6 + [-1.0]] * 3)
        self.assertGreaterEqual(round_trip_ms, 0.0)


class LivePreviewTest(unittest.TestCase):
    def test_png_encoder_preserves_dimensions(self) -> None:
        payload = encode_rgb_png(FakeRGBImage())
        self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(payload[12:16], b"IHDR")
        self.assertEqual(struct.unpack(">II", payload[16:24]), (2, 2))

    def test_preview_serves_frames_and_status(self) -> None:
        preview = LivePreviewServer(host="127.0.0.1", port=0, refresh_hz=10)
        url = preview.start()
        try:
            preview.publish(
                agentview_rgb=FakeRGBImage(),
                wrist_rgb=FakeRGBImage(),
                status={"phase": "evaluating", "step": 3, "success": False},
            )
            with urlopen(url + "api/status", timeout=2.0) as response:
                status = json.loads(response.read().decode("utf-8"))
            self.assertEqual(status["phase"], "evaluating")
            self.assertEqual(status["step"], 3)
            self.assertEqual(status["frame_sequence"], 1)

            with urlopen(url + "frame/agentview.png", timeout=2.0) as response:
                self.assertEqual(response.headers.get_content_type(), "image/png")
                self.assertTrue(response.read().startswith(b"\x89PNG\r\n\x1a\n"))
        finally:
            preview.close()


if __name__ == "__main__":
    unittest.main()
