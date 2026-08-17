"""Adapter from the evaluator protocol to OpenPI's official client."""

import math
import time
from typing import Any, Dict

import numpy as np

from ..policy_client import ClientInfo, PolicyClient
from ..protocol import ActionSpec, PolicyRequest, PolicyResponse
from .registry import register_client


class PolicyConnectionError(RuntimeError):
    pass


class PolicyTimeoutError(TimeoutError):
    pass


@register_client("openpi")
class OpenPIClient(PolicyClient):
    def __init__(self, host: str, port: int, timeout_seconds: float = 30.0,
                 image_size=224, api_key=None):
        if timeout_seconds <= 0:
            raise ValueError("connection.timeout_seconds must be positive")
        self.host, self.port = host, int(port)
        self.timeout_seconds = float(timeout_seconds)
        self.api_key = api_key
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError("inference.image_size must be positive")
        self._policy = None
        self._metadata: Dict[str, Any] = {}

    @classmethod
    def from_config(cls, cfg):
        connection = cfg.get("connection", {})
        if not connection.get("host") or connection.get("port") is None:
            raise ValueError("openpi connection.host and connection.port are required")
        inference = cfg.get("inference", {})
        return cls(connection["host"], connection["port"], connection.get("timeout_seconds", 30),
                   inference.get("image_size", 224), connection.get("api_key"))

    @staticmethod
    def _policy_class():
        try:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy
        except ImportError as exc:
            raise ImportError("OpenPIClient requires openpi-client==0.1.2") from exc
        return WebsocketClientPolicy

    def _connect(self):
        if self._policy is not None:
            return
        try:
            policy = self._policy_class()(self.host, self.port, api_key=self.api_key)
            self._metadata = dict(policy.get_server_metadata())
            self._policy = policy
        except Exception as exc:
            self.close()
            raise PolicyConnectionError("policy connection failed: {}".format(exc)) from exc

    def check(self):
        self._connect()
        return ClientInfo(True, "openpi", str(self._metadata.get("model_name", "")), dict(self._metadata))

    def close(self):
        policy, self._policy = self._policy, None
        if policy is not None:
            try:
                # openpi-client 0.1.2 doesn't expose close(), but owns this
                # connection. Prefer the public method when a future release adds it.
                close = getattr(policy, "close", None)
                if close is not None:
                    close()
                else:
                    policy._ws.close()
            except Exception:
                pass

    def reset(self, episode_id: str, instruction: str) -> None:
        self._connect()
        self._policy.reset()

    def infer(self, request: PolicyRequest) -> PolicyResponse:
        self._connect()
        payload = self._default_request_adapter(request)
        try:
            result = self._policy.infer(payload)
        except TimeoutError as exc:
            self.close()
            raise PolicyTimeoutError("policy inference timed out") from exc
        except (PolicyTimeoutError, ValueError):
            raise
        except Exception as exc:
            self.close()
            raise PolicyConnectionError("policy disconnected: {}".format(exc)) from exc
        if not isinstance(result, dict) or "actions" not in result:
            raise ValueError("OpenPI response does not contain 'actions'")
        spec_dict = result.get("action_spec", {})
        spec = ActionSpec(**spec_dict) if spec_dict else ActionSpec()
        return PolicyResponse(np.asarray(result["actions"]), spec,
                              self._response_metadata(result))

    @staticmethod
    def _response_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
        """Preserve OpenPI timing fields and expose server inference uniformly."""
        metadata = dict(result.get("metadata", {}))
        for key in ("server_timing", "policy_timing"):
            timing = result.get(key)
            if isinstance(timing, dict):
                metadata[key] = dict(timing)
        server_timing = result.get("server_timing")
        if ("server_inference_latency_ms" not in metadata
                and isinstance(server_timing, dict)
                and server_timing.get("infer_ms") is not None):
            metadata["server_inference_latency_ms"] = float(server_timing["infer_ms"])
        return metadata

    def _default_request_adapter(self, request: PolicyRequest) -> Dict[str, Any]:
        obs = request.observation
        try:
            from openpi_client import image_tools
        except ImportError as exc:
            raise ImportError("OpenPIClient requires openpi-client==0.1.2") from exc

        # Match models/openpi/examples/libero/main.py exactly. MuJoCo camera
        # observations must be rotated by 180 degrees to match LIBERO training
        # preprocessing, then resized with padding before transport.
        image = np.ascontiguousarray(obs.agentview_rgb[::-1, ::-1])
        wrist_image = np.ascontiguousarray(obs.wrist_rgb[::-1, ::-1])
        image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(image, self.image_size, self.image_size)
        )
        wrist_image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_image, self.image_size, self.image_size)
        )
        state = np.concatenate((
            obs.eef_pos,
            self._quat2axisangle(obs.eef_quat),
            obs.gripper_qpos,
        ))
        return {
            "observation/image": image,
            "observation/wrist_image": wrist_image,
            "observation/state": state,
            "prompt": request.instruction,
        }

    @staticmethod
    def _quat2axisangle(quat) -> np.ndarray:
        """Convert an xyzw quaternion using OpenPI's LIBERO reference logic."""
        quat = np.asarray(quat, dtype=np.float64)
        w = float(np.clip(quat[3], -1.0, 1.0))
        den = math.sqrt(max(0.0, 1.0 - w * w))
        if math.isclose(den, 0.0):
            return np.zeros(3, dtype=np.float64)
        return quat[:3] * (2.0 * math.acos(w) / den)
