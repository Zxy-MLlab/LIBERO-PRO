"""Dependency-light wire protocol shared by the evaluator and policy server.

The reference transport is JSON over HTTP. Images are transmitted as base64 encoded,
contiguous RGB uint8 bytes. Keeping this module free of NumPy and LIBERO imports lets
the mock server run in a small, separate Python environment.
"""

from __future__ import annotations

import base64
import json
import math
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


PROTOCOL_VERSION = "1.0"
ACTION_DIM = 7
DEFAULT_MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class ProtocolError(ValueError):
    """Raised when a request or response violates the policy protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_float_list(
    value: Any, *, name: str, expected_length: Optional[int] = None
) -> List[float]:
    _require(isinstance(value, list), f"{name} must be a JSON list")
    if expected_length is not None:
        _require(
            len(value) == expected_length,
            f"{name} must contain {expected_length} values; got {len(value)}",
        )
    result: List[float] = []
    for index, item in enumerate(value):
        _require(_is_number(item), f"{name}[{index}] must be a number")
        converted = float(item)
        _require(math.isfinite(converted), f"{name}[{index}] must be finite")
        result.append(converted)
    return result


def encode_rgb_uint8(image: Any) -> Dict[str, Any]:
    """Encode an HWC NumPy-like uint8 RGB array without importing NumPy here."""

    shape = tuple(int(v) for v in getattr(image, "shape", ()))
    dtype_name = str(getattr(image, "dtype", ""))
    _require(len(shape) == 3 and shape[2] == 3, "RGB image must have shape [H, W, 3]")
    _require(dtype_name == "uint8", f"RGB image dtype must be uint8; got {dtype_name!r}")
    raw = image.tobytes(order="C")
    _require(len(raw) == shape[0] * shape[1] * 3, "RGB image is not contiguous uint8 data")
    return {
        "encoding": "base64",
        "dtype": "uint8",
        "shape": [shape[0], shape[1], shape[2]],
        "color_space": "RGB",
        "orientation": "upright",
        "data": base64.b64encode(raw).decode("ascii"),
    }


def validate_encoded_rgb(value: Any, *, name: str) -> Tuple[int, int, int]:
    _require(isinstance(value, dict), f"{name} must be an object")
    _require(value.get("encoding") == "base64", f"{name}.encoding must be 'base64'")
    _require(value.get("dtype") == "uint8", f"{name}.dtype must be 'uint8'")
    _require(value.get("color_space") == "RGB", f"{name}.color_space must be 'RGB'")
    _require(value.get("orientation") == "upright", f"{name}.orientation must be 'upright'")
    shape = value.get("shape")
    _require(
        isinstance(shape, list)
        and len(shape) == 3
        and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in shape),
        f"{name}.shape must be three positive integers",
    )
    _require(shape[2] == 3, f"{name} must have exactly three RGB channels")
    encoded = value.get("data")
    _require(isinstance(encoded, str), f"{name}.data must be a base64 string")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise ProtocolError(f"{name}.data is not valid base64") from exc
    expected_size = shape[0] * shape[1] * shape[2]
    _require(
        len(raw) == expected_size,
        f"{name}.data has {len(raw)} bytes; expected {expected_size}",
    )
    return int(shape[0]), int(shape[1]), int(shape[2])


def make_action_request(
    *,
    suite: str,
    task_name: str,
    language_instruction: str,
    episode_index: int,
    init_state_id: int,
    step: int,
    agentview_rgb: Any,
    wrist_rgb: Any,
    eef_position: Sequence[float],
    eef_quaternion_xyzw: Sequence[float],
    gripper_qpos: Sequence[float],
    state_vector: Sequence[float],
    requested_action_horizon: int,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one canonical, model-independent action request."""

    payload: Dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id or str(uuid.uuid4()),
        "task": {
            "suite": suite,
            "task_name": task_name,
            "language_instruction": language_instruction,
        },
        "episode": {
            "episode_index": int(episode_index),
            "init_state_id": int(init_state_id),
            "step": int(step),
        },
        "observation": {
            "agentview_rgb": encode_rgb_uint8(agentview_rgb),
            "wrist_rgb": encode_rgb_uint8(wrist_rgb),
            "proprio": {
                "eef_position": [float(v) for v in eef_position],
                "eef_quaternion_xyzw": [float(v) for v in eef_quaternion_xyzw],
                "gripper_qpos": [float(v) for v in gripper_qpos],
            },
            # This convenience vector follows the common LIBERO policy layout:
            # xyz + axis-angle + two gripper joint positions = eight values.
            "state_vector": [float(v) for v in state_vector],
        },
        "parameters": {"requested_action_horizon": int(requested_action_horizon)},
    }
    validate_action_request(payload)
    return payload


def validate_action_request(payload: Any) -> Dict[str, Any]:
    _require(isinstance(payload, dict), "request body must be a JSON object")
    _require(
        payload.get("protocol_version") == PROTOCOL_VERSION,
        f"protocol_version must be {PROTOCOL_VERSION!r}",
    )
    request_id = payload.get("request_id")
    _require(isinstance(request_id, str) and bool(request_id), "request_id must be a non-empty string")

    task = payload.get("task")
    _require(isinstance(task, dict), "task must be an object")
    for key in ("suite", "task_name", "language_instruction"):
        _require(isinstance(task.get(key), str), f"task.{key} must be a string")

    episode = payload.get("episode")
    _require(isinstance(episode, dict), "episode must be an object")
    for key in ("episode_index", "init_state_id", "step"):
        value = episode.get(key)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"episode.{key} must be a non-negative integer",
        )

    observation = payload.get("observation")
    _require(isinstance(observation, dict), "observation must be an object")
    validate_encoded_rgb(observation.get("agentview_rgb"), name="observation.agentview_rgb")
    validate_encoded_rgb(observation.get("wrist_rgb"), name="observation.wrist_rgb")

    proprio = observation.get("proprio")
    _require(isinstance(proprio, dict), "observation.proprio must be an object")
    _finite_float_list(proprio.get("eef_position"), name="observation.proprio.eef_position", expected_length=3)
    _finite_float_list(
        proprio.get("eef_quaternion_xyzw"),
        name="observation.proprio.eef_quaternion_xyzw",
        expected_length=4,
    )
    gripper = _finite_float_list(
        proprio.get("gripper_qpos"), name="observation.proprio.gripper_qpos"
    )
    _require(bool(gripper), "observation.proprio.gripper_qpos cannot be empty")
    _finite_float_list(
        observation.get("state_vector"),
        name="observation.state_vector",
        expected_length=8,
    )

    parameters = payload.get("parameters")
    _require(isinstance(parameters, dict), "parameters must be an object")
    horizon = parameters.get("requested_action_horizon")
    _require(
        isinstance(horizon, int) and not isinstance(horizon, bool) and 1 <= horizon <= 100,
        "parameters.requested_action_horizon must be an integer in [1, 100]",
    )
    return payload


def make_action_response(
    *,
    request_id: str,
    model_name: str,
    actions: Sequence[Sequence[float]],
    inference_ms: float,
) -> Dict[str, Any]:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "model_name": model_name,
        "actions": [[float(value) for value in action] for action in actions],
        "inference_ms": float(inference_ms),
    }
    validate_action_response(payload, expected_request_id=request_id)
    return payload


def validate_action_response(
    payload: Any,
    *,
    expected_request_id: Optional[str] = None,
    require_normalized_actions: bool = True,
) -> List[List[float]]:
    _require(isinstance(payload, dict), "response body must be a JSON object")
    _require(
        payload.get("protocol_version") == PROTOCOL_VERSION,
        f"response protocol_version must be {PROTOCOL_VERSION!r}",
    )
    request_id = payload.get("request_id")
    _require(isinstance(request_id, str) and bool(request_id), "response request_id is invalid")
    if expected_request_id is not None:
        _require(request_id == expected_request_id, "response request_id does not match request")
    _require(
        isinstance(payload.get("model_name"), str) and bool(payload["model_name"]),
        "response model_name must be a non-empty string",
    )
    actions_value = payload.get("actions")
    _require(isinstance(actions_value, list) and bool(actions_value), "response actions cannot be empty")
    _require(len(actions_value) <= 100, "response contains more than 100 actions")
    actions: List[List[float]] = []
    for index, action in enumerate(actions_value):
        converted = _finite_float_list(action, name=f"response.actions[{index}]", expected_length=ACTION_DIM)
        if require_normalized_actions:
            _require(
                all(-1.0 <= value <= 1.0 for value in converted),
                f"response.actions[{index}] must be in LIBERO's normalized [-1, 1] range",
            )
        actions.append(converted)
    inference_ms = payload.get("inference_ms")
    _require(_is_number(inference_ms), "response inference_ms must be a number")
    _require(math.isfinite(float(inference_ms)) and float(inference_ms) >= 0, "response inference_ms is invalid")
    return actions


def _read_limited(response: Any, max_bytes: int) -> bytes:
    data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ProtocolError(f"HTTP response is larger than {max_bytes} bytes")
    return data


class PolicyClient:
    """Small synchronous HTTP client used by the environment evaluator."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 30.0,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.max_response_bytes = int(max_response_bytes)

    def _request(self, path: str, *, payload: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        body: Optional[bytes] = None
        method = "GET"
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
            method = "POST"
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = _read_limited(response, self.max_response_bytes)
        except urllib.error.HTTPError as exc:
            raw = _read_limited(exc, self.max_response_bytes)
            try:
                detail = json.loads(raw.decode("utf-8")).get("error", raw.decode("utf-8"))
            except Exception:
                detail = raw.decode("utf-8", errors="replace")
            raise ProtocolError(f"policy server returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProtocolError(f"cannot reach policy server at {url}: {exc.reason}") from exc
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError("policy server did not return valid UTF-8 JSON") from exc
        _require(isinstance(decoded, dict), "policy server response must be a JSON object")
        return decoded

    def health(self) -> Dict[str, Any]:
        return self._request("/healthz")

    def metadata(self) -> Dict[str, Any]:
        return self._request("/v1/metadata")

    def infer(self, request_payload: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[List[float]], float]:
        validate_action_request(request_payload)
        start = time.perf_counter()
        response = self._request("/v1/actions", payload=request_payload)
        round_trip_ms = (time.perf_counter() - start) * 1000.0
        actions = validate_action_response(
            response,
            expected_request_id=str(request_payload["request_id"]),
            require_normalized_actions=True,
        )
        return response, actions, round_trip_ms
