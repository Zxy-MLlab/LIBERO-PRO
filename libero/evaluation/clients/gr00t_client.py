"""Adapter from the evaluator protocol to GR00T's ZeroMQ TCP policy server."""

import math
from typing import Any, Dict, Iterable, Optional

import numpy as np

from ..policy_client import ClientInfo, PolicyClient
from ..protocol import PolicyRequest, PolicyResponse
from .registry import register_client


class PolicyConnectionError(RuntimeError):
    pass


class PolicyTimeoutError(TimeoutError):
    pass


@register_client("gr00t")
class Gr00tClient(PolicyClient):
    """Client for ``gr00t/eval/run_gr00t_server.py --use-sim-policy-wrapper``.

    The GR00T server is a ZeroMQ REQ/REP endpoint, rather than a raw TCP
    socket.  Its wire format is msgpack-numpy and the sim wrapper expects the
    flat ``video.*``, ``state.*`` and language fields used below.
    """

    _DEFAULT_ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")

    def __init__(
        self,
        host: str,
        port: int,
        timeout_seconds: float = 30.0,
        api_token: Optional[str] = None,
        action_keys: Iterable[str] = _DEFAULT_ACTION_KEYS,
    ):
        if timeout_seconds <= 0:
            raise ValueError("connection.timeout_seconds must be positive")
        self.host, self.port = host, int(port)
        self.timeout_seconds = float(timeout_seconds)
        self.api_token = api_token
        self.action_keys = tuple(action_keys)
        if self.action_keys != self._DEFAULT_ACTION_KEYS:
            raise ValueError(
                "GR00T LIBERO action_keys must be "
                + repr(list(self._DEFAULT_ACTION_KEYS))
            )
        self._context = None
        self._socket = None
        self._metadata: Dict[str, Any] = {}

    @classmethod
    def from_config(cls, cfg):
        connection = cfg.get("connection", {})
        if not connection.get("host") or connection.get("port") is None:
            raise ValueError("gr00t connection.host and connection.port are required")
        inference = cfg.get("inference", {})
        return cls(
            connection["host"],
            connection["port"],
            connection.get("timeout_seconds", 30),
            connection.get("api_token"),
            inference.get("action_keys", cls._DEFAULT_ACTION_KEYS),
        )

    @staticmethod
    def _zmq():
        try:
            import zmq
            import msgpack_numpy as mnp
        except ImportError as exc:
            raise ImportError(
                "Gr00tClient requires pyzmq and msgpack-numpy; "
                "install libero/evaluation/requirements.txt"
            ) from exc
        return zmq, mnp

    def _connect(self):
        if self._socket is not None:
            return
        zmq, _ = self._zmq()
        try:
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.REQ)
            self._socket.setsockopt(zmq.LINGER, 0)
            timeout_ms = int(self.timeout_seconds * 1000)
            self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
            self._socket.connect("tcp://{}:{}".format(self.host, self.port))
        except Exception as exc:
            self.close()
            raise PolicyConnectionError("GR00T connection failed: {}".format(exc)) from exc

    def _reconnect(self):
        self.close()
        self._connect()

    def _call(
        self, endpoint: str, data: Optional[Dict[str, Any]] = None, *, requires_input=True
    ):
        self._connect()
        zmq, mnp = self._zmq()
        message: Dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            message["data"] = data or {}
        if self.api_token:
            message["api_token"] = self.api_token
        try:
            self._socket.send(mnp.packb(message))
            response = mnp.unpackb(self._socket.recv(), raw=False)
        except zmq.error.Again as exc:
            # A timed-out REQ socket cannot issue another request safely.
            self._reconnect()
            raise PolicyTimeoutError("GR00T {} timed out".format(endpoint)) from exc
        except Exception as exc:
            self.close()
            raise PolicyConnectionError("GR00T {} failed: {}".format(endpoint, exc)) from exc
        if isinstance(response, dict) and "error" in response:
            raise PolicyConnectionError("GR00T server error: {}".format(response["error"]))
        return response

    def check(self):
        self._call("ping", requires_input=False)
        modality_config = self._call("get_modality_config", requires_input=False)
        self._metadata = {"modality_config": modality_config}
        return ClientInfo(True, "gr00t", "GR00T", dict(self._metadata))

    def close(self):
        socket, context = self._socket, self._context
        self._socket, self._context = None, None
        if socket is not None:
            try:
                socket.close(linger=0)
            except Exception:
                pass
        if context is not None:
            try:
                context.term()
            except Exception:
                pass

    def reset(self, episode_id: str, instruction: str) -> None:
        # The stock GR00T policy is stateless, but reset is part of its server
        # protocol and remains important for wrappers with temporal state.
        self._call("reset", {"options": None})

    def infer(self, request: PolicyRequest) -> PolicyResponse:
        observation = self._request_adapter(request)
        result = self._call("get_action", {"observation": observation, "options": None})
        if not isinstance(result, (list, tuple)) or len(result) != 2:
            raise ValueError("GR00T response must be [action, info]")
        action, info = result
        actions = self._decode_actions(action)
        metadata = dict(info) if isinstance(info, dict) else {"server_info": info}
        return PolicyResponse(actions, metadata=metadata)

    def _request_adapter(self, request: PolicyRequest) -> Dict[str, Any]:
        obs = request.observation
        # Match GR00T's LIBERO wrapper: MuJoCo RGB images are vertically and
        # horizontally reversed, while state is eef xyz + axis-angle + gripper.
        image = np.ascontiguousarray(obs.agentview_rgb[::-1, ::-1], dtype=np.uint8)
        wrist_image = np.ascontiguousarray(obs.wrist_rgb[::-1, ::-1], dtype=np.uint8)
        state = np.concatenate((
            np.asarray(obs.eef_pos, dtype=np.float32),
            self._quat2axisangle(obs.eef_quat),
            np.asarray(obs.gripper_qpos, dtype=np.float32),
        ))
        return {
            "video.image": image[None, None],
            "video.wrist_image": wrist_image[None, None],
            "state.x": state[0:1][None, None],
            "state.y": state[1:2][None, None],
            "state.z": state[2:3][None, None],
            "state.roll": state[3:4][None, None],
            "state.pitch": state[4:5][None, None],
            "state.yaw": state[5:6][None, None],
            "state.gripper": state[6:][None, None],
            "annotation.human.action.task_description": (request.instruction,),
        }

    def _decode_actions(self, action: Any) -> np.ndarray:
        if not isinstance(action, dict):
            raise ValueError("GR00T action response must be a dictionary")
        columns = []
        horizon = None
        for key in self.action_keys:
            wire_key = "action." + key
            if wire_key not in action:
                raise ValueError("GR00T response does not contain {!r}".format(wire_key))
            value = np.asarray(action[wire_key], dtype=np.float32)
            if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != 1:
                raise ValueError(
                    "GR00T {!r} must have shape [1, T, 1], got {}".format(wire_key, value.shape)
                )
            if horizon is None:
                horizon = value.shape[1]
            elif value.shape[1] != horizon:
                raise ValueError("GR00T action components have inconsistent horizons")
            columns.append(value[0])
        actions = np.concatenate(columns, axis=1).astype(np.float32, copy=False)
        # ``LiberoEnv.step()`` in GR00T applies these two functions before
        # calling robosuite: normalize_gripper_action(), then
        # invert_gripper_action().  This evaluator calls robosuite directly,
        # so preserve the trained checkpoint's convention here: GR00T's
        # gripper 0=close / 1=open becomes LIBERO's +1=close / -1=open.
        actions[:, 6] = -np.sign(2.0 * actions[:, 6] - 1.0)
        return actions

    @staticmethod
    def _quat2axisangle(quat) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float64)
        w = float(np.clip(quat[3], -1.0, 1.0))
        den = math.sqrt(max(0.0, 1.0 - w * w))
        if math.isclose(den, 0.0):
            return np.zeros(3, dtype=np.float32)
        return (quat[:3] * (2.0 * math.acos(w) / den)).astype(np.float32)
