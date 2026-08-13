"""A standalone mock action server for testing the LIBERO evaluator boundary."""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Sequence

from .protocol import (
    ACTION_DIM,
    PROTOCOL_VERSION,
    ProtocolError,
    make_action_response,
    validate_action_request,
)


DEFAULT_MAX_REQUEST_BYTES = 16 * 1024 * 1024


class MockPolicy:
    """Returns valid environment-ready actions without loading a model."""

    def __init__(
        self,
        *,
        mode: str = "noop",
        default_horizon: int = 5,
        random_scale: float = 0.05,
        seed: int = 7,
    ) -> None:
        if mode not in {"noop", "random"}:
            raise ValueError(f"unsupported mock mode: {mode}")
        if not 1 <= default_horizon <= 100:
            raise ValueError("default_horizon must be in [1, 100]")
        if not 0.0 <= random_scale <= 1.0:
            raise ValueError("random_scale must be in [0, 1]")
        self.mode = mode
        self.default_horizon = default_horizon
        self.random_scale = random_scale
        self._random = random.Random(seed)
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return f"mock/{self.mode}"

    def metadata(self) -> Dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "model_name": self.model_name,
            "server_type": "mock",
            "default_action_horizon": self.default_horizon,
            "observation_contract": {
                "images": "upright RGB uint8 HWC",
                "proprio": "raw LIBERO end-effector pose and gripper joints",
                "state_vector": "xyz + axis-angle + two gripper joints",
            },
            "action_contract": {
                "shape": ["T", ACTION_DIM],
                "range": [-1.0, 1.0],
                "layout": ["dx", "dy", "dz", "dRx", "dRy", "dRz", "gripper"],
                "ready_for_environment_step": True,
            },
        }

    def actions_for(self, request_payload: Dict[str, Any]) -> List[List[float]]:
        requested = request_payload["parameters"]["requested_action_horizon"]
        horizon = int(requested or self.default_horizon)
        if self.mode == "noop":
            return [[0.0] * (ACTION_DIM - 1) + [-1.0] for _ in range(horizon)]
        with self._lock:
            return [
                [self._random.uniform(-self.random_scale, self.random_scale) for _ in range(6)]
                + [-1.0]
                for _ in range(horizon)
            ]


class PolicyHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: Sequence[Any],
        policy: MockPolicy,
        *,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        quiet: bool = False,
    ) -> None:
        super().__init__(server_address, PolicyRequestHandler)
        self.policy = policy
        self.max_request_bytes = int(max_request_bytes)
        self.quiet = quiet


class PolicyRequestHandler(BaseHTTPRequestHandler):
    server: PolicyHTTPServer
    protocol_version = "HTTP/1.1"
    server_version = "LIBEROMockPolicy/1.0"

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)

    def _error(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})

    def _read_json_body(self) -> Dict[str, Any]:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise ProtocolError("Content-Type must be application/json")
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ProtocolError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ProtocolError("Content-Length must be an integer") from exc
        if length < 0 or length > self.server.max_request_bytes:
            raise ProtocolError(
                f"request body must be between 0 and {self.server.max_request_bytes} bytes"
            )
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolError("request body is not valid UTF-8 JSON") from exc
        if not isinstance(payload, dict):
            raise ProtocolError("request body must be a JSON object")
        return payload

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/healthz":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "protocol_version": PROTOCOL_VERSION,
                    "model_name": self.server.policy.model_name,
                },
            )
            return
        if self.path == "/v1/metadata":
            self._send_json(200, self.server.policy.metadata())
            return
        self._error(404, f"unknown endpoint: {self.path}")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/v1/actions":
            self._error(404, f"unknown endpoint: {self.path}")
            return
        try:
            request_payload = self._read_json_body()
            validate_action_request(request_payload)
            start = time.perf_counter()
            actions = self.server.policy.actions_for(request_payload)
            inference_ms = (time.perf_counter() - start) * 1000.0
            response = make_action_response(
                request_id=request_payload["request_id"],
                model_name=self.server.policy.model_name,
                actions=actions,
                inference_ms=inference_ms,
            )
        except ProtocolError as exc:
            self._error(400, str(exc))
            return
        except Exception as exc:  # Keep failures visible to the client without HTML.
            self._error(500, f"mock policy failed: {exc}")
            return
        self._send_json(200, response)

    def log_message(self, format_string: str, *args: Any) -> None:
        if not self.server.quiet:
            super().log_message(format_string, *args)


def build_server(
    *,
    host: str,
    port: int,
    policy: MockPolicy,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    quiet: bool = False,
) -> PolicyHTTPServer:
    return PolicyHTTPServer(
        (host, port),
        policy,
        max_request_bytes=max_request_bytes,
        quiet=quiet,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="bind address; use 0.0.0.0 for remote clients")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", choices=("noop", "random"), default="noop")
    parser.add_argument("--action-horizon", type=int, default=5)
    parser.add_argument("--random-scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-request-mib", type=float, default=16.0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    policy = MockPolicy(
        mode=args.mode,
        default_horizon=args.action_horizon,
        random_scale=args.random_scale,
        seed=args.seed,
    )
    server = build_server(
        host=args.host,
        port=args.port,
        policy=policy,
        max_request_bytes=int(args.max_request_mib * 1024 * 1024),
        quiet=args.quiet,
    )
    actual_host, actual_port = server.server_address[:2]
    print(
        f"mock policy ready: http://{actual_host}:{actual_port} "
        f"model={policy.model_name} horizon={policy.default_horizon}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping mock policy", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
