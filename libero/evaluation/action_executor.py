"""Fair, evaluator-owned action chunk execution."""

from collections import deque
from typing import Deque, Optional

import numpy as np

from .policy_client import PolicyClient
from .protocol import ActionSpec, PolicyRequest, RawObservation


class ActionChunkExecutor:
    """Consumes policy chunks using one shared replanning rule."""

    def __init__(
        self,
        client: PolicyClient,
        execute_horizon: int = 8,
        action_spec: Optional[ActionSpec] = None,
    ):
        if execute_horizon <= 0:
            raise ValueError("execute_horizon must be positive")
        self.client = client
        self.execute_horizon = execute_horizon
        self.action_spec = action_spec or ActionSpec()
        self._queue: Deque[np.ndarray] = deque()
        self._metadata_queue: Deque[dict] = deque()
        self._episode_id = ""
        self._instruction = ""
        self.last_action_metadata = {}
        self.query_count = 0
        self.round_trip_latency_ms = []
        self.server_inference_latency_ms = []

    def reset(self, episode_id: str, instruction: str) -> None:
        self._queue.clear()
        self._metadata_queue.clear()
        self._episode_id = episode_id
        self._instruction = instruction
        self.last_action_metadata = {}
        self.query_count = 0
        self.round_trip_latency_ms = []
        self.server_inference_latency_ms = []
        self.client.reset(episode_id, instruction)

    def act(self, obs, step: int) -> np.ndarray:
        if not self._episode_id:
            raise RuntimeError("reset must be called before act")
        if not self._queue:
            request = PolicyRequest(
                instruction=self._instruction,
                observation=RawObservation.from_libero(obs),
            )
            import time
            started = time.monotonic()
            response = self.client.infer(request).validate(self.action_spec)
            round_trip_ms = (time.monotonic() - started) * 1000.0
            self.round_trip_latency_ms.append(round_trip_ms)
            query_index = self.query_count
            self.query_count += 1
            latency = response.metadata.get("server_inference_latency_ms", response.metadata.get("inference_ms"))
            if latency is not None:
                self.server_inference_latency_ms.append(float(latency))
            base_metadata = dict(response.metadata)
            base_metadata.update(
                {
                    "policy_query_index": query_index,
                    "round_trip_latency_ms": round_trip_ms,
                }
            )
            for action_index, action in enumerate(
                response.actions[: self.execute_horizon]
            ):
                self._queue.append(action.copy())
                action_metadata = dict(base_metadata)
                action_metadata["chunk_action_index"] = action_index
                self._metadata_queue.append(action_metadata)
        self.last_action_metadata = self._metadata_queue.popleft()
        return self._queue.popleft()
