"""Client for OpenVLA's official json-numpy `/act` REST endpoint."""

from typing import Any, Optional

import numpy as np

from ..adapters import OpenVLAAdapter
from ..policy_client import ClientInfo, PolicyClient
from ..protocol import PolicyRequest, PolicyResponse
from .registry import register_client


class OpenVLAConnectionError(RuntimeError):
    """Raised when the official OpenVLA REST server cannot be reached."""


def _create_http_session():
    try:
        import json_numpy
        import requests
    except ImportError as exc:
        raise ImportError(
            "OpenVLAClient requires requests and json-numpy; install "
            "libero/evaluation/requirements.txt"
        ) from exc

    # This is the serialization mechanism used by official deploy.py for both
    # request NumPy arrays and its direct NumPy response.
    json_numpy.patch()
    return requests.Session()


@register_client("openvla")
class OpenVLAClient(PolicyClient):
    """Keep LIBERO adaptation local while leaving deploy.py unchanged."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 120.0,
        unnorm_key: str = "libero_10",
        *,
        adapter: Optional[OpenVLAAdapter] = None,
        session: Optional[Any] = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
            raise ValueError("openvla connection.base_url must be an HTTP(S) URL")
        if timeout_seconds <= 0:
            raise ValueError("openvla connection.timeout_seconds must be positive")
        if not isinstance(unnorm_key, str) or not unnorm_key.startswith("libero_"):
            raise ValueError(
                "openvla inference.unnorm_key must be a checkpoint key "
                "starting with 'libero_' (for example, 'libero_10')"
            )

        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.unnorm_key = unnorm_key
        self.adapter = adapter or OpenVLAAdapter()
        self._owns_session = session is None
        self._session = session or _create_http_session()

    @classmethod
    def from_config(cls, cfg):
        connection = cfg.get("connection", {})
        inference = cfg.get("inference", {})
        adapter = cfg.get("adapter", {})
        return cls(
            base_url=connection.get("base_url", ""),
            timeout_seconds=connection.get("timeout_seconds", 120.0),
            unnorm_key=inference.get("unnorm_key", "libero_10"),
            adapter=OpenVLAAdapter(
                image_preprocess=adapter.get(
                    "image_preprocess",
                    OpenVLAAdapter.PREPROCESS_OFFICIAL_LIBERO,
                ),
                center_crop=adapter.get("center_crop", False),
            ),
        )

    @staticmethod
    def _raise_for_status(response: Any) -> None:
        try:
            response.raise_for_status()
        except Exception as exc:
            raise OpenVLAConnectionError(
                "OpenVLA server returned an HTTP error: {}".format(exc)
            ) from exc

    def check(self) -> ClientInfo:
        # FastAPI exposes this endpoint automatically, so official deploy.py
        # needs no custom health route.
        try:
            response = self._session.get(
                self.base_url + "/openapi.json",
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            raise OpenVLAConnectionError(
                "cannot reach OpenVLA server at {}: {}".format(self.base_url, exc)
            ) from exc
        self._raise_for_status(response)
        document = response.json()
        paths = document.get("paths", {}) if isinstance(document, dict) else {}
        return ClientInfo(
            ready="/act" in paths,
            client_name="openvla",
            model_name="openvla",
            metadata={
                "base_url": self.base_url,
                "endpoint": "/act",
                "unnorm_key": self.unnorm_key,
                "image_preprocess": self.adapter.image_preprocess,
                "center_crop": self.adapter.center_crop,
            },
        )

    def reset(self, episode_id: str, instruction: str) -> None:
        # Base OpenVLA is stateless between requests.
        pass

    def infer(self, request: PolicyRequest) -> PolicyResponse:
        payload = self.adapter.adapt_observation(
            request.observation,
            request.instruction,
            self.unnorm_key,
        )
        try:
            response = self._session.post(
                self.base_url + "/act",
                json=payload,
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            raise OpenVLAConnectionError(
                "OpenVLA inference request failed: {}".format(exc)
            ) from exc
        self._raise_for_status(response)

        model_output = response.json()
        actions = self.adapter.actions_from_model_output(model_output)
        model_image = np.asarray(payload["image"])
        return PolicyResponse(
            actions=actions,
            metadata={
                # These values remain local.  They are recorded by the
                # evaluator's optional action trace and are never sent back to
                # the model server.
                "raw_model_action": np.asarray(
                    model_output, dtype=np.float32
                ).tolist(),
                "adapted_action_chunk": actions.tolist(),
                "unnorm_key": self.unnorm_key,
                "image_preprocess": self.adapter.image_preprocess,
                "center_crop": self.adapter.center_crop,
                "model_input_image_shape": list(model_image.shape),
                "model_input_image_dtype": str(model_image.dtype),
                "model_input_image_mean": float(model_image.mean()),
                "model_input_image_std": float(model_image.std()),
            },
        )

    def close(self) -> None:
        if self._owns_session:
            self._session.close()
