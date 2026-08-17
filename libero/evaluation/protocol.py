"""Internal, transport-neutral data passed between evaluator and clients."""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

import numpy as np


@dataclass(frozen=True)
class ActionSpec:
    """The action contract expected by the LIBERO environment."""

    type: str = "delta_ee"
    dim: int = 7
    controller: str = "OSC_POSE"
    control_frequency_hz: int = 20


@dataclass
class RawObservation:
    """Raw LIBERO values; model-specific transforms belong to PolicyClient adapters."""

    agentview_rgb: np.ndarray
    wrist_rgb: np.ndarray
    eef_pos: np.ndarray
    eef_quat: np.ndarray
    gripper_qpos: np.ndarray

    @classmethod
    def from_libero(cls, obs: Mapping[str, Any]) -> "RawObservation":
        return cls(
            agentview_rgb=np.asarray(obs["agentview_image"], dtype=np.uint8),
            wrist_rgb=np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8),
            eef_pos=np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            eef_quat=np.asarray(obs["robot0_eef_quat"], dtype=np.float32),
            gripper_qpos=np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )


@dataclass
class PolicyRequest:
    instruction: str
    observation: RawObservation


@dataclass
class PolicyResponse:
    actions: np.ndarray
    action_spec: ActionSpec = field(default_factory=ActionSpec)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self, expected: ActionSpec) -> "PolicyResponse":
        actions = np.asarray(self.actions)
        if actions.ndim != 2 or actions.shape[0] == 0:
            raise ValueError("policy actions must have non-empty shape [time, action_dim]")
        if actions.shape[1] != expected.dim:
            raise ValueError(
                "policy returned action_dim={}, expected {}".format(
                    actions.shape[1], expected.dim
                )
            )
        if self.action_spec.type != expected.type:
            raise ValueError(
                "policy returned action type {!r}, expected {!r}".format(
                    self.action_spec.type, expected.type
                )
            )
        if not np.isfinite(actions).all():
            raise ValueError("policy returned NaN or infinite actions")
        self.actions = actions
        return self
