from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .perturbation_config import parse_bddl_perturbation_config


DEFAULT_CASES = (
    "clean",
    "visual_noise_glare",
    "camera_view_angle",
    "runtime_object_move",
    "object_texture",
    "view_occlusion",
    "object_shape",
    "initial_pose_position_angle",
)

CASE_DETAILS = {
    "clean": "original LIBERO task without perturbation",
    "visual_noise_glare": "BDDL-configured lighting and policy-observation noise",
    "camera_view_angle": "BDDL-configured MuJoCo camera pose",
    "runtime_object_move": "BDDL-configured target movement during rollout",
    "object_texture": "BDDL-configured object appearance",
    "view_occlusion": "BDDL-configured foreground occluder placement",
    "object_shape": "BDDL-configured object mesh scaling",
    "initial_pose_position_angle": "BDDL-configured initial object position and yaw",
}


@dataclass(frozen=True)
class CaseSpec:
    name: str
    suite: str
    task: str
    perturbation_layer: str
    perturbation_detail: str
    runtime_move: bool = False
    sim_lighting: str | None = None
    observation_perturbation: str | None = None
    robot_initial_qpos: str | None = None
    robot_initial_motion: str | None = None
    camera_pose: str | None = None
    perturbation_config: dict[str, Any] | None = None
    bddl_path: Path | None = None
    init_suite: str | None = None


@dataclass
class PerturbationRuntimeOptions:
    jitter_dx: int = 13
    jitter_dy: int = -9
    dark_brightness: float = 0.42
    noise_sigma: float = 24.0
    glare_strength: float = 0.78
    dark_light_diffuse: float = 0.22
    dark_light_ambient: float = 0.01
    dark_light_specular: float = 0.02
    glare_light_diffuse: float = 1.65
    glare_light_ambient: float = 0.18
    glare_light_specular: float = 1.8
    glare_secondary_diffuse: float = 0.35
    glare_secondary_specular: float = 0.15
    glare_light_x: float = 0.25
    glare_light_y: float = -0.25
    glare_light_z: float = 1.25
    backlight_diffuse: float = 1.85
    backlight_ambient: float = 0.035
    backlight_specular: float = 2.2
    backlight_secondary_diffuse: float = 0.16
    backlight_secondary_specular: float = 0.05
    backlight_x: float = -0.22
    backlight_y: float = 0.18
    backlight_z: float = 1.35
    backlight_attenuation_constant: float = 0.55
    backlight_attenuation_linear: float = 0.05
    backlight_attenuation_quadratic: float = 0.02
    camera_angle_name: str = "agentview"
    camera_angle_x: float = 0.48
    camera_angle_y: float = 0.24
    camera_angle_z: float = 1.62
    camera_angle_quat_w: float = 0.6099232
    camera_angle_quat_x: float = 0.3594558
    camera_angle_quat_y: float = 0.3594558
    camera_angle_quat_z: float = 0.6099232
    object_name: str = "akita_black_bowl_1"
    move_step: int = 80
    move_dx: float = 0.10
    move_dy: float = 0.04
    runtime_move_trigger: str = "step"
    runtime_move_distance_threshold: float = 0.09
    runtime_move_fallback_step: int = 160
    min_object_gap: float = 0.01
    default_object_radius: float = 0.04
    scene_contact_tolerance: float = -1e-3
    min_safe_move_distance: float = 0.06
    max_safe_move_attempts: int = 32
    safe_z_overlap_tolerance: float = 0.04
    safe_x_min: float = -0.6
    safe_x_max: float = 0.6
    safe_y_min: float = -0.6
    safe_y_max: float = 0.6

def section(text: str, name: str) -> str:
    start = text.find(f"(:{name}")
    if start < 0:
        return ""
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "(":
            depth += 1
        elif text[idx] == ")":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return text[start:]

def compact(text: str) -> str:
    return " ".join(text.split())

def parse_bddl_summary(bddl_path: Path) -> dict[str, Any]:
    text = bddl_path.read_text(encoding="utf-8")
    perturbation_config = parse_bddl_perturbation_config(bddl_path)
    language = re.sub(
        r"^\(:language\s*|\)$", "", section(text, "language").strip(), flags=re.S
    ).strip()
    objects: dict[str, str] = {}
    for line in section(text, "objects").splitlines():
        line = line.strip()
        if " - " not in line:
            continue
        lhs, rhs = line.split(" - ", 1)
        for obj in lhs.split():
            objects[obj] = rhs.strip()
    init_on = re.findall(r"\(On\s+(\S+)\s+(\S+)\)", section(text, "init"), re.I)
    yaw_blocks = re.findall(
        r"\(:yaw_rotation\s*\(\s*\(([^)]+)\)\s*\)\s*\)", text, flags=re.S
    )
    return {
        "bddl": str(bddl_path),
        "language": language,
        "objects": objects,
        "init_on": init_on,
        "goal": compact(section(text, "goal")),
        "yaw_rotations": [compact(yaw) for yaw in yaw_blocks],
        "perturbation_config": perturbation_config,
    }

def config_section(config: dict[str, Any] | None, name: str) -> dict[str, Any]:
    value = (config or {}).get(name)
    return value if isinstance(value, dict) else {}

def config_section_enabled(config: dict[str, Any]) -> bool:
    if not config:
        return False
    if not config_bool(config, "enabled", True):
        return False
    mode = str(config_value(config, "mode", "")).lower()
    return mode not in {"none", "default", "identity", "disabled", "off"}

def active_config_section(config: dict[str, Any] | None, name: str) -> dict[str, Any]:
    section = config_section(config, name)
    return section if config_section_enabled(section) else {}

def config_value(config: dict[str, Any], key: str, default: Any) -> Any:
    return config[key] if key in config else default

def config_float(config: dict[str, Any], key: str, default: float) -> float:
    return float(config_value(config, key, default))

def config_int(config: dict[str, Any], key: str, default: int) -> int:
    return int(config_value(config, key, default))

def config_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    value = config_value(config, key, default)
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)

def config_list(
    config: dict[str, Any],
    key: str,
    default: list[float] | tuple[float, ...],
    expected_len: int | None = None,
) -> list[float]:
    value = config_value(config, key, default)
    if not isinstance(value, list):
        raise ValueError(f"BDDL perturbation config field {key!r} must be a list.")
    out = [float(item) for item in value]
    if expected_len is not None and len(out) != expected_len:
        raise ValueError(
            f"BDDL perturbation config field {key!r} must have {expected_len} values; got {len(out)}"
        )
    return out

def config_name_list(
    config: dict[str, Any],
    key: str,
    default: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    value = config_value(config, key, list(default or []))
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return list(default or [])
    return [str(value)]

def normalize_quat(raw_quat: list[float]) -> np.ndarray:
    quat = np.array([float(value) for value in raw_quat], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        raise ValueError(f"Invalid zero quaternion: {raw_quat}")
    return quat / norm

def config_nested_float_list(value: Any, expected_len: int, field_name: str) -> list[list[float]]:
    if value is None or value == []:
        return []
    if not isinstance(value, list):
        raise ValueError(f"BDDL perturbation config field {field_name!r} must be a list.")
    if value and not isinstance(value[0], list):
        value = [value]
    out = []
    for item in value:
        if not isinstance(item, list) or len(item) != expected_len:
            raise ValueError(
                f"BDDL perturbation config field {field_name!r} entries must have {expected_len} values."
            )
        out.append([float(part) for part in item])
    return out

def config_array(
    config: dict[str, Any],
    key: str,
    default: float | list[float] | tuple[float, ...],
) -> float | list[float]:
    value = config_value(config, key, default)
    if isinstance(value, list):
        return [float(item) for item in value]
    return float(value)

def infer_spec_from_config(
    case: str,
    suite: str,
    task: str,
    config: dict[str, Any],
    bddl_path: Path | None = None,
    init_suite: str | None = None,
) -> CaseSpec:
    lighting_cfg = active_config_section(config, "lighting")
    observation_cfg = active_config_section(config, "observation")
    runtime_cfg = active_config_section(config, "runtime_move")
    robot_cfg = active_config_section(config, "robot_initial")
    camera_cfg = active_config_section(config, "camera")
    object_appearance_cfg = active_config_section(config, "object_appearance")
    object_shape_cfg = active_config_section(config, "object_shape")
    object_placement_cfg = active_config_section(config, "object_placement")
    initial_pose_cfg = active_config_section(config, "initial_pose")

    perturbation_layer = str(config.get("layer") or "bddl_and_init_state")
    if lighting_cfg:
        perturbation_layer = "bddl_config_sim_mujoco_lighting"
    elif observation_cfg:
        perturbation_layer = "bddl_config_policy_observation_pixels"
    elif runtime_cfg:
        perturbation_layer = "bddl_config_sim_runtime_qpos"
    elif robot_cfg:
        perturbation_layer = "bddl_config_sim_robot_initial_state"
    elif camera_cfg:
        perturbation_layer = "bddl_config_sim_mujoco_camera_pose"
    elif object_appearance_cfg:
        perturbation_layer = "bddl_config_sim_object_appearance"
    elif object_shape_cfg:
        perturbation_layer = "bddl_config_sim_object_shape"
    elif object_placement_cfg:
        perturbation_layer = "bddl_config_sim_object_placement"
    elif initial_pose_cfg:
        perturbation_layer = "bddl_config_sim_initial_pose"

    robot_mode = str(robot_cfg.get("mode", "")) if robot_cfg else ""
    return CaseSpec(
        name=case,
        suite=suite,
        task=task,
        perturbation_layer=perturbation_layer,
        perturbation_detail=str(config.get("detail") or CASE_DETAILS.get(case, "load BDDL perturbation config")),
        runtime_move=bool(runtime_cfg),
        sim_lighting=str(lighting_cfg.get("mode", case)) if lighting_cfg else None,
        observation_perturbation=str(observation_cfg.get("mode", case)) if observation_cfg else None,
        robot_initial_qpos=robot_mode if robot_mode == "low" else None,
        robot_initial_motion=robot_mode if robot_mode == "tabletop_natural" else None,
        camera_pose=str(camera_cfg.get("mode", case)) if camera_cfg else None,
        perturbation_config=config,
        bddl_path=bddl_path,
        init_suite=init_suite or suite,
    )

def image_batch_keys(observation: dict[str, Any]) -> list[str]:
    if "pixels" not in observation:
        return []
    return [key for key, value in observation["pixels"].items() if isinstance(value, np.ndarray)]

def add_dark_noise_rgb(image: np.ndarray, brightness: float, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dark = image.astype(np.float32) * brightness
    noise = rng.normal(0.0, sigma, image.shape)
    return np.clip(dark + noise, 0, 255).astype(np.uint8)

def add_glare_rgb(image: np.ndarray, strength: float) -> np.ndarray:
    h, w = image.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    cx, cy = int(w * 0.34), int(h * 0.31)
    radius = min(h, w) * 0.24
    mask = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * radius**2)))
    flare = np.zeros_like(image, dtype=np.float32)
    flare[..., 0] = 255
    flare[..., 1] = 235
    flare[..., 2] = 205
    mixed = image.astype(np.float32) * (1 - strength * mask[..., None]) + flare * (
        strength * mask[..., None]
    )
    cv2.circle(mixed, (cx, cy), int(radius * 0.34), (255, 255, 255), -1, cv2.LINE_AA)
    return np.clip(mixed, 0, 255).astype(np.uint8)

def jitter_rgb(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    transform = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        image,
        transform,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

def copy_observation(observation: dict[str, Any]) -> dict[str, Any]:
    out = dict(observation)
    if "pixels" in observation:
        out["pixels"] = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in observation["pixels"].items()
        }
    return out

def apply_case_to_observation(
    observation: dict[str, Any],
    spec: CaseSpec,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if spec.observation_perturbation is None:
        return observation
    obs_cfg = active_config_section(spec.perturbation_config, "observation")
    mode = spec.observation_perturbation
    out = copy_observation(observation)
    for idx, key in enumerate(image_batch_keys(out)):
        batch = out["pixels"][key]
        for b in range(batch.shape[0]):
            image = batch[b]
            if mode == "dark_noise":
                brightness = config_float(obs_cfg, "brightness", args.dark_brightness)
                sigma = config_float(obs_cfg, "sigma", args.noise_sigma)
                image = add_dark_noise_rgb(image, brightness, sigma, seed + idx + b)
            elif mode == "glare":
                strength = config_float(obs_cfg, "strength", args.glare_strength)
                image = add_glare_rgb(image, strength)
            elif mode == "camera_jitter":
                dx = config_int(obs_cfg, "dx", args.jitter_dx)
                dy = config_int(obs_cfg, "dy", args.jitter_dy)
                image = jitter_rgb(image, dx, dy)
            else:
                raise ValueError(f"Unsupported observation perturbation: {mode}")
            batch[b] = image
    return out

def robot_contact_pairs(inner) -> list[dict[str, Any]]:
    contacts = []
    model = inner.sim.model
    data = inner.sim.data
    for idx in range(data.ncon):
        contact = data.contact[idx]
        geom_a = model.geom_id2name(contact.geom1) or ""
        geom_b = model.geom_id2name(contact.geom2) or ""
        if (
            "robot0" in geom_a
            or "robot0" in geom_b
            or "gripper" in geom_a
            or "gripper" in geom_b
        ):
            contacts.append(
                {
                    "geom_a": geom_a,
                    "geom_b": geom_b,
                    "distance": float(contact.dist),
                }
            )
    return contacts

def capture_robot_state(inner) -> dict[str, Any]:
    inner._update_observables(force=True)
    raw_obs = inner._get_observations()
    robot = inner.robots[0]
    return {
        "joint_pos": np.asarray(raw_obs["robot0_joint_pos"]).tolist(),
        "eef_pos": np.asarray(raw_obs["robot0_eef_pos"]).tolist(),
        "eef_quat": np.asarray(raw_obs["robot0_eef_quat"]).tolist(),
        "gripper_qpos": np.asarray(raw_obs["robot0_gripper_qpos"]).tolist(),
        "sim_qpos_arm": np.asarray(inner.sim.data.qpos[robot._ref_joint_pos_indexes]).tolist(),
        "robot_contacts": robot_contact_pairs(inner),
    }

def capture_lighting_state(inner) -> dict[str, Any]:
    model = inner.sim.model
    state = {
        "nlight": int(model.nlight),
    }
    for name in [
        "light_active",
        "light_ambient",
        "light_diffuse",
        "light_specular",
        "light_pos",
        "light_dir",
        "light_attenuation",
        "light_castshadow",
    ]:
        if hasattr(model, name):
            value = getattr(model, name)
            state[name] = np.asarray(value).tolist()
    return state

def apply_sim_lighting(vec_env, spec: CaseSpec, args: argparse.Namespace) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if spec.sim_lighting is None:
        return None, None
    env = vec_env.envs[0]
    if getattr(env, "_env", None) is None:
        raise RuntimeError("Underlying LIBERO env is not initialized.")
    inner = env._env.env
    model = inner.sim.model
    before = capture_lighting_state(inner)

    if model.nlight <= 0:
        raise RuntimeError("Cannot apply lighting perturbation: MuJoCo model has no lights.")

    lighting_cfg = active_config_section(spec.perturbation_config, "lighting")
    if spec.sim_lighting == "dark_lighting":
        model.light_active[:] = True
        model.light_ambient[:] = config_array(lighting_cfg, "ambient", args.dark_light_ambient)
        model.light_diffuse[:] = config_array(lighting_cfg, "diffuse", args.dark_light_diffuse)
        model.light_specular[:] = config_array(lighting_cfg, "specular", args.dark_light_specular)
        if hasattr(model, "light_intensity"):
            model.light_intensity[:] = 0.0
    elif spec.sim_lighting == "glare_lighting":
        model.light_active[:] = True
        model.light_ambient[:] = config_array(lighting_cfg, "ambient", args.glare_light_ambient)
        model.light_diffuse[:] = config_array(lighting_cfg, "diffuse", args.glare_light_diffuse)
        model.light_specular[:] = config_array(lighting_cfg, "specular", args.glare_light_specular)
        model.light_pos[0] = np.array(
            config_list(
                lighting_cfg,
                "pos",
                [args.glare_light_x, args.glare_light_y, args.glare_light_z],
                expected_len=3,
            ),
            dtype=model.light_pos.dtype,
        )
        if model.nlight > 1:
            model.light_diffuse[1:] = config_array(
                lighting_cfg,
                "secondary_diffuse",
                args.glare_secondary_diffuse,
            )
            model.light_specular[1:] = config_array(
                lighting_cfg,
                "secondary_specular",
                args.glare_secondary_specular,
            )
        if hasattr(model, "light_intensity"):
            model.light_intensity[:] = 0.0
    elif spec.sim_lighting == "backlight_lighting":
        model.light_active[:] = True
        model.light_ambient[:] = config_array(lighting_cfg, "ambient", args.backlight_ambient)
        model.light_diffuse[:] = config_array(
            lighting_cfg,
            "secondary_diffuse",
            args.backlight_secondary_diffuse,
        )
        model.light_specular[:] = config_array(
            lighting_cfg,
            "secondary_specular",
            args.backlight_secondary_specular,
        )
        model.light_pos[0] = np.array(
            config_list(
                lighting_cfg,
                "pos",
                [args.backlight_x, args.backlight_y, args.backlight_z],
                expected_len=3,
            ),
            dtype=model.light_pos.dtype,
        )
        model.light_diffuse[0] = config_array(lighting_cfg, "diffuse", args.backlight_diffuse)
        model.light_specular[0] = config_array(lighting_cfg, "specular", args.backlight_specular)
        if hasattr(model, "light_attenuation"):
            model.light_attenuation[0] = np.array(
                config_list(
                    lighting_cfg,
                    "attenuation",
                    [
                        args.backlight_attenuation_constant,
                        args.backlight_attenuation_linear,
                        args.backlight_attenuation_quadratic,
                    ],
                    expected_len=3,
                ),
                dtype=model.light_attenuation.dtype,
            )
        if hasattr(model, "light_intensity"):
            model.light_intensity[:] = 0.0
    else:
        raise ValueError(f"Unsupported sim lighting case: {spec.sim_lighting}")

    inner.sim.forward()
    env._env._post_process()
    env._env._update_observables(force=True)
    after = capture_lighting_state(inner)
    return before, after

def capture_camera_state(inner, camera_name: str) -> dict[str, Any]:
    cam_id = inner.sim.model.camera_name2id(camera_name)
    return {
        "camera_name": camera_name,
        "camera_id": int(cam_id),
        "pos": np.asarray(inner.sim.model.cam_pos[cam_id]).tolist(),
        "quat": np.asarray(inner.sim.model.cam_quat[cam_id]).tolist(),
    }

def apply_camera_pose(vec_env, spec: CaseSpec, args: argparse.Namespace) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if spec.camera_pose is None:
        return None, None
    env = vec_env.envs[0]
    if getattr(env, "_env", None) is None:
        raise RuntimeError("Underlying LIBERO env is not initialized.")
    inner = env._env.env
    camera_cfg = active_config_section(spec.perturbation_config, "camera")
    camera_name = str(config_value(camera_cfg, "name", args.camera_angle_name))
    before = capture_camera_state(inner, camera_name)
    cam_id = inner.sim.model.camera_name2id(camera_name)
    if spec.camera_pose == "agentview_angle":
        inner.sim.model.cam_pos[cam_id] = np.array(
            config_list(
                camera_cfg,
                "pos",
                [args.camera_angle_x, args.camera_angle_y, args.camera_angle_z],
                expected_len=3,
            ),
            dtype=inner.sim.model.cam_pos.dtype,
        )
        inner.sim.model.cam_quat[cam_id] = np.array(
            config_list(
                camera_cfg,
                "quat",
                [
                    args.camera_angle_quat_w,
                    args.camera_angle_quat_x,
                    args.camera_angle_quat_y,
                    args.camera_angle_quat_z,
                ],
                expected_len=4,
            ),
            dtype=inner.sim.model.cam_quat.dtype,
        )
    else:
        raise ValueError(f"Unsupported camera pose perturbation: {spec.camera_pose}")
    inner.sim.forward()
    env._env._post_process()
    env._env._update_observables(force=True)
    after = capture_camera_state(inner, camera_name)
    return before, after

def get_inner_env(vec_env):
    env = vec_env.envs[0]
    if getattr(env, "_env", None) is None:
        raise RuntimeError("Underlying LIBERO env is not initialized.")
    return env, env._env.env

def scene_entity_names(inner) -> list[str]:
    return sorted(inner.obj_body_id)

def require_scene_entity(inner, object_name: str, context: str) -> None:
    if object_name not in inner.obj_body_id:
        raise KeyError(f"{context} {object_name!r} is not in this scene. Available: {scene_entity_names(inner)}")

def scene_entity_pose(inner, object_name: str) -> np.ndarray:
    require_scene_entity(inner, object_name, "scene entity")
    if object_name in inner.objects_dict:
        obj = inner.objects_dict[object_name]
        return np.array(inner.sim.data.get_joint_qpos(obj.joints[-1])).copy()
    body_id = int(inner.obj_body_id[object_name])
    return np.concatenate(
        [
            np.array(inner.sim.model.body_pos[body_id], dtype=np.float64).copy(),
            np.array(inner.sim.model.body_quat[body_id], dtype=np.float64).copy(),
        ]
    )

def set_scene_entity_pose(inner, object_name: str, pose: np.ndarray) -> None:
    require_scene_entity(inner, object_name, "scene entity")
    if object_name in inner.objects_dict:
        obj = inner.objects_dict[object_name]
        inner.sim.data.set_joint_qpos(obj.joints[-1], pose)
        return
    body_id = int(inner.obj_body_id[object_name])
    inner.sim.model.body_pos[body_id] = np.asarray(pose[:3], dtype=inner.sim.model.body_pos.dtype)
    inner.sim.model.body_quat[body_id] = np.asarray(pose[3:7], dtype=inner.sim.model.body_quat.dtype)

def object_body_name(inner, object_name: str) -> str:
    if object_name not in inner.obj_body_id:
        raise KeyError(f"{object_name} is not in this scene. Available bodies: {sorted(inner.obj_body_id)}")
    body_id = inner.obj_body_id[object_name]
    body_name = inner.sim.model.body_id2name(body_id)
    return body_name or object_name

def descendant_body_ids(inner, root_body_id: int) -> set[int]:
    body_ids = {int(root_body_id)}
    changed = True
    while changed:
        changed = False
        for body_id in range(inner.sim.model.nbody):
            if body_id in body_ids:
                continue
            parent_id = int(inner.sim.model.body_parentid[body_id])
            if parent_id in body_ids:
                body_ids.add(body_id)
                changed = True
    return body_ids

def is_visual_geom(inner, geom_id: int) -> bool:
    if not hasattr(inner.sim.model, "geom_group"):
        return True
    group = int(inner.sim.model.geom_group[geom_id])
    return group == 1

def body_geom_ids(inner, object_name: str) -> list[int]:
    root_body_id = int(inner.obj_body_id[object_name])
    body_ids = descendant_body_ids(inner, root_body_id)
    body_prefix = object_body_name(inner, object_name)
    geom_ids = []
    for geom_id in range(inner.sim.model.ngeom):
        geom_name = inner.sim.model.geom_id2name(geom_id) or ""
        body_id = int(inner.sim.model.geom_bodyid[geom_id])
        body_name = inner.sim.model.body_id2name(body_id) or ""
        if (
            body_id in body_ids
            or body_name.startswith(body_prefix)
            or geom_name.startswith(body_prefix)
            or body_name.startswith(object_name)
            or geom_name.startswith(object_name)
        ):
            geom_ids.append(geom_id)
    if not geom_ids:
        raise RuntimeError(f"No geoms found for object {object_name!r}.")
    return geom_ids

def body_mesh_ids(inner, object_name: str) -> list[int]:
    mesh_ids = []
    for geom_id in body_geom_ids(inner, object_name):
        mesh_id = int(inner.sim.model.geom_dataid[geom_id])
        if mesh_id >= 0 and mesh_id not in mesh_ids:
            mesh_ids.append(mesh_id)
    return mesh_ids

def body_geom_names(inner, object_name: str) -> set[str]:
    return {
        inner.sim.model.geom_id2name(geom_id) or str(geom_id)
        for geom_id in body_geom_ids(inner, object_name)
    }

def material_name(model, mat_id: int) -> str:
    if hasattr(model, "mat_id2name"):
        name = model.mat_id2name(mat_id)
        if name:
            return name
    return str(mat_id)

def capture_object_visual_state(inner, object_names: list[str]) -> dict[str, Any]:
    state = {}
    for object_name in object_names:
        geom_ids = body_geom_ids(inner, object_name)
        mesh_ids = body_mesh_ids(inner, object_name)
        material_ids = sorted(
            {
                int(inner.sim.model.geom_matid[geom_id])
                for geom_id in geom_ids
                if hasattr(inner.sim.model, "geom_matid")
                and int(inner.sim.model.geom_matid[geom_id]) >= 0
            }
        )
        state[object_name] = {
            "xy": object_xy(inner, object_name).tolist(),
            "quat": np.asarray(
                inner.sim.data.body_xquat[inner.obj_body_id[object_name]]
            ).tolist(),
            "geom_rgba": {
                inner.sim.model.geom_id2name(geom_id) or str(geom_id): np.asarray(
                    inner.sim.model.geom_rgba[geom_id]
                ).tolist()
                for geom_id in geom_ids
            },
            "material_rgba": {
                material_name(inner.sim.model, mat_id): np.asarray(
                    inner.sim.model.mat_rgba[mat_id]
                ).tolist()
                for mat_id in material_ids
                if hasattr(inner.sim.model, "mat_rgba")
            },
            "material_texid": {
                material_name(inner.sim.model, mat_id): int(
                    np.asarray(inner.sim.model.mat_texid[mat_id]).reshape(-1)[0]
                )
                for mat_id in material_ids
                if hasattr(inner.sim.model, "mat_texid")
            },
            "mesh_scale": {
                inner.sim.model.mesh_id2name(mesh_id) or str(mesh_id): np.asarray(
                    inner.sim.model.mesh_scale[mesh_id]
                ).tolist()
                for mesh_id in mesh_ids
            },
        }
    return state

def apply_object_appearance(vec_env, spec: CaseSpec, args: argparse.Namespace) -> dict[str, Any] | None:
    cfg = active_config_section(spec.perturbation_config, "object_appearance")
    if not cfg:
        return None
    env, inner = get_inner_env(vec_env)
    targets = config_name_list(cfg, "targets") or config_name_list(cfg, "target")
    if not targets:
        raise ValueError("object_appearance config needs (:target ...) or (:targets ...).")
    rgba = np.array(
        config_list(cfg, "rgba", [0.58, 0.45, 0.31, 1.0], expected_len=4),
        dtype=inner.sim.model.geom_rgba.dtype,
    )
    override_material_rgba = config_bool(cfg, "override_material_rgba", True)
    disable_material_texture = config_bool(cfg, "disable_material_texture", True)
    apply_visible_geom_rgba = config_bool(cfg, "apply_visible_geom_rgba", False)
    before = capture_object_visual_state(inner, targets)
    changed_geoms: list[str] = []
    changed_materials: list[str] = []
    disabled_material_textures: list[str] = []
    changed_material_ids: set[int] = set()
    for object_name in targets:
        for geom_id in body_geom_ids(inner, object_name):
            if apply_visible_geom_rgba or int(inner.sim.model.geom_group[geom_id]) == 1:
                if not is_visual_geom(inner, geom_id):
                    continue
                inner.sim.model.geom_rgba[geom_id] = rgba
                changed_geoms.append(inner.sim.model.geom_id2name(geom_id) or str(geom_id))
            if not hasattr(inner.sim.model, "geom_matid"):
                continue
            mat_id = int(inner.sim.model.geom_matid[geom_id])
            if mat_id < 0 or mat_id in changed_material_ids:
                continue
            mat_name = material_name(inner.sim.model, mat_id)
            if override_material_rgba and hasattr(inner.sim.model, "mat_rgba"):
                inner.sim.model.mat_rgba[mat_id] = rgba
                changed_materials.append(mat_name)
            if disable_material_texture and hasattr(inner.sim.model, "mat_texid"):
                try:
                    inner.sim.model.mat_texid[mat_id] = -1
                    disabled_material_textures.append(mat_name)
                except (TypeError, ValueError, IndexError):
                    pass
            changed_material_ids.add(mat_id)
    inner.sim.forward()
    env._env._post_process()
    env._env._update_observables(force=True)
    after = capture_object_visual_state(inner, targets)
    return {
        "mode": str(config_value(cfg, "mode", "rgba")),
        "targets": targets,
        "rgba": rgba.tolist(),
        "override_material_rgba": override_material_rgba,
        "disable_material_texture": disable_material_texture,
        "apply_visible_geom_rgba": apply_visible_geom_rgba,
        "changed_geoms": changed_geoms,
        "changed_materials": changed_materials,
        "disabled_material_textures": disabled_material_textures,
        "before": before,
        "after": after,
    }

def apply_object_shape(vec_env, spec: CaseSpec, args: argparse.Namespace) -> dict[str, Any] | None:
    cfg = active_config_section(spec.perturbation_config, "object_shape")
    if not cfg:
        return None
    env, inner = get_inner_env(vec_env)
    targets = config_name_list(cfg, "targets") or config_name_list(cfg, "target")
    if not targets:
        raise ValueError("object_shape config needs (:target ...) or (:targets ...).")
    scale = np.array(
        config_list(cfg, "mesh_scale", [1.18, 1.18, 1.18], expected_len=3),
        dtype=inner.sim.model.mesh_scale.dtype,
    )
    before = capture_object_visual_state(inner, targets)
    changed_meshes: list[str] = []
    for object_name in targets:
        for mesh_id in body_mesh_ids(inner, object_name):
            inner.sim.model.mesh_scale[mesh_id] *= scale
            changed_meshes.append(inner.sim.model.mesh_id2name(mesh_id) or str(mesh_id))
    inner.sim.forward()
    env._env._post_process()
    env._env._update_observables(force=True)
    after = capture_object_visual_state(inner, targets)
    return {
        "mode": str(config_value(cfg, "mode", "mesh_scale")),
        "targets": targets,
        "mesh_scale_multiplier": scale.tolist(),
        "changed_meshes": changed_meshes,
        "before": before,
        "after": after,
    }

def apply_object_placement(vec_env, spec: CaseSpec, args: argparse.Namespace) -> dict[str, Any] | None:
    cfg = active_config_section(spec.perturbation_config, "object_placement")
    if not cfg:
        return None
    env, inner = get_inner_env(vec_env)
    targets = config_name_list(cfg, "targets") or config_name_list(cfg, "target")
    if not targets:
        raise ValueError("object_placement config needs (:target ...) or (:targets ...).")
    base_object = str(config_value(cfg, "reference", targets[0]))
    require_scene_entity(inner, base_object, "object_placement reference")
    base_xy = object_xy(inner, base_object)
    offsets = config_value(cfg, "offsets_xy", [])
    if not isinstance(offsets, list):
        raise ValueError("object_placement (:offsets_xy ...) must be a list.")
    if offsets and not isinstance(offsets[0], list):
        offsets = [offsets]
    if not offsets:
        offsets = [[0.0, -0.14]]
    raw_quats = config_value(
        cfg,
        "target_quats",
        config_value(cfg, "target_quat", []),
    )
    target_quats = config_nested_float_list(raw_quats, 4, "target_quats")
    raw_z_offsets = config_value(
        cfg,
        "z_offsets",
        config_value(cfg, "z_offset", []),
    )
    if raw_z_offsets == [] or raw_z_offsets is None:
        z_offsets: list[float] = []
    elif isinstance(raw_z_offsets, list):
        z_offsets = [float(value) for value in raw_z_offsets]
    else:
        z_offsets = [float(raw_z_offsets)]
    avoid_targets = (
        config_name_list(cfg, "avoid_targets")
        or config_name_list(cfg, "avoid_target")
        or config_name_list(cfg, "forbidden_objects")
    )
    min_avoid_distance = float(config_value(cfg, "min_avoid_distance", 0.0))
    min_reference_distance = config_value(cfg, "min_reference_distance", None)
    if min_reference_distance is not None:
        min_reference_distance = float(min_reference_distance)
    mesh_scale = config_value(cfg, "mesh_scale", [])
    mesh_scale_multiplier = None
    if mesh_scale:
        mesh_scale_multiplier = np.array(
            config_list(cfg, "mesh_scale", [1.0, 1.0, 1.0], expected_len=3),
            dtype=inner.sim.model.mesh_scale.dtype,
        )
        for object_name in targets:
            if object_name not in inner.objects_dict:
                raise KeyError(f"object_placement target {object_name!r} is not in this scene.")
            for mesh_id in body_mesh_ids(inner, object_name):
                inner.sim.model.mesh_scale[mesh_id] *= mesh_scale_multiplier
        inner.sim.forward()
        env._env._post_process()
        env._env._update_observables(force=True)
    pin_targets = config_bool(cfg, "pin_targets", False)
    front_direction = np.array(
        config_list(cfg, "front_direction", [1.0, 0.0], expected_len=2),
        dtype=np.float64,
    )
    before = scene_object_states(inner, args)
    moved_geom_names: set[str] = set()
    separation_info = None
    between_object = str(config_value(cfg, "between_object", ""))
    if between_object:
        if between_object not in inner.objects_dict:
            raise KeyError(f"object_placement between_object {between_object!r} is not in this scene.")
        min_between_distance = float(config_value(cfg, "min_between_distance", 0.0))
        base_xy = object_xy(inner, base_object)
        between_xy = object_xy(inner, between_object)
        current_distance = float(np.linalg.norm(between_xy - base_xy))
        if current_distance < min_between_distance:
            between_delta = tuple(
                config_list(cfg, "between_delta_xy", [0.14, 0.0], expected_len=2)
            )
            moved_geom_names.update(body_geom_names(inner, between_object))
            obj = inner.objects_dict[between_object]
            qpos = np.array(inner.sim.data.get_joint_qpos(obj.joints[-1])).copy()
            original = np.array(qpos[:2]).copy()
            safe_delta, rejected_candidates = choose_safe_delta(
                inner,
                between_object,
                (float(between_delta[0]), float(between_delta[1])),
                args,
            )
            qpos[0] += safe_delta[0]
            qpos[1] += safe_delta[1]
            inner.sim.data.set_joint_qpos(obj.joints[-1], qpos)
            inner.sim.forward()
            env._env._post_process()
            env._env._update_observables(force=True)
            separation_info = {
                "object": between_object,
                "reference": base_object,
                "min_between_distance": min_between_distance,
                "before_distance_xy": current_distance,
                "requested_delta_xy": [float(between_delta[0]), float(between_delta[1])],
                "applied_delta_xy": safe_delta.tolist(),
                "before_xy": original.tolist(),
                "after_xy": object_xy(inner, between_object).tolist(),
                "after_distance_xy": float(np.linalg.norm(object_xy(inner, between_object) - base_xy)),
                "rejected_candidate_count": len(rejected_candidates),
                "rejected_candidates": rejected_candidates[:5],
            }
        else:
            separation_info = {
                "object": between_object,
                "reference": base_object,
                "min_between_distance": min_between_distance,
                "before_distance_xy": current_distance,
                "skipped": True,
            }
        base_xy = object_xy(inner, base_object)
    moves = []
    pinned_placements = []
    for idx, object_name in enumerate(targets):
        if object_name not in inner.objects_dict:
            raise KeyError(f"object_placement target {object_name!r} is not in this scene.")
        moved_geom_names.update(body_geom_names(inner, object_name))
        offset = np.array([float(v) for v in offsets[min(idx, len(offsets) - 1)]], dtype=np.float64)
        obj = inner.objects_dict[object_name]
        qpos = np.array(inner.sim.data.get_joint_qpos(obj.joints[-1])).copy()
        target_quat = (
            normalize_quat(target_quats[min(idx, len(target_quats) - 1)])
            if target_quats
            else None
        )
        target_z_offset = float(z_offsets[min(idx, len(z_offsets) - 1)]) if z_offsets else 0.0

        def candidate_qpos_for_offset(candidate_xy: np.ndarray) -> np.ndarray:
            candidate = qpos.copy()
            candidate[0] = float(candidate_xy[0])
            candidate[1] = float(candidate_xy[1])
            if z_offsets:
                candidate[2] += target_z_offset
            if target_quat is not None:
                candidate[3:7] = target_quat
            return candidate

        target_xy, rejected_candidates = choose_safe_reference_offset(
            inner,
            object_name,
            base_object,
            base_xy,
            offset,
            args,
            front_direction=front_direction,
            avoid_targets=avoid_targets,
            min_avoid_distance=min_avoid_distance,
            min_reference_distance_override=min_reference_distance,
            candidate_qpos_builder=candidate_qpos_for_offset,
        )
        original = np.array(qpos[:2]).copy()
        qpos[0] = target_xy[0]
        qpos[1] = target_xy[1]
        if z_offsets:
            qpos[2] += target_z_offset
        if target_quat is not None:
            qpos[3:7] = target_quat
        inner.sim.data.set_joint_qpos(obj.joints[-1], qpos)
        inner.sim.forward()
        env._env._post_process()
        env._env._update_observables(force=True)
        moves.append(
            {
                "object": object_name,
                "reference": base_object,
                "requested_offset_xy": offset.tolist(),
                "applied_offset_xy": (target_xy - base_xy).tolist(),
                "before_xy": original.tolist(),
                "after_xy": target_xy.tolist(),
                "target_quat": qpos[3:7].tolist() if len(qpos) >= 7 else None,
                "z_offset": float(z_offsets[min(idx, len(z_offsets) - 1)]) if z_offsets else 0.0,
                "rejected_candidate_count": len(rejected_candidates),
                "rejected_candidates": rejected_candidates[:5],
            }
        )
        if pin_targets:
            pinned_placements.append(
                {
                    "object": object_name,
                    "joint": obj.joints[-1],
                    "qpos": qpos.tolist(),
                }
            )
    inner.sim.forward()
    env._env._post_process()
    env._env._update_observables(force=True)
    post_move_violations = scene_overlap_violations(inner, args)
    if post_move_violations:
        raise RuntimeError(f"Unsafe object_placement left overlaps: {post_move_violations}")
    post_move_contact_violations = [
        violation
        for violation in scene_contact_violations(inner, args)
        if violation["geom_a"] in moved_geom_names or violation["geom_b"] in moved_geom_names
    ]
    if post_move_contact_violations:
        raise RuntimeError(f"Unsafe object_placement left contacts: {post_move_contact_violations[:20]}")
    after = scene_object_states(inner, args)
    return {
        "mode": str(config_value(cfg, "mode", "front_occlusion")),
        "reference": base_object,
        "front_direction": front_direction.tolist(),
        "avoid_targets": avoid_targets,
        "min_avoid_distance": min_avoid_distance,
        "min_reference_distance": min_reference_distance,
        "mesh_scale_multiplier": mesh_scale_multiplier.tolist() if mesh_scale_multiplier is not None else None,
        "pin_targets": pin_targets,
        "pinned_placements": pinned_placements,
        "separation": separation_info,
        "moves": moves,
        "before": before,
        "after": after,
    }

def apply_initial_pose(vec_env, spec: CaseSpec, args: argparse.Namespace) -> dict[str, Any] | None:
    cfg = active_config_section(spec.perturbation_config, "initial_pose")
    if not cfg:
        return None
    env, inner = get_inner_env(vec_env)
    target = str(config_value(cfg, "target", ""))
    if not target:
        raise ValueError("initial_pose config needs (:target ...).")
    require_scene_entity(inner, target, "initial_pose target")
    before = capture_object_visual_state(inner, [target])
    qpos = scene_entity_pose(inner, target)
    requested_delta_xy = None
    applied_delta_xy = None
    rejected_candidates: list[dict[str, Any]] = []
    yaw = None
    if "yaw" in cfg:
        from robosuite.utils import transform_utils as T

        yaw = float(config_value(cfg, "yaw", 0.7853981633974483))
        qpos[3:7] = np.array(T.convert_quat(T.axisangle2quat([0, 0, yaw]), to="wxyz"))
    if "delta_xy" in cfg:
        delta_xy = tuple(config_list(cfg, "delta_xy", [0.06, 0.0], expected_len=2))
        requested_delta_xy = [float(delta_xy[0]), float(delta_xy[1])]
        min_delta_norm = config_value(cfg, "min_delta_norm", None)
        min_delta_norm = None if min_delta_norm is None else float(min_delta_norm)

        def candidate_qpos_for_delta(delta: np.ndarray) -> np.ndarray:
            candidate = qpos.copy()
            candidate[0] = float(qpos[0] + delta[0])
            candidate[1] = float(qpos[1] + delta[1])
            return candidate

        safe_delta, rejected_candidates = choose_safe_delta(
            inner,
            target,
            delta_xy,
            args,
            candidate_qpos_builder=candidate_qpos_for_delta,
            min_delta_norm=min_delta_norm,
        )
        applied_delta_xy = safe_delta.tolist()
        qpos[0] += safe_delta[0]
        qpos[1] += safe_delta[1]
    else:
        min_delta_norm = None
    set_scene_entity_pose(inner, target, qpos)
    inner.sim.forward()
    env._env._post_process()
    env._env._update_observables(force=True)
    post_move_violations = scene_overlap_violations(inner, args)
    if post_move_violations:
        raise RuntimeError(f"Unsafe initial_pose left overlaps: {post_move_violations}")
    moved_geom_names = body_geom_names(inner, target)
    post_move_contact_violations = [
        violation
        for violation in scene_contact_violations(inner, args)
        if violation["geom_a"] in moved_geom_names or violation["geom_b"] in moved_geom_names
    ]
    if post_move_contact_violations:
        raise RuntimeError(f"Unsafe initial_pose left contacts: {post_move_contact_violations[:20]}")
    after = capture_object_visual_state(inner, [target])
    return {
        "mode": str(config_value(cfg, "mode", "delta_xy_yaw")),
        "target": target,
        "requested_delta_xy": requested_delta_xy,
        "applied_delta_xy": applied_delta_xy,
        "requested_min_delta_norm": min_delta_norm,
        "rejected_candidate_count": len(rejected_candidates),
        "rejected_candidates": rejected_candidates[:5],
        "yaw": yaw,
        "before": before,
        "after": after,
    }

def apply_static_bddl_config_perturbations(
    vec_env,
    spec: CaseSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "object_appearance": apply_object_appearance(vec_env, spec, args),
        "object_shape": apply_object_shape(vec_env, spec, args),
        "object_placement": apply_object_placement(vec_env, spec, args),
        "initial_pose": apply_initial_pose(vec_env, spec, args),
    }

def apply_pinned_object_placements(vec_env, static_config_info: dict[str, Any]) -> bool:
    placement_info = (static_config_info or {}).get("object_placement")
    if not placement_info:
        return False
    pinned_placements = placement_info.get("pinned_placements") or []
    if not pinned_placements:
        return False
    env, inner = get_inner_env(vec_env)
    for item in pinned_placements:
        joint = item.get("joint")
        qpos = item.get("qpos")
        if not joint or qpos is None:
            continue
        inner.sim.data.set_joint_qpos(str(joint), np.array(qpos, dtype=np.float64))
    inner.sim.forward()
    env._env._post_process()
    env._env._update_observables(force=True)
    return True

def object_horizontal_radius(inner, object_name: str, default: float) -> float:
    obj = inner.objects_dict.get(object_name)
    if obj is None:
        return default
    try:
        radius = float(np.asarray(obj.horizontal_radius).reshape(-1)[0])
    except (AttributeError, TypeError, ValueError, IndexError):
        return default
    if radius <= 0.0:
        return default
    return radius

def object_xy(inner, object_name: str) -> np.ndarray:
    return np.array(inner.sim.data.body_xpos[inner.obj_body_id[object_name]][:2], dtype=np.float64)

def object_xyz(inner, object_name: str) -> np.ndarray:
    return np.array(inner.sim.data.body_xpos[inner.obj_body_id[object_name]][:3], dtype=np.float64)

def same_support_layer(z_a: float, z_b: float, args: argparse.Namespace) -> bool:
    return abs(float(z_a) - float(z_b)) <= args.safe_z_overlap_tolerance

def candidate_within_bounds(candidate_xy: np.ndarray, args: argparse.Namespace) -> bool:
    return (
        args.safe_x_min <= candidate_xy[0] <= args.safe_x_max
        and args.safe_y_min <= candidate_xy[1] <= args.safe_y_max
    )

def overlap_violations(
    inner,
    object_name: str,
    candidate_xy: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    moving_radius = object_horizontal_radius(inner, object_name, args.default_object_radius)
    moving_z = float(object_xyz(inner, object_name)[2])
    violations = []
    for other_name in sorted(inner.objects_dict):
        if other_name == object_name:
            continue
        other_xyz = object_xyz(inner, other_name)
        if not same_support_layer(moving_z, float(other_xyz[2]), args):
            continue
        other_xy = other_xyz[:2]
        other_radius = object_horizontal_radius(inner, other_name, args.default_object_radius)
        distance = float(np.linalg.norm(candidate_xy - other_xy))
        min_distance = moving_radius + other_radius + args.min_object_gap
        if distance < min_distance:
            violations.append(
                {
                    "other_object": other_name,
                    "distance_xy": distance,
                    "required_min_distance_xy": min_distance,
                    "other_xy": other_xy.tolist(),
                }
            )
    return violations

def scene_object_states(inner, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    states = {}
    for name in sorted(inner.objects_dict):
        xyz = object_xyz(inner, name)
        states[name] = {
            "xy": xyz[:2].tolist(),
            "z": float(xyz[2]),
            "horizontal_radius": object_horizontal_radius(inner, name, args.default_object_radius),
        }
    return states

def allowed_task_support_pairs(inner) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for section in ("initial_state", "goal_state"):
        for predicate in inner.parsed_problem.get(section, []):
            if not isinstance(predicate, list) or len(predicate) < 3:
                continue
            relation = str(predicate[0]).lower()
            if relation not in {"on", "in"}:
                continue
            first = str(predicate[1])
            second = str(predicate[2])
            if first in inner.objects_dict and second in inner.objects_dict:
                pairs.add(tuple(sorted((first, second))))
    return pairs

def scene_overlap_violations(inner, args: argparse.Namespace) -> list[dict[str, Any]]:
    states = scene_object_states(inner, args)
    allowed_pairs = allowed_task_support_pairs(inner)
    violations = []
    names = sorted(states)
    for idx, name in enumerate(names):
        xy = np.array(states[name]["xy"], dtype=np.float64)
        radius = float(states[name]["horizontal_radius"])
        for other_name in names[idx + 1 :]:
            if tuple(sorted((name, other_name))) in allowed_pairs:
                continue
            other_xy = np.array(states[other_name]["xy"], dtype=np.float64)
            other_radius = float(states[other_name]["horizontal_radius"])
            z_gap = abs(float(states[name]["z"]) - float(states[other_name]["z"]))
            if z_gap > args.safe_z_overlap_tolerance:
                continue
            distance = float(np.linalg.norm(xy - other_xy))
            min_distance = radius + other_radius + args.min_object_gap
            if distance < min_distance:
                violations.append(
                    {
                        "object_a": name,
                        "object_b": other_name,
                        "distance_xy": distance,
                        "required_min_distance_xy": min_distance,
                        "object_a_xy": xy.tolist(),
                        "object_b_xy": other_xy.tolist(),
                        "object_a_z": float(states[name]["z"]),
                        "object_b_z": float(states[other_name]["z"]),
                        "z_gap": z_gap,
                        "safe_z_overlap_tolerance": args.safe_z_overlap_tolerance,
                        "object_a_radius": radius,
                        "object_b_radius": other_radius,
                    }
                )
    return violations

def scene_bounds_violations(
    object_states: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    violations = []
    for name, state in object_states.items():
        x, y = state["xy"]
        if not (
            args.safe_x_min <= x <= args.safe_x_max
            and args.safe_y_min <= y <= args.safe_y_max
        ):
            violations.append(
                {
                    "object": name,
                    "xy": [x, y],
                    "safe_x_range": [args.safe_x_min, args.safe_x_max],
                    "safe_y_range": [args.safe_y_min, args.safe_y_max],
                }
            )
    return violations

def scene_contact_violations(inner, args: argparse.Namespace) -> list[dict[str, Any]]:
    violations = []
    model = inner.sim.model
    data = inner.sim.data
    for idx in range(data.ncon):
        contact = data.contact[idx]
        if float(contact.dist) >= args.scene_contact_tolerance:
            continue
        geom_a = model.geom_id2name(contact.geom1) or ""
        geom_b = model.geom_id2name(contact.geom2) or ""
        pair = f"{geom_a} {geom_b}".lower()
        if not geom_a or not geom_b:
            continue
        if "table_collision" in pair or "floor" in pair:
            continue
        violations.append(
            {
                "geom_a": geom_a,
                "geom_b": geom_b,
                "distance": float(contact.dist),
            }
        )
    return violations

def candidate_scene_violations(
    inner,
    object_name: str,
    candidate_xy: np.ndarray,
    args: argparse.Namespace,
    candidate_qpos: np.ndarray | None = None,
) -> dict[str, Any]:
    original_qpos = scene_entity_pose(inner, object_name)
    original_geom_names = body_geom_names(inner, object_name)
    try:
        qpos = np.array(candidate_qpos).copy() if candidate_qpos is not None else original_qpos.copy()
        qpos[0] = float(candidate_xy[0])
        qpos[1] = float(candidate_xy[1])
        set_scene_entity_pose(inner, object_name, qpos)
        inner.sim.forward()
        overlap_violations_after = [
            violation
            for violation in scene_overlap_violations(inner, args)
            if object_name in {violation["object_a"], violation["object_b"]}
        ]
        contact_violations_after = [
            violation
            for violation in scene_contact_violations(inner, args)
            if violation["geom_a"] in original_geom_names or violation["geom_b"] in original_geom_names
        ]
        return {
            "overlap_violations": overlap_violations_after,
            "contact_violations": contact_violations_after,
        }
    finally:
        set_scene_entity_pose(inner, object_name, original_qpos)
        inner.sim.forward()

def validate_scene_no_overlaps(vec_env, args: argparse.Namespace, context: str) -> dict[str, Any]:
    env = vec_env.envs[0]
    if getattr(env, "_env", None) is None:
        raise RuntimeError("Underlying LIBERO env is not initialized.")
    inner = env._env.env
    object_states = scene_object_states(inner, args)
    violations = scene_overlap_violations(inner, args)
    bounds_violations = scene_bounds_violations(object_states, args)
    contact_violations = scene_contact_violations(inner, args)
    if violations:
        raise RuntimeError(
            f"Unsafe object overlap detected during {context}: "
            f"{json.dumps(violations, indent=2)}"
        )
    if bounds_violations:
        raise RuntimeError(
            f"Unsafe object position detected during {context}: "
            f"{json.dumps(bounds_violations, indent=2)}"
        )
    return {
        "object_states": object_states,
        "min_object_gap": args.min_object_gap,
        "safe_xy_bounds": {
            "x": [args.safe_x_min, args.safe_x_max],
            "y": [args.safe_y_min, args.safe_y_max],
        },
        "object_overlap_violations": violations,
        "object_bounds_violations": bounds_violations,
        "object_contact_violations": contact_violations,
    }

def candidate_deltas(
    requested_delta: tuple[float, float],
    args: argparse.Namespace,
    min_delta_norm: float | None = None,
) -> list[np.ndarray]:
    requested = np.array(requested_delta, dtype=np.float64)
    norm = float(np.linalg.norm(requested))
    min_required = float(min_delta_norm or 0.0)
    if norm < 1e-6:
        norm = max(args.min_safe_move_distance, min_required)
        base_angle = 0.0
    else:
        base_angle = float(np.arctan2(requested[1], requested[0]))

    base_radius = max(norm, min_required)
    candidates = [
        np.array([base_radius * np.cos(base_angle), base_radius * np.sin(base_angle)])
    ]
    radii = [
        base_radius,
        max(base_radius * 0.90, min_required, args.min_safe_move_distance),
        base_radius * 1.10,
        base_radius * 1.25,
    ]
    angle_offsets = [0, np.pi / 2, -np.pi / 2, np.pi, np.pi / 4, -np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4]
    for radius in radii:
        for offset in angle_offsets:
            angle = base_angle + offset
            candidates.append(np.array([radius * np.cos(angle), radius * np.sin(angle)]))
    # Stable fallback directions for tabletop tasks. When a minimum runtime
    # displacement is requested, keep these fallbacks at that minimum distance.
    fallback_deltas = [
        np.array([-0.08, 0.08]),
        np.array([-0.10, 0.06]),
        np.array([-0.06, 0.10]),
        np.array([0.00, 0.10]),
        np.array([-0.10, 0.00]),
    ]
    if min_required > 0.0:
        scaled_fallbacks = []
        fallback_radius = max(min_required, args.min_safe_move_distance)
        for delta in fallback_deltas:
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > 1e-6:
                scaled_fallbacks.append(delta / delta_norm * fallback_radius)
        fallback_deltas = scaled_fallbacks
    candidates.extend(fallback_deltas)
    deduped = []
    seen = set()
    for delta in candidates:
        key = tuple(np.round(delta, 4))
        if key not in seen:
            deduped.append(delta)
            seen.add(key)
    return deduped[: args.max_safe_move_attempts]

def candidate_reference_offsets(
    requested_offset: np.ndarray,
    args: argparse.Namespace,
    front_direction: np.ndarray | None = None,
) -> list[np.ndarray]:
    norm = float(np.linalg.norm(requested_offset))
    has_front_direction = front_direction is not None and float(np.linalg.norm(front_direction)) > 1e-6
    if norm < args.min_safe_move_distance:
        norm = max(args.min_safe_move_distance, 0.10)
        base_angle = (
            float(np.arctan2(front_direction[1], front_direction[0]))
            if has_front_direction
            else -np.pi / 2
        )
    else:
        base_angle = float(np.arctan2(requested_offset[1], requested_offset[0]))
    if has_front_direction:
        front_angle = float(np.arctan2(front_direction[1], front_direction[0]))
        requested_angle = float(np.arctan2(requested_offset[1], requested_offset[0]))
        angle_delta = float(
            np.arctan2(np.sin(requested_angle - front_angle), np.cos(requested_angle - front_angle))
        )
        if abs(angle_delta) > 1.2:
            base_angle = front_angle
    radii = [
        max(norm, 0.12),
        max(norm + 0.04, 0.16),
        max(norm + 0.08, 0.20),
    ]
    angle_offsets = (
        [0.0, -0.16, 0.16, -0.32, 0.32, -0.48, 0.48]
        if has_front_direction
        else [0.0, -0.20, 0.20, -0.40, 0.40, -0.65, 0.65, np.pi / 2, -np.pi / 2]
    )
    candidates = [requested_offset]
    for radius in radii:
        for offset in angle_offsets:
            angle = base_angle + offset
            candidates.append(np.array([radius * np.cos(angle), radius * np.sin(angle)]))
    if has_front_direction:
        unit = np.array(front_direction, dtype=np.float64)
        unit = unit / float(np.linalg.norm(unit))
        side = np.array([-unit[1], unit[0]])
        for forward in [0.075, 0.095, 0.12, 0.16, 0.20, 0.24]:
            for lateral in [0.0, -0.065, 0.065, -0.10, 0.10]:
                candidates.append(unit * forward + side * lateral)
    else:
        candidates.extend(
            [
                requested_offset + np.array([0.06, 0.00]),
                requested_offset + np.array([-0.06, 0.00]),
                requested_offset + np.array([0.00, 0.06]),
                requested_offset + np.array([0.00, -0.06]),
                np.array([0.18, -0.18]),
                np.array([-0.18, -0.18]),
                np.array([0.22, 0.00]),
                np.array([-0.22, 0.00]),
                np.array([0.00, 0.22]),
                np.array([0.00, -0.22]),
                np.array([0.24, 0.14]),
                np.array([-0.24, 0.14]),
                np.array([0.24, -0.14]),
                np.array([-0.24, -0.14]),
            ]
        )
    deduped = []
    seen = set()
    for candidate in candidates:
        key = tuple(np.round(candidate, 4))
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped[: args.max_safe_move_attempts]

def choose_safe_reference_offset(
    inner,
    object_name: str,
    reference_name: str,
    reference_xy: np.ndarray,
    requested_offset: np.ndarray,
    args: argparse.Namespace,
    front_direction: np.ndarray | None = None,
    avoid_targets: list[str] | None = None,
    min_avoid_distance: float = 0.0,
    min_reference_distance_override: float | None = None,
    candidate_qpos_builder=None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rejected = []
    moving_radius = object_horizontal_radius(inner, object_name, args.default_object_radius)
    reference_radius = object_horizontal_radius(inner, reference_name, args.default_object_radius)
    if min_reference_distance_override is None:
        min_reference_distance = moving_radius + reference_radius + args.min_object_gap
    else:
        min_reference_distance = float(min_reference_distance_override)
    avoid_targets = list(avoid_targets or [])
    for offset in candidate_reference_offsets(requested_offset, args, front_direction=front_direction):
        candidate_xy = reference_xy + offset
        distance_to_reference = float(np.linalg.norm(candidate_xy - reference_xy))
        if not candidate_within_bounds(candidate_xy, args):
            rejected.append(
                {
                    "candidate_offset_xy": offset.tolist(),
                    "candidate_xy": candidate_xy.tolist(),
                    "reason": "outside_safe_table_bounds",
                }
            )
            continue
        if distance_to_reference < min_reference_distance:
            rejected.append(
                {
                    "candidate_offset_xy": offset.tolist(),
                    "candidate_xy": candidate_xy.tolist(),
                    "reason": "would_press_on_reference_object",
                    "distance_xy": distance_to_reference,
                    "required_min_distance_xy": min_reference_distance,
                }
            )
            continue
        avoid_violations = []
        for avoid_name in avoid_targets:
            if avoid_name == object_name:
                continue
            if avoid_name not in inner.objects_dict:
                continue
            avoid_xy = object_xy(inner, avoid_name)
            avoid_radius = object_horizontal_radius(inner, avoid_name, args.default_object_radius)
            distance_to_avoid = float(np.linalg.norm(candidate_xy - avoid_xy))
            required_avoid_distance = moving_radius + avoid_radius + float(min_avoid_distance)
            if distance_to_avoid < required_avoid_distance:
                avoid_violations.append(
                    {
                        "avoid_target": avoid_name,
                        "distance_xy": distance_to_avoid,
                        "required_min_distance_xy": required_avoid_distance,
                    }
                )
        if avoid_violations:
            rejected.append(
                {
                    "candidate_offset_xy": offset.tolist(),
                    "candidate_xy": candidate_xy.tolist(),
                    "reason": "too_close_to_avoid_target",
                    "avoid_violations": avoid_violations,
                }
            )
            continue
        violations = overlap_violations(inner, object_name, candidate_xy, args)
        violations = [item for item in violations if item["other_object"] != object_name]
        if violations:
            rejected.append(
                {
                    "candidate_offset_xy": offset.tolist(),
                    "candidate_xy": candidate_xy.tolist(),
                    "reason": "would_overlap_existing_object",
                    "violations": violations,
                }
            )
            continue
        candidate_qpos = candidate_qpos_builder(candidate_xy) if candidate_qpos_builder else None
        scene_violations = candidate_scene_violations(
            inner,
            object_name,
            candidate_xy,
            args,
            candidate_qpos=candidate_qpos,
        )
        if scene_violations["overlap_violations"] or scene_violations["contact_violations"]:
            rejected.append(
                {
                    "candidate_offset_xy": offset.tolist(),
                    "candidate_xy": candidate_xy.tolist(),
                    "reason": "would_create_scene_violation",
                    "overlap_violations": scene_violations["overlap_violations"][:10],
                    "contact_violations": scene_violations["contact_violations"][:10],
                }
            )
            continue
        return candidate_xy, rejected
    raise RuntimeError(
        f"No safe placement candidate found for {object_name} near {reference_name}. "
        f"Rejected {len(rejected)} candidates; first rejection: {rejected[:1]}"
    )

def choose_safe_delta(
    inner,
    object_name: str,
    requested_delta: tuple[float, float],
    args: argparse.Namespace,
    candidate_qpos_builder=None,
    min_delta_norm: float | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    current_xy = object_xy(inner, object_name)
    rejected = []
    for delta in candidate_deltas(requested_delta, args, min_delta_norm=min_delta_norm):
        delta_norm = float(np.linalg.norm(delta))
        if min_delta_norm is not None and delta_norm + 1e-6 < float(min_delta_norm):
            rejected.append(
                {
                    "candidate_delta_xy": delta.tolist(),
                    "reason": "below_min_delta_norm",
                    "delta_norm": delta_norm,
                    "min_delta_norm": float(min_delta_norm),
                }
            )
            continue
        candidate_xy = current_xy + delta
        if not candidate_within_bounds(candidate_xy, args):
            rejected.append(
                {
                    "candidate_delta_xy": delta.tolist(),
                    "candidate_xy": candidate_xy.tolist(),
                    "reason": "outside_safe_table_bounds",
                }
            )
            continue
        violations = overlap_violations(inner, object_name, candidate_xy, args)
        if violations:
            rejected.append(
                {
                    "candidate_delta_xy": delta.tolist(),
                    "candidate_xy": candidate_xy.tolist(),
                    "reason": "would_overlap_existing_object",
                    "violations": violations,
                }
            )
            continue
        candidate_qpos = candidate_qpos_builder(delta) if candidate_qpos_builder is not None else None
        scene_violations = candidate_scene_violations(
            inner,
            object_name,
            candidate_xy,
            args,
            candidate_qpos=candidate_qpos,
        )
        if scene_violations["overlap_violations"] or scene_violations["contact_violations"]:
            rejected.append(
                {
                    "candidate_delta_xy": delta.tolist(),
                    "candidate_xy": candidate_xy.tolist(),
                    "reason": "would_create_scene_violation",
                    "overlap_violations": scene_violations["overlap_violations"][:10],
                    "contact_violations": scene_violations["contact_violations"][:10],
                }
            )
            continue
        return delta, rejected
    raise RuntimeError(
        f"No safe runtime move candidate found for {object_name}. "
        f"Rejected {len(rejected)} candidates; first rejection: {rejected[:1]}"
    )

def move_object_xy(
    vec_env,
    object_name: str,
    delta_xy: tuple[float, float],
    args: argparse.Namespace,
    min_delta_norm: float | None = None,
    trigger_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    env = vec_env.envs[0]
    if getattr(env, "_env", None) is None:
        raise RuntimeError("Underlying LIBERO env is not initialized.")
    inner = env._env.env
    require_scene_entity(inner, object_name, "runtime move target")
    moving_geom_names = body_geom_names(inner, object_name)
    before = np.array(inner.sim.data.body_xpos[inner.obj_body_id[object_name]]).copy()
    safe_delta, rejected_candidates = choose_safe_delta(
        inner,
        object_name,
        delta_xy,
        args,
        min_delta_norm=min_delta_norm,
    )
    qpos = scene_entity_pose(inner, object_name)
    qpos[0] += safe_delta[0]
    qpos[1] += safe_delta[1]
    set_scene_entity_pose(inner, object_name, qpos)
    inner.sim.forward()
    after = np.array(inner.sim.data.body_xpos[inner.obj_body_id[object_name]]).copy()
    env._env._post_process()
    env._env._update_observables(force=True)
    post_move_violations = scene_overlap_violations(inner, args)
    if post_move_violations:
        raise RuntimeError(f"Unsafe runtime move left overlaps: {post_move_violations}")
    post_move_contact_violations = [
        violation
        for violation in scene_contact_violations(inner, args)
        if violation["geom_a"] in moving_geom_names or violation["geom_b"] in moving_geom_names
    ]
    if post_move_contact_violations:
        raise RuntimeError(f"Unsafe runtime move left contacts: {post_move_contact_violations[:20]}")
    info = {
        "object_name": object_name,
        "before_position": before.tolist(),
        "after_position": after.tolist(),
        "requested_delta_xy": [delta_xy[0], delta_xy[1]],
        "applied_delta_xy": safe_delta.tolist(),
        "requested_min_delta_norm": min_delta_norm,
        "actual_delta_xyz": (after - before).tolist(),
        "rejected_candidate_count": len(rejected_candidates),
        "rejected_candidates": rejected_candidates[:5],
        "min_object_gap": args.min_object_gap,
    }
    if trigger_info is not None:
        info["trigger"] = trigger_info
    return info

def runtime_move_settings(spec: CaseSpec, args: argparse.Namespace) -> tuple[int, str, tuple[float, float], float | None]:
    runtime_cfg = active_config_section(spec.perturbation_config, "runtime_move")
    step = config_int(runtime_cfg, "step", args.move_step)
    object_name = str(config_value(runtime_cfg, "object", args.object_name))
    delta_xy = config_list(runtime_cfg, "delta_xy", [args.move_dx, args.move_dy], expected_len=2)
    min_delta_norm = config_value(runtime_cfg, "min_delta_norm", None)
    return (
        step,
        object_name,
        (float(delta_xy[0]), float(delta_xy[1])),
        None if min_delta_norm is None else float(min_delta_norm),
    )

def runtime_move_trigger_settings(spec: CaseSpec, args: argparse.Namespace) -> dict[str, Any]:
    runtime_cfg = active_config_section(spec.perturbation_config, "runtime_move")
    legacy_step = config_int(runtime_cfg, "step", getattr(args, "move_step", 80))
    fallback_default = getattr(args, "runtime_move_fallback_step", legacy_step)
    fallback_step = config_value(runtime_cfg, "fallback_step", fallback_default)
    distance_default = getattr(args, "runtime_move_distance_threshold", 0.09)
    return {
        "trigger": str(config_value(runtime_cfg, "trigger", getattr(args, "runtime_move_trigger", "step"))).lower(),
        "configured_step": legacy_step,
        "fallback_step": None if fallback_step in (None, "", "none", "None") else int(fallback_step),
        "distance_threshold": config_float(runtime_cfg, "distance_threshold", distance_default),
        "distance_metric": str(config_value(runtime_cfg, "distance_metric", "xyz")).lower(),
    }


def runtime_move_trigger_decision(
    step: int,
    eef_pos: np.ndarray | list[float] | tuple[float, ...],
    object_pos: np.ndarray | list[float] | tuple[float, ...],
    settings: dict[str, Any],
) -> dict[str, Any] | None:
    """Evaluate a runtime-move trigger without depending on a simulator instance."""
    trigger = str(settings["trigger"])
    if trigger in {"step", "fixed_step"}:
        if step == int(settings["configured_step"]):
            return {"mode": "step", "reason": "fixed_step", "step": step, **settings}
        return None
    if trigger not in {"near_grasp", "near_object", "distance"}:
        raise ValueError(f"Unsupported runtime move trigger: {trigger}")

    eef = np.asarray(eef_pos, dtype=np.float64)
    target = np.asarray(object_pos, dtype=np.float64)
    metric = str(settings["distance_metric"])
    if metric == "xy":
        distance = float(np.linalg.norm(eef[:2] - target[:2]))
    elif metric == "xyz":
        distance = float(np.linalg.norm(eef - target))
    else:
        raise ValueError(f"Unsupported runtime move distance metric: {metric}")

    common = {
        "mode": trigger,
        "step": step,
        "eef_pos": eef.tolist(),
        "object_pos": target.tolist(),
        "distance": distance,
        **settings,
    }
    if distance <= float(settings["distance_threshold"]):
        return {"reason": "distance_threshold", **common}
    fallback_step = settings["fallback_step"]
    if fallback_step is not None and int(fallback_step) > 0 and step >= int(fallback_step):
        return {"reason": "fallback_step", **common}
    return None


def runtime_move_trigger_status(
    vec_env,
    object_name: str,
    step: int,
    spec: CaseSpec,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    settings = runtime_move_trigger_settings(spec, args)
    if settings["trigger"] in {"step", "fixed_step"}:
        return runtime_move_trigger_decision(step, [], [], settings)

    env = vec_env.envs[0]
    if getattr(env, "_env", None) is None:
        raise RuntimeError("Underlying LIBERO env is not initialized.")
    inner = env._env.env
    require_scene_entity(inner, object_name, "runtime move target")
    eef_pos = np.asarray(capture_robot_state(inner)["eef_pos"], dtype=np.float64)
    object_pos = object_xyz(inner, object_name)
    return runtime_move_trigger_decision(step, eef_pos, object_pos, settings)
