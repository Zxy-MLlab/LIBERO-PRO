from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.robustness_perturbations import (
    DEFAULT_CASES,
    PerturbationRuntimeOptions,
    active_config_section,
    apply_camera_pose,
    apply_case_to_observation,
    apply_pinned_object_placements,
    apply_sim_lighting,
    apply_static_bddl_config_perturbations,
    infer_spec_from_config,
    move_object_xy,
    parse_bddl_summary,
    runtime_move_settings,
    runtime_move_trigger_status,
)
from libero.libero.envs.perturbation_config import parse_bddl_perturbation_config


CASE_FOLDERS = {
    "visual_noise_glare": "01_visual_noise_glare",
    "camera_view_angle": "02_camera_view_angle",
    "runtime_object_move": "03_runtime_object_move",
    "object_texture": "04_object_texture",
    "view_occlusion": "05_view_occlusion",
    "object_shape": "06_object_shape",
    "initial_pose_position_angle": "07_initial_pose_position_angle",
}


def resolve_bddl_path(args: argparse.Namespace) -> Path:
    if args.bddl:
        return Path(args.bddl).resolve()
    if args.case == "clean":
        return (
            ROOT
            / "libero"
            / "libero"
            / "bddl_files"
            / args.suite
            / f"{args.task}.bddl"
        )
    folder = CASE_FOLDERS[args.case]
    return (
        Path(args.dataset_root).resolve()
        / "bddl_files"
        / folder
        / "bddl"
        / args.suite
        / f"{args.task}.bddl"
    )


def make_vec_like(env: OffScreenRenderEnv):
    return SimpleNamespace(envs=[SimpleNamespace(_env=env)])


def observation_for_runtime(raw_obs: dict[str, object]) -> dict[str, object]:
    pixels = {}
    for key, value in raw_obs.items():
        if key.endswith("_image") and isinstance(value, np.ndarray):
            pixels[key] = value[None].copy()
    return {"pixels": pixels}


def run(args: argparse.Namespace) -> dict[str, object]:
    bddl_path = resolve_bddl_path(args)
    if not bddl_path.is_file():
        raise FileNotFoundError(bddl_path)

    config = parse_bddl_perturbation_config(bddl_path)
    spec = infer_spec_from_config(
        args.case,
        args.suite,
        args.task,
        config,
        bddl_path=bddl_path,
        init_suite=args.suite,
    )
    options = PerturbationRuntimeOptions()
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=args.height,
        camera_widths=args.width,
        camera_names=["agentview", "robot0_eye_in_hand"],
        use_camera_obs=True,
    )
    try:
        raw_obs = env.reset()
        vec_env = make_vec_like(env)
        static_info = apply_static_bddl_config_perturbations(vec_env, spec, options)
        lighting_info = (
            apply_sim_lighting(vec_env, spec, options)
            if active_config_section(config, "lighting")
            else None
        )
        camera_info = (
            apply_camera_pose(vec_env, spec, options)
            if active_config_section(config, "camera")
            else None
        )

        observation = apply_case_to_observation(
            observation_for_runtime(raw_obs), spec, args.seed, options
        )
        transformed_images = sorted(observation.get("pixels", {}))

        runtime_move = None
        runtime_settings = None
        if spec.runtime_move:
            _, object_name, delta_xy, min_delta_norm = runtime_move_settings(spec, options)
            runtime_settings = {
                "object": object_name,
                "delta_xy": list(delta_xy),
                "min_delta_norm": min_delta_norm,
            }
        else:
            object_name = ""
            delta_xy = (0.0, 0.0)
            min_delta_norm = None

        success = bool(env.check_success())
        for step in range(args.steps):
            if spec.runtime_move and runtime_move is None:
                trigger = runtime_move_trigger_status(
                    vec_env, object_name, step, spec, options
                )
                if trigger:
                    runtime_move = move_object_xy(
                        vec_env,
                        object_name,
                        delta_xy,
                        options,
                        min_delta_norm=min_delta_norm,
                    )
                    runtime_move["trigger"] = trigger
            raw_obs, _, done, _ = env.step([0.0] * 6 + [-1.0])
            apply_pinned_object_placements(vec_env, static_info)
            success = success or bool(env.check_success())
            if done or success:
                break

        return {
            "ok": True,
            "case": args.case,
            "suite": args.suite,
            "task": args.task,
            "bddl": str(bddl_path),
            "bddl_summary": parse_bddl_summary(bddl_path),
            "static_info": static_info,
            "lighting_info": lighting_info,
            "camera_info": camera_info,
            "runtime_settings": runtime_settings,
            "runtime_move": runtime_move,
            "transformed_images": transformed_images,
            "success": success,
        }
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load one LIBERO-Pro robustness BDDL and run a no-policy smoke rollout."
    )
    parser.add_argument("--dataset-root", default=str(ROOT / "libero_pro_dataset"))
    parser.add_argument("--bddl", help="Optional direct path to a BDDL file.")
    parser.add_argument("--case", choices=DEFAULT_CASES, default="clean")
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task", default="put_the_bowl_on_the_plate")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    payload = json.dumps(result, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
