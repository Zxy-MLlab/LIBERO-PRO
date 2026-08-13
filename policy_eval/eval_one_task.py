"""Run one LIBERO task against a remote policy action server.

This process owns simulation only. It never imports a policy model and therefore does
not need the model server's CUDA, checkpoints, or Python dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .live_preview import LivePreviewServer
from .protocol import PolicyClient, ProtocolError, make_action_request


DEFAULT_TASK_NAME = "pick_up_the_cream_cheese_and_place_it_in_the_basket"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return normalized or "run"


def parse_init_state_ids(value: Optional[str], n_episodes: int) -> List[int]:
    if value is None:
        return list(range(n_episodes))
    parsed: List[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"invalid init-state range: {token!r}")
            parsed.extend(range(start, end + 1))
        else:
            parsed.append(int(token))
    if not parsed:
        raise ValueError("--init-state-ids did not contain any IDs")
    if any(item < 0 for item in parsed):
        raise ValueError("init-state IDs must be non-negative")
    if len(parsed) != n_episodes:
        raise ValueError(
            f"--init-state-ids contains {len(parsed)} IDs, but --n-episodes is {n_episodes}"
        )
    return parsed


def resolve_data_paths(
    repo_root: Path, explicit_data_root: Optional[Path], suite_name: str
) -> Dict[str, Path]:
    package_root = repo_root / "libero" / "libero"
    candidates: List[Path] = []
    if explicit_data_root is not None:
        candidates.append(explicit_data_root.expanduser().resolve())
    # The repository is self-contained by default. --data-root remains available
    # only as an explicit override for deployments that keep data elsewhere.
    candidates.append(package_root)

    def select_category(category: str) -> Path:
        # BDDL and init-state archives do not necessarily contain the same suites.
        # Select each category independently and require the requested suite folder,
        # rather than accepting a superficially valid but incomplete data root.
        selected = next(
            (
                candidate / category
                for candidate in candidates
                if (candidate / category / suite_name).is_dir()
            ),
            None,
        )
        if selected is None:
            searched = ", ".join(str(path / category / suite_name) for path in candidates)
            raise FileNotFoundError(
                f"could not find suite {suite_name!r} in {category}; searched: {searched}"
            )
        return selected

    datasets = repo_root / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    paths = {
        "benchmark_root": package_root,
        "bddl_files": select_category("bddl_files"),
        "init_states": select_category("init_files"),
        "datasets": datasets,
        "assets": package_root / "assets",
    }
    missing = [f"{key}={value}" for key, value in paths.items() if not value.exists()]
    if missing:
        raise FileNotFoundError("required LIBERO paths are missing: " + ", ".join(missing))
    return paths


def configure_libero_before_imports(
    *,
    repo_root: Path,
    data_root: Optional[Path],
    suite_name: str,
    config_dir: Path,
    render_backend: str,
) -> Dict[str, str]:
    """Create a non-interactive LIBERO config before importing the package."""

    paths = resolve_data_paths(repo_root, data_root, suite_name)
    config_dir = config_dir.expanduser().resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {key: str(value.resolve()) for key, value in paths.items()}
    # YAML 1.2 accepts JSON; this avoids importing PyYAML before the environment is ready.
    (config_dir / "config.yaml").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    os.environ["MUJOCO_GL"] = render_backend
    os.environ["PYOPENGL_PLATFORM"] = render_backend
    return config_payload


def package_versions(names: Iterable[str]) -> Dict[str, Optional[str]]:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # Python 3.7 fallback
        from importlib_metadata import PackageNotFoundError, version  # type: ignore

    result: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None, "max_ms": None}
    converted = [float(value) for value in values]
    return {
        "count": len(converted),
        "mean_ms": statistics.fmean(converted),
        "median_ms": statistics.median(converted),
        "p95_ms": percentile(converted, 0.95),
        "max_ms": max(converted),
    }


class VideoRecorder:
    """Incremental MP4 writer; imports video dependencies only when requested."""

    def __init__(self, path: Path, *, enabled: bool, fps: int) -> None:
        self.path = path
        self.enabled = enabled
        self.fps = fps
        self._writer: Any = None

    def __enter__(self) -> "VideoRecorder":
        if self.enabled:
            try:
                import imageio.v2 as imageio
            except ImportError as exc:
                raise RuntimeError(
                    "video output needs imageio and imageio-ffmpeg; install policy_eval/requirements-eval.txt"
                ) from exc
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = imageio.get_writer(str(self.path), fps=self.fps)
        return self

    def append(self, rgb_upright: Any) -> None:
        if self._writer is not None:
            self._writer.append_data(rgb_upright)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._writer is not None:
            self._writer.close()


def upright_rgb(obs: Dict[str, Any], key: str, np: Any) -> Any:
    if key not in obs:
        raise KeyError(f"LIBERO observation has no {key!r}; available keys: {sorted(obs)}")
    image = np.asarray(obs[key])
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"observation {key!r} is not an HWC RGB image: shape={image.shape}")
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.max(initial=0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    # MuJoCo's offscreen camera rows are bottom-to-top. This is a vertical flip only.
    return np.ascontiguousarray(image[::-1])


def publish_live_preview(
    preview: Optional[LivePreviewServer],
    obs: Dict[str, Any],
    *,
    status: Dict[str, Any],
    np: Any,
) -> None:
    if preview is None:
        return
    preview.publish(
        agentview_rgb=upright_rgb(obs, "agentview_image", np),
        wrist_rgb=upright_rgb(obs, "robot0_eye_in_hand_image", np),
        status=status,
    )


def policy_state(obs: Dict[str, Any], np: Any, transform_utils: Any) -> Tuple[Any, Any, Any, Any]:
    position = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
    quaternion = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1)
    gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if len(position) != 3 or len(quaternion) != 4 or len(gripper) != 2:
        raise ValueError(
            "unexpected LIBERO proprio shapes: "
            f"eef_pos={position.shape}, eef_quat={quaternion.shape}, gripper={gripper.shape}"
        )
    axis_angle = np.asarray(transform_utils.quat2axisangle(quaternion), dtype=np.float32)
    state_vector = np.concatenate((position, axis_angle, gripper), axis=0)
    if state_vector.shape != (8,):
        raise ValueError(f"policy state_vector must be shape (8,), got {state_vector.shape}")
    return position, quaternion, gripper, state_vector


def select_task(suite: Any, *, task_name: Optional[str], task_id: Optional[int]) -> Tuple[int, Any]:
    task_names = suite.get_task_names()
    if task_name is not None:
        if task_name not in task_names:
            preview = "\n  - ".join(task_names)
            raise ValueError(
                f"task {task_name!r} is not in suite {suite.name!r}. Available tasks:\n  - {preview}"
            )
        selected_id = task_names.index(task_name)
    else:
        if task_id is None or not 0 <= task_id < suite.get_num_tasks():
            raise ValueError(f"task ID must be in [0, {suite.get_num_tasks() - 1}]")
        selected_id = task_id
    return selected_id, suite.get_task(selected_id)


def evaluate_episode(
    *,
    env: Any,
    initial_state: Any,
    client: PolicyClient,
    suite_name: str,
    task_name: str,
    language_instruction: str,
    episode_index: int,
    init_state_id: int,
    max_steps: int,
    stabilization_steps: int,
    action_horizon: int,
    replan_steps: int,
    video_path: Path,
    save_video: bool,
    video_fps: int,
    video_stride: int,
    live_preview: Optional[LivePreviewServer],
    live_preview_stride: int,
    np: Any,
    transform_utils: Any,
) -> Dict[str, Any]:
    started_at = utc_now()
    start = time.perf_counter()
    round_trip_latencies: List[float] = []
    server_latencies: List[float] = []
    total_reward = 0.0
    policy_queries = 0
    steps_executed = 0
    success = False
    error: Optional[str] = None
    obs: Optional[Dict[str, Any]] = None

    with VideoRecorder(video_path, enabled=save_video, fps=video_fps) as video:
        try:
            env.reset()
            state_value = initial_state
            if hasattr(state_value, "detach"):
                state_value = state_value.detach().cpu().numpy()
            obs = env.set_init_state(state_value)
            publish_live_preview(
                live_preview,
                obs,
                status={
                    "phase": "stabilizing",
                    "episode_index": episode_index,
                    "init_state_id": init_state_id,
                    "stabilization_step": 0,
                    "stabilization_steps": stabilization_steps,
                    "step": 0,
                    "max_steps": max_steps,
                    "policy_queries": 0,
                    "success": False,
                    "error": None,
                },
                np=np,
            )
            stabilization_action = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
            for stabilization_step in range(stabilization_steps):
                obs, _, _, _ = env.step(stabilization_action)
                completed_stabilization_steps = stabilization_step + 1
                if (
                    completed_stabilization_steps % live_preview_stride == 0
                    or completed_stabilization_steps == stabilization_steps
                ):
                    publish_live_preview(
                        live_preview,
                        obs,
                        status={
                            "phase": "stabilizing",
                            "episode_index": episode_index,
                            "init_state_id": init_state_id,
                            "stabilization_step": completed_stabilization_steps,
                            "stabilization_steps": stabilization_steps,
                            "step": 0,
                            "max_steps": max_steps,
                            "policy_queries": 0,
                            "success": False,
                        },
                        np=np,
                    )

            video.append(upright_rgb(obs, "agentview_image", np))
            success = bool(env.check_success())
            publish_live_preview(
                live_preview,
                obs,
                status={
                    "phase": "episode_complete" if success else "evaluating",
                    "episode_index": episode_index,
                    "init_state_id": init_state_id,
                    "stabilization_step": stabilization_steps,
                    "stabilization_steps": stabilization_steps,
                    "step": 0,
                    "max_steps": max_steps,
                    "policy_queries": 0,
                    "success": success,
                },
                np=np,
            )
            action_queue: List[List[float]] = []

            while steps_executed < max_steps and not success:
                if not action_queue:
                    agentview = upright_rgb(obs, "agentview_image", np)
                    wrist = upright_rgb(obs, "robot0_eye_in_hand_image", np)
                    position, quaternion, gripper, state_vector = policy_state(
                        obs, np, transform_utils
                    )
                    request = make_action_request(
                        suite=suite_name,
                        task_name=task_name,
                        language_instruction=language_instruction,
                        episode_index=episode_index,
                        init_state_id=init_state_id,
                        step=steps_executed,
                        agentview_rgb=agentview,
                        wrist_rgb=wrist,
                        eef_position=position,
                        eef_quaternion_xyzw=quaternion,
                        gripper_qpos=gripper,
                        state_vector=state_vector,
                        requested_action_horizon=action_horizon,
                    )
                    if live_preview is not None:
                        live_preview.update_status(
                            {
                                "phase": "waiting_for_policy",
                                "episode_index": episode_index,
                                "init_state_id": init_state_id,
                                "step": steps_executed,
                                "max_steps": max_steps,
                                "policy_queries": policy_queries,
                                "success": False,
                            }
                        )
                    response, actions, round_trip_ms = client.infer(request)
                    policy_queries += 1
                    round_trip_latencies.append(round_trip_ms)
                    server_latencies.append(float(response["inference_ms"]))
                    action_queue = actions[:replan_steps]

                action = np.asarray(action_queue.pop(0), dtype=np.float32)
                obs, reward, _, _ = env.step(action)
                steps_executed += 1
                total_reward += float(reward)
                success = bool(env.check_success())
                if steps_executed % video_stride == 0 or success:
                    video.append(upright_rgb(obs, "agentview_image", np))
                if steps_executed % live_preview_stride == 0 or success:
                    publish_live_preview(
                        live_preview,
                        obs,
                        status={
                            "phase": "episode_complete" if success else "evaluating",
                            "episode_index": episode_index,
                            "init_state_id": init_state_id,
                            "step": steps_executed,
                            "max_steps": max_steps,
                            "policy_queries": policy_queries,
                            "success": success,
                            "last_round_trip_ms": round_trip_latencies[-1],
                            "last_server_inference_ms": server_latencies[-1],
                        },
                        np=np,
                    )
                if steps_executed % 50 == 0 or success:
                    print(
                        f"  episode={episode_index} step={steps_executed}/{max_steps} "
                        f"queries={policy_queries} success={success}",
                        flush=True,
                    )
            final_status: Dict[str, Any] = {
                "phase": "episode_complete",
                "termination_reason": "success" if success else "step_limit",
                "episode_index": episode_index,
                "init_state_id": init_state_id,
                "step": steps_executed,
                "max_steps": max_steps,
                "policy_queries": policy_queries,
                "success": success,
            }
            if round_trip_latencies:
                final_status["last_round_trip_ms"] = round_trip_latencies[-1]
            if server_latencies:
                final_status["last_server_inference_ms"] = server_latencies[-1]
            publish_live_preview(live_preview, obs, status=final_status, np=np)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if live_preview is not None:
                live_preview.update_status(
                    {
                        "phase": "episode_error",
                        "episode_index": episode_index,
                        "init_state_id": init_state_id,
                        "step": steps_executed,
                        "max_steps": max_steps,
                        "policy_queries": policy_queries,
                        "success": bool(success),
                        "error": error,
                    }
                )

    duration_seconds = time.perf_counter() - start
    return {
        "episode_index": episode_index,
        "init_state_id": init_state_id,
        "status": "error" if error is not None else "completed",
        "success": bool(success),
        "steps": steps_executed,
        "policy_queries": policy_queries,
        "total_reward": total_reward,
        "duration_seconds": duration_seconds,
        "started_at": started_at,
        "finished_at": utc_now(),
        "round_trip_latency": latency_stats(round_trip_latencies),
        "server_inference_latency": latency_stats(server_latencies),
        "round_trip_latency_ms": round_trip_latencies,
        "server_inference_latency_ms": server_latencies,
        "video": str(video_path) if save_video else None,
        "error": error,
    }


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    repo_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-url", default="http://127.0.0.1:8000")
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--suite", default="libero_object")
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    task_group.add_argument("--task-id", type=int)
    parser.add_argument("--n-episodes", type=int, default=1)
    parser.add_argument(
        "--init-state-ids",
        help="comma-separated IDs or ranges, e.g. 0 or 0-4; count must match --n-episodes",
    )
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument("--stabilization-steps", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=5)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--render-backend", choices=("osmesa", "egl"), default="osmesa")
    parser.add_argument("--repo-root", type=Path, default=repo_default)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="optional data-root override; defaults to <repo>/libero/libero",
    )
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--video-stride", type=int, default=2)
    parser.add_argument("--save-video", dest="save_video", action="store_true", default=True)
    parser.add_argument("--no-save-video", dest="save_video", action="store_false")
    parser.add_argument(
        "--live-preview",
        action="store_true",
        help="serve current agent and wrist camera frames in a local browser page",
    )
    parser.add_argument("--live-preview-host", default="127.0.0.1")
    parser.add_argument("--live-preview-port", type=int, default=8765)
    parser.add_argument("--live-preview-fps", type=float, default=10.0)
    parser.add_argument(
        "--live-preview-stride",
        type=int,
        default=1,
        help="publish one live frame every N environment steps",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "n_episodes",
        "max_steps",
        "action_horizon",
        "replan_steps",
        "camera_size",
        "video_fps",
        "video_stride",
        "live_preview_stride",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.stabilization_steps < 0:
        raise ValueError("--stabilization-steps cannot be negative")
    if args.request_timeout <= 0:
        raise ValueError("--request-timeout must be positive")
    if args.action_horizon > 100:
        raise ValueError("--action-horizon cannot be larger than 100")
    if args.replan_steps > args.action_horizon:
        raise ValueError("--replan-steps cannot be larger than --action-horizon")
    if not 0 <= args.live_preview_port <= 65535:
        raise ValueError("--live-preview-port must be in [0, 65535]")
    if not 0 < args.live_preview_fps <= 60:
        raise ValueError("--live-preview-fps must be in (0, 60]")
    if not args.live_preview_host:
        raise ValueError("--live-preview-host cannot be empty")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    init_state_ids = parse_init_state_ids(args.init_state_ids, args.n_episodes)
    repo_root = args.repo_root.expanduser().resolve()
    config_dir = (args.config_dir or (repo_root / ".runtime" / "libero_config")).resolve()
    output_root = (args.output_root or (repo_root / "outputs" / "policy_eval")).resolve()

    config_paths = configure_libero_before_imports(
        repo_root=repo_root,
        data_root=args.data_root,
        suite_name=args.suite,
        config_dir=config_dir,
        render_backend=args.render_backend,
    )

    try:
        import numpy as np
        from robosuite.utils import transform_utils
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
    except Exception as exc:
        raise RuntimeError(
            "LIBERO environment imports failed. Activate the WSL CPU environment and install "
            "the repository plus policy_eval/requirements-eval.txt. Original error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    np.random.seed(args.seed)
    benchmark_mapping = benchmark.get_benchmark_dict()
    if args.suite not in benchmark_mapping:
        raise ValueError(
            f"unknown suite {args.suite!r}; available: {', '.join(sorted(benchmark_mapping))}"
        )
    suite = benchmark_mapping[args.suite]()
    selected_task_name = None if args.task_id is not None else args.task_name
    task_id, task = select_task(suite, task_name=selected_task_name, task_id=args.task_id)
    bddl_path = Path(suite.get_task_bddl_file_path(task_id)).resolve()
    if not bddl_path.is_file():
        raise FileNotFoundError(f"task BDDL file does not exist: {bddl_path}")
    init_states = suite.get_task_init_states(task_id)
    for init_state_id in init_state_ids:
        if init_state_id >= len(init_states):
            raise ValueError(
                f"init-state ID {init_state_id} is out of range; task has {len(init_states)} states"
            )

    client = PolicyClient(args.policy_url, timeout_seconds=args.request_timeout)
    health = client.health()
    metadata = client.metadata()
    if health.get("status") != "ok":
        raise ProtocolError(f"policy server health check failed: {health}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.suite}_{task.name}_{timestamp}"
    run_dir = output_root / safe_name(run_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    results_path = run_dir / "results.json"

    summary: Dict[str, Any] = {
        "schema_version": "1.0",
        "status": "running",
        "started_at": utc_now(),
        "run_directory": str(run_dir),
        "configuration": {
            "policy_url": args.policy_url,
            "suite": args.suite,
            "task_id": task_id,
            "task_name": task.name,
            "language_instruction": task.language,
            "bddl_file": str(bddl_path),
            "init_state_ids": init_state_ids,
            "max_steps": args.max_steps,
            "stabilization_steps": args.stabilization_steps,
            "action_horizon": args.action_horizon,
            "replan_steps": args.replan_steps,
            "camera_size": args.camera_size,
            "render_backend": args.render_backend,
            "seed": args.seed,
            "save_video": args.save_video,
            "video_fps": args.video_fps,
            "video_stride": args.video_stride,
            "live_preview": args.live_preview,
            "live_preview_host": args.live_preview_host if args.live_preview else None,
            "live_preview_port": args.live_preview_port if args.live_preview else None,
            "live_preview_fps": args.live_preview_fps if args.live_preview else None,
            "live_preview_stride": args.live_preview_stride if args.live_preview else None,
        },
        "libero_paths": config_paths,
        "policy_server": {"health": health, "metadata": metadata},
        "system": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "packages": package_versions(
                ("libero", "torch", "numpy", "robosuite", "mujoco", "imageio")
            ),
        },
        "episodes": [],
        "aggregate": {},
    }
    atomic_write_json(results_path, summary)

    print(f"task: {args.suite}/{task.name}", flush=True)
    print(f"instruction: {task.language}", flush=True)
    print(f"policy: {metadata.get('model_name')} at {args.policy_url}", flush=True)
    print(f"output: {run_dir}", flush=True)

    live_preview: Optional[LivePreviewServer] = None
    env: Any = None
    try:
        if args.live_preview:
            live_preview = LivePreviewServer(
                host=args.live_preview_host,
                port=args.live_preview_port,
                refresh_hz=args.live_preview_fps,
            )
            preview_url = live_preview.start()
            summary["configuration"]["live_preview_url"] = preview_url
            atomic_write_json(results_path, summary)
            print(f"live preview: {preview_url}", flush=True)

        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            camera_heights=args.camera_size,
            camera_widths=args.camera_size,
            horizon=args.max_steps + args.stabilization_steps + 20,
            ignore_done=True,
        )
        env.seed(args.seed)
        for episode_index, init_state_id in enumerate(init_state_ids):
            print(
                f"starting episode={episode_index} init_state={init_state_id} "
                f"budget={args.max_steps}",
                flush=True,
            )
            result = evaluate_episode(
                env=env,
                initial_state=init_states[init_state_id],
                client=client,
                suite_name=args.suite,
                task_name=task.name,
                language_instruction=task.language,
                episode_index=episode_index,
                init_state_id=init_state_id,
                max_steps=args.max_steps,
                stabilization_steps=args.stabilization_steps,
                action_horizon=args.action_horizon,
                replan_steps=args.replan_steps,
                video_path=run_dir / f"episode_{episode_index:03d}_init_{init_state_id:03d}.mp4",
                save_video=args.save_video,
                video_fps=args.video_fps,
                video_stride=args.video_stride,
                live_preview=live_preview,
                live_preview_stride=args.live_preview_stride,
                np=np,
                transform_utils=transform_utils,
            )
            summary["episodes"].append(result)
            atomic_write_json(results_path, summary)
            print(
                f"finished episode={episode_index} status={result['status']} "
                f"success={result['success']} steps={result['steps']}",
                flush=True,
            )
        if live_preview is not None:
            live_preview.update_status(
                {
                    "phase": "all_episodes_complete",
                    "episodes_completed": len(summary["episodes"]),
                    "episodes_requested": args.n_episodes,
                    "successes": sum(
                        int(episode["success"])
                        for episode in summary["episodes"]
                        if episode["status"] == "completed"
                    ),
                }
            )
    except Exception as exc:
        if live_preview is not None:
            live_preview.update_status(
                {
                    "phase": "evaluation_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        raise
    finally:
        if env is not None:
            env.close()
        if live_preview is not None:
            live_preview.close()

    completed = [episode for episode in summary["episodes"] if episode["status"] == "completed"]
    successes = sum(int(episode["success"]) for episode in completed)
    all_round_trip: List[float] = []
    all_server_inference: List[float] = []
    for episode in completed:
        all_round_trip.extend(float(value) for value in episode["round_trip_latency_ms"])
        all_server_inference.extend(
            float(value) for value in episode["server_inference_latency_ms"]
        )
    summary["aggregate"] = {
        "episodes_requested": args.n_episodes,
        "episodes_completed": len(completed),
        "episodes_with_error": args.n_episodes - len(completed),
        "successes": successes,
        "success_rate": (successes / len(completed)) if completed else None,
        "total_steps": sum(int(episode["steps"]) for episode in summary["episodes"]),
        "total_policy_queries": sum(
            int(episode["policy_queries"]) for episode in summary["episodes"]
        ),
        "round_trip_latency": latency_stats(all_round_trip),
        "server_inference_latency": latency_stats(all_server_inference),
    }
    summary["status"] = "completed" if len(completed) == args.n_episodes else "completed_with_errors"
    summary["finished_at"] = utc_now()
    atomic_write_json(results_path, summary)
    print(
        f"evaluation {summary['status']}: success_rate={summary['aggregate']['success_rate']} "
        f"results={results_path}",
        flush=True,
    )
    return 0 if summary["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
