"""Serial LIBERO evaluation runner with task-level environment reuse."""

import json
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .action_executor import ActionChunkExecutor
from .record import ActionTraceRecorder, VideoRecorder, upright_rgb


def latency_stats(values):
    values = list(map(float, values))
    return {"count": len(values), "mean_ms": statistics.mean(values) if values else None,
            "median_ms": statistics.median(values) if values else None,
            "max_ms": max(values) if values else None}


@dataclass
class EpisodeResult:
    policy: str; suite: str; task_id: int; task: str; prompt: str; episode_id: str
    base_suite: str; effective_suite: str; perturbation_type: str
    init_state_id: int; seed: int; success: bool; steps: int; duration_seconds: float
    policy_query_count: int; termination_reason: str = ""
    round_trip_latency_ms: list = field(default_factory=list)
    server_inference_latency_ms: list = field(default_factory=list)
    video_path: str = ""
    wrist_video_path: str = ""
    action_trace_path: str = ""


class EvaluationRunner:
    def __init__(self, cfg, client, suite_factory=None, env_factory=None, preview=None, suite_identity=None):
        self.cfg, self.client, self.preview = cfg, client, preview
        self.suite_factory = suite_factory or self._default_suite_factory
        self.env_factory = env_factory or self._default_env_factory
        if suite_identity is None:
            suite = str(cfg.benchmark.suite)
            from .perturbation import EvaluationSuite
            suite_identity = EvaluationSuite(suite, suite, "none")
        self.suite_identity = suite_identity
        self.executor = ActionChunkExecutor(client, int(cfg.rollout.execute_horizon), self._action_spec(cfg.policy.action))

    @staticmethod
    def _action_spec(cfg):
        from .protocol import ActionSpec
        return ActionSpec(**dict(cfg))
    @staticmethod
    def _default_suite_factory(name):
        from libero.libero.benchmark import get_benchmark
        return get_benchmark(name)(0)
    def _default_env_factory(self, task):
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        return OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
            camera_heights=int(self.cfg.recording.camera_height), camera_widths=int(self.cfg.recording.camera_width),
            horizon=int(self.cfg.rollout.max_steps) + int(self.cfg.rollout.warmup_steps) + 20, ignore_done=True)

    def run(self):
        identity = self.suite_identity; suite_name = identity.effective_suite
        suite = self.suite_factory(suite_name)
        seed = int(self.cfg.benchmark.seed); random.seed(seed); np.random.seed(seed)
        task_ids = list(self.cfg.benchmark.task_ids) or list(range(suite.n_tasks)); results = []
        for task_id in task_ids:
            if not 0 <= task_id < suite.n_tasks: raise ValueError("task_id out of range: {}".format(task_id))
            task, states = suite.get_task(task_id), suite.get_task_init_states(task_id)
            schedule = list(self.cfg.benchmark.init_state_ids)
            if not schedule:
                if not len(states): raise ValueError("task has no init states")
                episode_count = int(self.cfg.benchmark.episodes_per_task)
                if episode_count > len(states):
                    raise ValueError("episodes_per_task={} exceeds available init states={}".format(
                        episode_count, len(states)))
                schedule = list(range(episode_count))
            if any(i < 0 or i >= len(states) for i in schedule): raise ValueError("init_state_id out of range")
            env = None
            try:
                env = self.env_factory(task)
                if hasattr(env, "seed"): env.seed(seed)
                for episode_index, state_id in enumerate(schedule):
                    result = self._run_episode(env, identity, task_id, task, episode_index, state_id, states[state_id])
                    results.append(result); self._append_jsonl(result); self._write_summary(self._summary(identity, results))
            finally:
                if env is not None: env.close()
        summary = self._summary(identity, results); self._write_summary(summary); return summary, results

    def _publish(self, obs, status):
        if self.preview: self.preview.publish(agentview_rgb=upright_rgb(obs, "agentview_image"), wrist_rgb=upright_rgb(obs, "robot0_eye_in_hand_image"), status=status)

    def _action_trace_settings(self):
        diagnostics = getattr(self.cfg, "diagnostics", None)
        action_trace = getattr(diagnostics, "action_trace", None)
        enabled = bool(getattr(action_trace, "enabled", False))
        default_directory = Path(str(self.cfg.output.directory)) / "action_traces"
        directory = Path(str(getattr(action_trace, "directory", default_directory)))
        return enabled, directory

    @staticmethod
    def _trace_record(episode_id, step, instruction, metadata, action, before, after, success):
        action = np.asarray(action, dtype=np.float32)
        before_pos = np.asarray(before["robot0_eef_pos"], dtype=np.float32)
        after_pos = np.asarray(after["robot0_eef_pos"], dtype=np.float32)
        observed_delta = after_pos - before_pos
        image = np.asarray(before["agentview_image"])
        return {
            "episode_id": episode_id,
            "step": int(step),
            "instruction": instruction,
            "policy_query_index": metadata.get("policy_query_index"),
            "chunk_action_index": metadata.get("chunk_action_index"),
            "round_trip_latency_ms": metadata.get("round_trip_latency_ms"),
            "raw_model_action": metadata.get("raw_model_action"),
            "adapted_action_chunk": metadata.get("adapted_action_chunk"),
            "libero_action": action.tolist(),
            "translation_command_norm": float(np.linalg.norm(action[:3])),
            "rotation_command_norm": float(np.linalg.norm(action[3:6])),
            "gripper_command": float(action[6]),
            "before_eef_pos": before_pos.tolist(),
            "after_eef_pos": after_pos.tolist(),
            "observed_eef_delta": observed_delta.tolist(),
            "observed_eef_delta_norm": float(np.linalg.norm(observed_delta)),
            "before_eef_quat": np.asarray(before["robot0_eef_quat"], dtype=np.float32).tolist(),
            "after_eef_quat": np.asarray(after["robot0_eef_quat"], dtype=np.float32).tolist(),
            "before_gripper_qpos": np.asarray(before["robot0_gripper_qpos"], dtype=np.float32).tolist(),
            "after_gripper_qpos": np.asarray(after["robot0_gripper_qpos"], dtype=np.float32).tolist(),
            "agentview_shape": list(image.shape),
            "agentview_dtype": str(image.dtype),
            "agentview_mean": float(image.mean()),
            "agentview_std": float(image.std()),
            "model_input_image_shape": metadata.get("model_input_image_shape"),
            "model_input_image_dtype": metadata.get("model_input_image_dtype"),
            "model_input_image_mean": metadata.get("model_input_image_mean"),
            "model_input_image_std": metadata.get("model_input_image_std"),
            "unnorm_key": metadata.get("unnorm_key"),
            "image_preprocess": metadata.get("image_preprocess"),
            "center_crop": metadata.get("center_crop"),
            "success": bool(success),
        }

    def _run_episode(self, env, identity, task_id, task, episode_index, state_id, state):
        suite = identity.effective_suite
        seed = int(self.cfg.benchmark.seed)
        started = time.monotonic(); steps = 0; success = False; reason = ""; prompt = ""
        episode_id = "{}/{}/{}".format(suite, task_id, episode_index)
        video_path = Path(str(self.cfg.recording.directory)) / "task_{}_episode_{}_init_{}.mp4".format(task_id, episode_index, state_id)
        wrist_video_path = Path(str(self.cfg.recording.directory)) / "task_{}_episode_{}_init_{}_wrist.mp4".format(
            task_id, episode_index, state_id)
        trace_enabled, trace_directory = self._action_trace_settings()
        trace_path = trace_directory / "task_{}_episode_{}_init_{}.jsonl".format(task_id, episode_index, state_id)
        try:
            env.reset(); obs = env.set_init_state(state)
            warmup = np.array([0, 0, 0, 0, 0, 0, -1], np.float32)
            for warmup_step in range(int(self.cfg.rollout.warmup_steps)):
                obs, _, _, _ = env.step(warmup)
                if warmup_step % int(self.cfg.live_preview.stride) == 0: self._publish(obs, {"phase":"warmup", "episode_id":episode_id})
            prompt = str(env.language_instruction)
            self.executor.reset(episode_id, prompt)
            with VideoRecorder(video_path, bool(self.cfg.recording.enabled), int(self.cfg.recording.fps)) as video, \
                    VideoRecorder(wrist_video_path, bool(self.cfg.recording.enabled), int(self.cfg.recording.fps)) as wrist_video, \
                    ActionTraceRecorder(trace_path, trace_enabled) as action_trace:
                deadline = time.monotonic() + float(self.cfg.rollout.episode_timeout_seconds)
                last_recorded = False
                while steps < int(self.cfg.rollout.max_steps):
                    if time.monotonic() >= deadline: reason = "episode_timeout"; break
                    before = obs
                    action = self.executor.act(before, steps)
                    obs, _, done, _ = env.step(action.tolist()); steps += 1
                    success = bool(done)
                    action_trace.append(self._trace_record(
                        episode_id, steps - 1, prompt,
                        self.executor.last_action_metadata, action, before, obs, success
                    ))
                    if (steps - 1) % int(self.cfg.recording.stride) == 0:
                        video.append(upright_rgb(obs, "agentview_image"))
                        wrist_video.append(upright_rgb(obs, "robot0_eye_in_hand_image"))
                        last_recorded = True
                    else: last_recorded = False
                    if (steps - 1) % int(self.cfg.live_preview.stride) == 0: self._publish(obs, {"phase":"rollout", "step":steps, "success":success})
                    if success: reason = "success"; break
                if not success and not reason: reason = "max_steps"
                if steps and not last_recorded:
                    video.append(upright_rgb(obs, "agentview_image"))
                    wrist_video.append(upright_rgb(obs, "robot0_eye_in_hand_image"))
                self._publish(obs, {"phase":"episode_complete", "step":steps, "success":success, "termination_reason":reason})
        except Exception as exc:
            reason = "{}: {}".format(type(exc).__name__, exc)
        return EpisodeResult(str(self.cfg.policy.name), suite, task_id, task.name, prompt, episode_id,
            identity.base_suite, identity.effective_suite, identity.perturbation_type, state_id, seed,
            success, steps, time.monotonic()-started, self.executor.query_count, reason,
            list(self.executor.round_trip_latency_ms), list(self.executor.server_inference_latency_ms),
            str(video_path) if bool(self.cfg.recording.enabled) else "",
            str(wrist_video_path) if bool(self.cfg.recording.enabled) else "",
            str(trace_path) if trace_enabled else "")

    def _append_jsonl(self, result):
        path = Path(str(self.cfg.output.directory)) / str(self.cfg.output.episodes_file); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream: stream.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    def _summary(self, identity, results):
        success = sum(r.success for r in results); rt = sum((r.round_trip_latency_ms for r in results), []); server = sum((r.server_inference_latency_ms for r in results), [])
        return {"policy":str(self.cfg.policy.name), "suite":identity.effective_suite, **identity.as_dict(),
            "episodes":len(results), "successes":success,
            "success_rate":success/len(results) if results else 0.0, "policy_query_count":sum(r.policy_query_count for r in results),
            "round_trip_latency":latency_stats(rt), "server_inference_latency":latency_stats(server)}
    def _write_summary(self, summary):
        path = Path(str(self.cfg.output.directory)) / str(self.cfg.output.summary_file); path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8"); os.replace(str(tmp), str(path))
