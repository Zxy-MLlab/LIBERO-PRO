"""Single Hydra CLI entry point for policy evaluation."""

import json
import os
import platform
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from .clients import create_client
from .protocol import ActionSpec
from .runner import EvaluationRunner
from .live_preview import LivePreviewServer


def validate_config(cfg):
    spec = ActionSpec(**OmegaConf.to_container(cfg.policy.action, resolve=True))
    if spec.dim != 7 or spec.type != "delta_ee": raise ValueError("LIBERO requires delta_ee actions with dim=7")
    for field in ("execute_horizon", "max_steps", "episode_timeout_seconds"):
        if float(cfg.rollout[field]) <= 0: raise ValueError("rollout.{} must be positive".format(field))
    if int(cfg.rollout.warmup_steps) < 0: raise ValueError("rollout.warmup_steps cannot be negative")
    if float(cfg.policy.connection.timeout_seconds) <= 0: raise ValueError("policy.connection.timeout_seconds must be positive")
    for section in ("recording", "live_preview"):
        if int(cfg[section].stride) <= 0: raise ValueError("{}.stride must be positive".format(section))


def prepare_environment(cfg):
    repo = Path(__file__).resolve().parents[2]; package = repo / "libero" / "libero"
    paths = {"benchmark_root":package, "bddl_files":package/"bddl_files", "init_states":package/"init_files",
             "datasets":repo/"datasets", "assets":package/"assets"}
    missing = [str(p) for k,p in paths.items() if k != "datasets" and not p.exists()]
    if missing: raise FileNotFoundError("required LIBERO paths missing: " + ", ".join(missing))
    paths["datasets"].mkdir(exist_ok=True)
    config_dir = repo / ".runtime" / "libero_config"; config_dir.mkdir(parents=True, exist_ok=True)
    payload = {k:str(v.resolve()) for k,v in paths.items()}
    (config_dir/"config.yaml").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    os.environ["MUJOCO_GL"] = str(cfg.environment.render_backend)
    os.environ["PYOPENGL_PLATFORM"] = str(cfg.environment.render_backend)
    return payload


def check_render_backend(backend):
    """Fail early with an actionable error before importing robosuite."""
    try:
        if backend == "egl":
            from mujoco.egl import GLContext
        elif backend == "osmesa":
            from mujoco.osmesa import GLContext
        else:
            raise ValueError("environment.render_backend must be 'egl' or 'osmesa'")
        context = GLContext(16, 16)
        try:
            context.make_current()
        finally:
            context.free()
    except Exception as exc:
        hint = ("use environment.render_backend=egl" if backend == "osmesa" else
                "verify that the system EGL driver is installed")
        raise RuntimeError("MuJoCo {} rendering is unavailable ({}: {}); {}".format(
            backend, type(exc).__name__, exc, hint)) from exc


def run(cfg):
    validate_config(cfg)
    paths = prepare_environment(cfg)
    from .perturbation import prepare_evaluation
    identity = prepare_evaluation(str(cfg.benchmark.evaluation_config_path))
    check_render_backend(str(cfg.environment.render_backend))
    output = Path(str(cfg.output.directory)); output.mkdir(parents=True, exist_ok=True)
    (output/"config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    client = create_client(cfg.policy); preview = None
    try:
        info = client.check()
        if not info.ready: raise RuntimeError("policy client is not ready")
        if cfg.live_preview.enabled:
            preview = LivePreviewServer(host=str(cfg.live_preview.host), port=int(cfg.live_preview.port),
                                        refresh_hz=float(cfg.live_preview.refresh_hz)); url = preview.start()
        else: url = None
        metadata = {**identity.as_dict(), "suite":identity.effective_suite,
                    "client": info.__dict__, "libero_paths": paths, "live_preview_url": url,
                    "python":sys.version, "platform":platform.platform()}
        (output/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return EvaluationRunner(cfg, client, preview=preview, suite_identity=identity).run()
    finally:
        if preview is not None: preview.close()
        client.close()


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg: DictConfig):
    summary, _ = run(cfg); print(OmegaConf.to_yaml(summary))


if __name__ == "__main__": main()
