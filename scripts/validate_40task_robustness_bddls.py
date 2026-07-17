from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT / "libero_pro_dataset"
DATASET_ROOT = DEFAULT_DATASET_ROOT
DEMO_ROOT = DATASET_ROOT / "bddl_files"
for path in [ROOT]:
    if path.exists():
        sys.path.insert(0, str(path))

SOURCE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CATEGORY_FOLDERS = (
    "01_visual_noise_glare",
    "02_camera_view_angle",
    "03_runtime_object_move",
    "04_object_texture",
    "05_view_occlusion",
    "06_object_shape",
    "07_initial_pose_position_angle",
)

CATEGORY_ALIASES = {
    folder: folder for folder in CATEGORY_FOLDERS
}
CATEGORY_ALIASES.update(
    {
        folder.split("_", 1)[1]: folder
        for folder in CATEGORY_FOLDERS
    }
)


def load_local_perturbation_config_parser():
    parser_path = ROOT / "libero" / "libero" / "envs" / "perturbation_config.py"
    spec = importlib.util.spec_from_file_location("libero_pro_perturbation_config", parser_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load perturbation config parser from {parser_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_bddl_perturbation_config


def tokenize_sexpr(text: str) -> list[str]:
    spaced = text.replace("(", " ( ").replace(")", " ) ")
    return [token for token in spaced.split() if token]


def parse_sexpr(tokens: list[str]) -> list[Any]:
    if not tokens:
        raise ValueError("empty token stream")
    token = tokens.pop(0)
    if token == "(":
        out = []
        while tokens and tokens[0] != ")":
            out.append(parse_sexpr(tokens))
        if not tokens:
            raise ValueError("unclosed list")
        tokens.pop(0)
        return out
    if token == ")":
        raise ValueError("unexpected )")
    return token


def normalize_key(key: Any) -> str:
    return str(key).lstrip(":").replace("-", "_")


def parse_atom(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        if any(char in value for char in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def looks_like_field(value: Any) -> bool:
    return (
        isinstance(value, list)
        and value
        and isinstance(value[0], str)
        and value[0].startswith(":")
    )


def parse_node(node: Any) -> Any:
    if isinstance(node, list):
        if looks_like_field(node):
            return {normalize_key(node[0]): parse_values(node[1:])}
        return [parse_node(value) for value in node]
    return parse_atom(node)


def parse_values(values: list[Any]) -> Any:
    if not values:
        return True
    if len(values) == 1:
        return parse_node(values[0])
    if all(looks_like_field(value) for value in values):
        out: dict[str, Any] = {}
        for value in values:
            key = normalize_key(value[0])
            parsed = parse_values(value[1:])
            if key in out:
                if not isinstance(out[key], list):
                    out[key] = [out[key]]
                out[key].append(parsed)
            else:
                out[key] = parsed
        return out
    return [parse_node(value) for value in values]


def parse_bddl_perturbation_config_fallback(path: Path) -> dict[str, Any]:
    config_text = section(path.read_text(encoding="utf-8"), "perturbation_config")
    if not config_text:
        return {}
    parsed = parse_sexpr(tokenize_sexpr(config_text))
    if not parsed or parsed[0] != ":perturbation_config":
        return {}
    value = parse_values(parsed[1:])
    return value if isinstance(value, dict) else {}


try:
    parse_bddl_perturbation_config = load_local_perturbation_config_parser()
except ModuleNotFoundError:
    parse_bddl_perturbation_config = parse_bddl_perturbation_config_fallback


def section(text: str, name: str) -> str:
    start = text.find(f"(:{name}")
    if start < 0:
        return ""
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise ValueError(f"Unclosed section :{name}")


def parse_typed_names(text: str, section_name: str) -> dict[str, str]:
    out: dict[str, str] = {}
    block = section(text, section_name)
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(f"(:{section_name}") or " - " not in line:
            continue
        lhs, rhs = line.split(" - ", 1)
        category = rhs.strip().split()[0]
        for name in lhs.split():
            out[name] = category
    return out


def config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    return value if isinstance(value, dict) else {}


def section_enabled(config: dict[str, Any]) -> bool:
    if not config:
        return False
    value = config.get("enabled", True)
    if isinstance(value, str):
        enabled = value.lower() == "true"
    else:
        enabled = bool(value)
    if not enabled:
        return False
    return str(config.get("mode", "")).lower() not in {"none", "default", "identity", "disabled", "off"}


def require_default_section(config: dict[str, Any], name: str, errors: list[str]) -> None:
    require(bool(config), f"missing common section {name}", errors)
    if not config:
        return
    require(config.get("enabled") is False, f"inactive {name} must set enabled false", errors)
    require(str(config.get("mode", "")).lower() == "none", f"inactive {name} mode must be none", errors)


def as_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def object_from_reference(reference: str, objects: dict[str, str]) -> str | None:
    if reference in objects:
        return reference
    for name in sorted(objects, key=len, reverse=True):
        if reference.startswith(f"{name}_"):
            return name
    return None


def bddl_tokens(raw: str) -> list[str]:
    return [item for item in raw.replace("(", " ").replace(")", " ").split() if item and not item.startswith(":")]


def goal_objects(text: str, objects: dict[str, str]) -> set[str]:
    return {item for item in bddl_tokens(section(text, "goal")) if item in objects}


def object_reference_pattern(name: str):
    import re

    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?=(_|[^A-Za-z0-9_]|$))")


def validate_target_uniqueness(text: str, objects: dict[str, str], warnings: list[str]) -> None:
    interests = [item for item in bddl_tokens(section(text, "obj_of_interest")) if item != "obj_of_interest"]
    critical = {name for name in interests if name in objects} | goal_objects(text, objects)
    critical_types = {objects[name] for name in critical}
    extras = [
        name
        for name, object_type in objects.items()
        if object_type in critical_types and name not in critical
    ]
    if extras:
        warnings.append(f"source-scene duplicate target-type objects are preserved: {extras}")
    init_block = section(text, "init")
    goal_block = section(text, "goal")
    for extra in extras:
        pattern = object_reference_pattern(extra)
        if pattern.search(init_block):
            warnings.append(f"preserved duplicate {extra!r} is still referenced in init")
        if pattern.search(goal_block):
            warnings.append(f"preserved duplicate {extra!r} is still referenced in goal")


def parse_categories(raw: str) -> tuple[str, ...]:
    if not raw:
        return CATEGORY_FOLDERS
    categories = []
    for item in [part.strip() for part in raw.split(",") if part.strip()]:
        if item not in CATEGORY_ALIASES:
            raise ValueError(
                f"Unknown category {item!r}. Use one of: {', '.join(CATEGORY_FOLDERS)}"
            )
        categories.append(CATEGORY_ALIASES[item])
    return tuple(dict.fromkeys(categories))


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def require_keys(config: dict[str, Any], section: str, keys: tuple[str, ...], errors: list[str]) -> None:
    for key in keys:
        require(key in config, f"{section} missing {key}", errors)


def require_xy_pairs(value: Any, section: str, errors: list[str]) -> None:
    require(isinstance(value, list) and bool(value), f"{section} must be a non-empty list", errors)
    if not isinstance(value, list):
        return
    pairs = value if value and isinstance(value[0], list) else [value]
    for idx, pair in enumerate(pairs):
        require(isinstance(pair, list) and len(pair) == 2, f"{section}[{idx}] must have 2 values", errors)


def validate_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    objects = parse_typed_names(text, "objects")
    fixtures = parse_typed_names(text, "fixtures")
    config = parse_bddl_perturbation_config(path)
    errors: list[str] = []
    warnings: list[str] = []

    require(bool(config), "missing :perturbation_config", errors)
    category = str(config.get("category", ""))
    require(bool(category), "missing category", errors)
    require(str(config.get("case", "")) == category, "case must match category", errors)
    require(config.get("bddl_only") is True, "bddl_only is not true", errors)
    require(
        str(config.get("source_suite", "")).lower() == path.parent.name.lower(),
        "source_suite must match suite folder",
        errors,
    )
    require(
        str(config.get("source_task", "")).lower() == path.stem.lower(),
        "source_task must match bddl filename",
        errors,
    )
    validate_target_uniqueness(text, objects, warnings)

    lighting = config_section(config, "lighting")
    observation = config_section(config, "observation")
    camera = config_section(config, "camera")
    runtime_move = config_section(config, "runtime_move")
    object_appearance = config_section(config, "object_appearance")
    object_shape = config_section(config, "object_shape")
    object_placement = config_section(config, "object_placement")
    initial_pose = config_section(config, "initial_pose")

    common_sections = {
        "visual_noise_glare": ("lighting", "observation"),
        "camera_view_angle": ("camera",),
        "runtime_object_move": ("runtime_move",),
        "object_texture": ("object_appearance",),
        "view_occlusion": ("object_placement",),
        "object_shape": ("object_shape",),
        "initial_pose_position_angle": ("initial_pose",),
    }
    all_sections = {
        "lighting": lighting,
        "observation": observation,
        "camera": camera,
        "runtime_move": runtime_move,
        "object_appearance": object_appearance,
        "object_placement": object_placement,
        "object_shape": object_shape,
        "initial_pose": initial_pose,
    }
    active_section_names = set(common_sections.get(category, ()))
    for name, value in all_sections.items():
        require(bool(value), f"missing common section {name}", errors)
        if name in active_section_names:
            require(section_enabled(value), f"{name} should be active for {category}", errors)
        else:
            require_default_section(value, name, errors)

    if category == "visual_noise_glare":
        require_keys(lighting, "lighting", ("mode", "ambient", "diffuse", "specular"), errors)
        require_keys(observation, "observation", ("mode", "brightness", "sigma"), errors)
        require(lighting.get("mode") == "dark_lighting", "visual category missing dark_lighting", errors)
        require(observation.get("mode") == "dark_noise", "visual category missing dark_noise", errors)
        require(0.60 <= float(observation.get("brightness", 1.0)) <= 1.0, "brightness should keep moderate dimming", errors)
        require(0.0 <= float(observation.get("sigma", 0.0)) <= 16.0, "noise sigma should be moderate", errors)
    elif category == "camera_view_angle":
        require_keys(camera, "camera", ("mode", "name", "pos", "quat"), errors)
        require(camera.get("mode") == "agentview_angle", "camera category missing agentview_angle", errors)
        require(len(camera.get("pos", [])) == 3, "camera pos must have 3 values", errors)
        require(len(camera.get("quat", [])) == 4, "camera quat must have 4 values", errors)
    elif category == "runtime_object_move":
        require_keys(
            runtime_move,
            "runtime_move",
            ("mode", "object", "step", "trigger", "distance_threshold", "fallback_step", "delta_xy"),
            errors,
        )
        require(runtime_move.get("mode") == "qpos_delta_xy", "runtime_move mode must be qpos_delta_xy", errors)
        target = str(runtime_move.get("object", ""))
        scene_targets = {**objects, **fixtures}
        require(target in scene_targets, f"runtime target {target!r} is not in objects or fixtures", errors)
        require(int(runtime_move.get("step", 0)) > 0, "runtime step must be positive", errors)
        require(runtime_move.get("trigger") == "near_grasp", "runtime trigger must be near_grasp", errors)
        require(
            float(runtime_move.get("distance_threshold", 999.0)) <= 0.09,
            "runtime distance_threshold must be at most 0.09m",
            errors,
        )
        require(int(runtime_move.get("fallback_step", 0)) > int(runtime_move.get("step", 0)), "runtime fallback_step must be after step", errors)
        require(len(runtime_move.get("delta_xy", [])) == 2, "runtime delta_xy must have 2 values", errors)
    elif category == "object_texture":
        require_keys(
            object_appearance,
            "object_appearance",
            ("mode", "targets", "rgba", "override_material_rgba", "disable_material_texture"),
            errors,
        )
        require(object_appearance.get("mode") in {"rgba", "table_like_material"}, "object_appearance mode must be supported", errors)
        targets = as_names(object_appearance.get("targets") or object_appearance.get("target"))
        require(bool(targets), "object_texture has no targets", errors)
        scene_targets = {**objects, **fixtures}
        for target in targets:
            require(target in scene_targets, f"appearance target {target!r} is not in objects or fixtures", errors)
        require(len(object_appearance.get("rgba", [])) == 4, "appearance rgba must have 4 values", errors)
        require(object_appearance.get("override_material_rgba") is True, "object_texture should override material rgba", errors)
        require(object_appearance.get("disable_material_texture") is True, "object_texture should disable original material textures", errors)
    elif category == "view_occlusion":
        require_keys(object_placement, "object_placement", ("mode", "reference", "targets", "offsets_xy"), errors)
        require(object_placement.get("mode") == "front_occlusion", "object_placement mode must be front_occlusion", errors)
        reference = str(object_placement.get("reference", ""))
        targets = as_names(object_placement.get("targets") or object_placement.get("target"))
        scene_targets = {**objects, **fixtures}
        require(reference in scene_targets, f"occlusion reference {reference!r} is not in objects or fixtures", errors)
        require(bool(targets), "occlusion has no targets", errors)
        for target in targets:
            require(target in objects, f"occluder {target!r} is not a movable object", errors)
        offsets = object_placement.get("offsets_xy", [])
        require_xy_pairs(offsets, "object_placement offsets_xy", errors)
    elif category == "object_shape":
        require_keys(object_shape, "object_shape", ("mode", "target", "mesh_scale"), errors)
        require(object_shape.get("mode") == "mesh_scale", "object_shape mode must be mesh_scale", errors)
        target = str(object_shape.get("target", ""))
        scene_targets = {**objects, **fixtures}
        require(target in scene_targets, f"shape target {target!r} is not in objects or fixtures", errors)
        require(len(object_shape.get("mesh_scale", [])) == 3, "mesh_scale must have 3 values", errors)
    elif category == "initial_pose_position_angle":
        require_keys(initial_pose, "initial_pose", ("mode", "target", "delta_xy", "yaw"), errors)
        require(initial_pose.get("mode") == "delta_xy_yaw", "initial_pose mode must be delta_xy_yaw", errors)
        target = str(initial_pose.get("target", ""))
        scene_targets = {**objects, **fixtures}
        require(target in scene_targets, f"initial pose target {target!r} is not in objects or fixtures", errors)
        require(len(initial_pose.get("delta_xy", [])) == 2, "initial_pose delta_xy must have 2 values", errors)
        require("yaw" in initial_pose, "initial_pose missing yaw", errors)
    else:
        errors.append(f"unknown category {category!r}")

    return {
        "path": str(path),
        "category": category,
        "suite": path.parent.name,
        "task": path.stem,
        "objects": objects,
        "fixtures": fixtures,
        "config": config,
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }


def validate_all() -> dict[str, Any]:
    results = []
    counts: dict[str, dict[str, int]] = {}
    seen_keys: dict[tuple[str, str, str], str] = {}
    duplicate_results: list[dict[str, Any]] = []
    for category in CATEGORY_FOLDERS:
        counts[category] = {}
        for suite in SOURCE_SUITES:
            suite_dir = DEMO_ROOT / category / "bddl" / suite
            files = sorted(suite_dir.glob("*.bddl"))
            counts[category][suite] = len(files)
            for path in files:
                result = validate_config(path)
                key = (category, suite, path.stem)
                if key in seen_keys:
                    result["errors"].append(
                        f"duplicate BDDL for category/suite/task: {key}; first={seen_keys[key]}"
                    )
                    duplicate_results.append(
                        {
                            "key": list(key),
                            "first": seen_keys[key],
                            "duplicate": str(path),
                        }
                    )
                else:
                    seen_keys[key] = str(path)
                results.append(result)
    failures = [item for item in results if not item["ok"]]
    summary = {
        "demo_root": str(DEMO_ROOT),
        "expected_categories": len(CATEGORY_FOLDERS),
        "expected_suites": len(SOURCE_SUITES),
        "expected_total": len(CATEGORY_FOLDERS) * len(SOURCE_SUITES) * 10,
        "actual_total": len(results),
        "unique_total": len(seen_keys),
        "duplicate_count": len(duplicate_results),
        "duplicates": duplicate_results,
        "counts": counts,
        "failure_count": len(failures),
        "failures": failures,
        "results": results,
    }
    summary["ok"] = (
        summary["actual_total"] == summary["expected_total"]
        and summary["unique_total"] == summary["expected_total"]
        and not duplicate_results
        and not failures
    )
    return summary


def configure_libero_paths() -> None:
    import yaml

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ["LIBERO_CONFIG_PATH"] = os.environ.get(
        "LIBERO_CONFIG_PATH", str(ROOT / ".libero_demo_validation")
    )
    config_dir = Path(os.environ["LIBERO_CONFIG_PATH"])
    config_dir.mkdir(parents=True, exist_ok=True)
    benchmark_root = ROOT / "libero" / "libero"
    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(DATASET_ROOT / "bddl_files"),
        "init_states": str(DATASET_ROOT / "init_files"),
        "datasets": str(ROOT / "libero" / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    (config_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def validate_loaded_env(path: Path) -> dict[str, Any]:
    from libero.libero.envs.robustness_perturbations import (
        PerturbationRuntimeOptions,
        apply_camera_pose,
        apply_sim_lighting,
        apply_static_bddl_config_perturbations,
        active_config_section,
        infer_spec_from_config,
        move_object_xy,
        parse_bddl_summary,
        runtime_move_settings,
        scene_overlap_violations,
    )
    from libero.libero.envs import OffScreenRenderEnv

    configure_libero_paths()
    config = parse_bddl_perturbation_config(path)
    suite = path.parent.name
    spec = infer_spec_from_config(str(config.get("case", path.parent.parent.parent.name)), suite, path.stem, config)
    args = PerturbationRuntimeOptions(scene_contact_tolerance=-1e-4, object_name="")
    env = OffScreenRenderEnv(
        bddl_file_name=str(path),
        camera_heights=64,
        camera_widths=64,
        camera_names=["agentview", "robot0_eye_in_hand"],
        use_camera_obs=True,
    )
    try:
        env.reset()

        class VecLike:
            envs = [
                type(
                    "EnvLike",
                    (),
                    {
                        "_env": env,
                    },
                )()
            ]

        vec_env = VecLike()
        static_info = apply_static_bddl_config_perturbations(vec_env, spec, args)
        lighting_info = None
        camera_info = None
        if active_config_section(config, "lighting"):
            lighting_info = apply_sim_lighting(vec_env, spec, args)
        if active_config_section(config, "camera"):
            camera_info = apply_camera_pose(vec_env, spec, args)
        runtime_move_info = None
        if active_config_section(config, "runtime_move"):
            _, runtime_object, runtime_delta, min_delta_norm = runtime_move_settings(spec, args)
            runtime_move_info = move_object_xy(
                vec_env,
                runtime_object,
                runtime_delta,
                args,
                min_delta_norm=min_delta_norm,
            )
        overlap_violations = scene_overlap_violations(env.env, args)
        if overlap_violations:
            raise RuntimeError(f"overlap after perturbation: {overlap_violations}")
        return {
            "path": str(path),
            "ok": True,
            "bddl_summary": parse_bddl_summary(path),
            "static_info": static_info,
            "lighting_info": lighting_info,
            "camera_info": camera_info,
            "runtime_move_info": runtime_move_info,
            "overlap_violations": overlap_violations,
        }
    finally:
        env.close()


def validate_loaded_sample(
    limit_per_category: int | None,
    categories: tuple[str, ...],
) -> dict[str, Any]:
    loaded = []
    seen_paths: set[str] = set()
    duplicate_paths: list[str] = []
    for category in categories:
        paths = []
        for suite in SOURCE_SUITES:
            paths.extend(sorted((DEMO_ROOT / category / "bddl" / suite).glob("*.bddl")))
        selected_paths = paths if limit_per_category is None else paths[:limit_per_category]
        for path in selected_paths:
            path_key = str(path)
            if path_key in seen_paths:
                duplicate_paths.append(path_key)
                continue
            seen_paths.add(path_key)
            try:
                result = validate_loaded_env(path)
            except Exception as exc:  # noqa: BLE001
                result = {"path": str(path), "ok": False, "error": repr(exc)}
            loaded.append(result)
    return {
        "limit_per_category": limit_per_category,
        "categories": categories,
        "count": len(loaded),
        "unique_count": len(seen_paths),
        "duplicate_count": len(duplicate_paths),
        "duplicates": duplicate_paths,
        "failure_count": len([item for item in loaded if not item["ok"]]),
        "results": loaded,
    }


def main() -> None:
    global DATASET_ROOT, DEMO_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help="Dataset checkout containing bddl_files/, init_files/, and metadata/.",
    )
    parser.add_argument(
        "--output",
        help="Output JSON path; defaults to <dataset-root>/metadata/.",
    )
    parser.add_argument(
        "--load-env-sample",
        type=int,
        default=0,
        help="Also instantiate this many BDDLs per category and apply BDDL-config perturbations.",
    )
    parser.add_argument(
        "--load-env-all",
        action="store_true",
        help="Instantiate every BDDL in the selected load-env categories.",
    )
    parser.add_argument(
        "--load-env-categories",
        default="",
        help="Comma-separated category folders or names to load. Defaults to all categories.",
    )
    args = parser.parse_args()
    DATASET_ROOT = Path(args.dataset_root).resolve()
    DEMO_ROOT = DATASET_ROOT / "bddl_files"
    summary = validate_all()
    if args.load_env_all or args.load_env_sample > 0:
        categories = parse_categories(args.load_env_categories)
        limit = None if args.load_env_all else args.load_env_sample
        summary["loaded_env_validation"] = validate_loaded_sample(limit, categories)
        if (
            summary["loaded_env_validation"]["failure_count"]
            or summary["loaded_env_validation"]["duplicate_count"]
        ):
            summary["ok"] = False
    output = (
        Path(args.output)
        if args.output
        else DATASET_ROOT / "metadata" / "static_validation_40task_7perturbation_summary.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"validated {summary['actual_total']} BDDL files")
    print(f"failures: {summary['failure_count']}")
    print(f"summary: {output}")
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
