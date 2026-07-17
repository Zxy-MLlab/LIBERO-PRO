from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT / "libero_pro_dataset"
DEFAULT_SOURCE_ROOT = ROOT / "libero" / "libero" / "bddl_files"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATASET_ROOT / "bddl_files"
BDDL_ROOT = DEFAULT_SOURCE_ROOT
DEMO_ROOT = DEFAULT_OUTPUT_ROOT
MANIFEST_PATH = DEMO_ROOT.parent / "metadata" / "bddl_40task_7perturbation_manifest.json"

SOURCE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


@dataclass(frozen=True)
class Category:
    folder: str
    case_name: str
    config_kind: str


CATEGORIES = (
    Category("01_visual_noise_glare", "visual_noise_glare", "visual_noise_glare"),
    Category("02_camera_view_angle", "camera_view_angle", "camera_view_angle"),
    Category("03_runtime_object_move", "runtime_object_move", "runtime_object_move"),
    Category("04_object_texture", "object_texture", "object_texture"),
    Category("05_view_occlusion", "view_occlusion", "view_occlusion"),
    Category("06_object_shape", "object_shape", "object_shape"),
    Category("07_initial_pose_position_angle", "initial_pose_position_angle", "initial_pose_position_angle"),
)

PRESERVE_SOURCE_SCENE_CONFIGS = {category.config_kind for category in CATEGORIES}

INACTIVE_TARGET_OVERRIDES = {
    ("libero_goal", "open_the_middle_drawer_of_the_cabinet"): "akita_black_bowl_1",
    ("libero_goal", "turn_on_the_stove"): "akita_black_bowl_1",
}

APPEARANCE_TARGET_OVERRIDES = {
    ("libero_goal", "push_the_plate_to_the_front_of_the_stove"): ["plate_1"],
}

OCCLUSION_BETWEEN_OVERRIDES = {
    ("libero_goal", "put_the_wine_bottle_on_the_rack"): {
        "object": "akita_black_bowl_1",
        "min_distance": "0.18",
        "delta_xy": ["-0.18", "0.10"],
        "after_avoid": True,
    },
    ("libero_goal", "put_the_wine_bottle_on_top_of_the_cabinet"): {
        "object": "akita_black_bowl_1",
        "min_distance": "0.18",
        "delta_xy": ["-0.18", "0.10"],
        "after_avoid": True,
    },
}


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


def tokens(raw: str) -> list[str]:
    return [item for item in raw.replace("(", " ").replace(")", " ").split() if item and not item.startswith(":")]


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


def parse_region_targets(text: str) -> dict[str, str]:
    regions: dict[str, str] = {}
    block = section(text, "regions")
    for match in re.finditer(r"\(\s*([A-Za-z0-9_]+)\s+\(:target\s+([A-Za-z0-9_]+)\)", block):
        regions[match.group(1)] = match.group(2)
    return regions


def parse_relations(text: str, objects: dict[str, str], section_name: str) -> list[tuple[str, str]]:
    relations: list[tuple[str, str]] = []
    block = section(text, section_name)
    for _, child, raw_parent in re.findall(r"\((On|In)\s+(\S+)\s+(\S+)\)", block, flags=re.I):
        child = child.rstrip(")")
        raw_parent = raw_parent.rstrip(")")
        if child not in objects:
            continue
        parent = object_from_reference(raw_parent, objects)
        if parent is not None:
            relations.append((child, parent))
    return relations


def is_surface_target(name: str, object_type: str = "") -> bool:
    target_text = f"{name} {object_type}".lower()
    return "table" in target_text or "floor" in target_text


def object_from_reference(reference: str, objects: dict[str, str]) -> str | None:
    if reference in objects:
        return reference
    for name in sorted(objects, key=len, reverse=True):
        if reference.startswith(f"{name}_"):
            return name
    return None


def name_without_index(name: str) -> str:
    return re.sub(r"_\d+$", "", name)


def semantic_reference_keywords(name: str, object_type: str) -> list[str]:
    candidates = [name_without_index(name), object_type]
    for candidate in list(candidates):
        parts = [part for part in candidate.split("_") if part and not part.isdigit()]
        if len(parts) > 1:
            candidates.append("_".join(parts[-2:]))
        if parts:
            candidates.append(parts[-1])
    return unique_ordered(
        [
            candidate.lower()
            for candidate in candidates
            if candidate and candidate.lower() not in {"1", "region", "object"}
        ]
    )


def object_from_semantic_reference(reference: str, objects: dict[str, str]) -> str | None:
    reference_lower = reference.lower()
    scored: list[tuple[int, int, str]] = []
    for name, object_type in objects.items():
        if is_surface_target(name, object_type):
            continue
        for keyword in semantic_reference_keywords(name, object_type):
            if f"_{keyword}_" in f"_{reference_lower}_" or reference_lower.startswith(f"{keyword}_"):
                scored.append((len(keyword), len(name), name))
                break
    if not scored:
        return None
    return sorted(scored, reverse=True)[0][2]


def goal_reference_target(
    reference: str,
    objects: dict[str, str],
    fixtures: dict[str, str],
    region_targets: dict[str, str],
) -> str | None:
    combined = {**objects, **fixtures}
    direct = object_from_reference(reference, combined)
    if direct is not None and not is_surface_target(direct, combined.get(direct, "")):
        return direct
    semantic = object_from_semantic_reference(reference, combined)
    if semantic is not None:
        return semantic
    region_target = region_targets.get(reference)
    if region_target is not None:
        resolved = object_from_reference(region_target, combined)
        if resolved is not None:
            return resolved
    return direct


def parse_goal_destination_targets(
    text: str,
    objects: dict[str, str],
    fixtures: dict[str, str],
) -> list[str]:
    region_targets = parse_region_targets(text)
    block = section(text, "goal")
    destinations: list[str] = []
    for _, child, raw_parent in re.findall(r"\((On|In)\s+(\S+)\s+(\S+)\)", block, flags=re.I):
        child = child.rstrip(")")
        raw_parent = raw_parent.rstrip(")")
        if child not in objects:
            continue
        parent = goal_reference_target(raw_parent, objects, fixtures, region_targets)
        if parent is not None and parent != child:
            destinations.append(parent)
    for raw_target in re.findall(r"\((Turnon|Open|Close)\s+(\S+)\)", block, flags=re.I):
        target_ref = raw_target[1].rstrip(")")
        target = goal_reference_target(target_ref, objects, fixtures, region_targets)
        if target is not None:
            destinations.append(target)
    return unique_ordered(destinations)


def is_bowl_type(object_type: str) -> bool:
    return "bowl" in object_type


def is_plate_type(object_type: str) -> bool:
    return object_type == "plate" or "plate" in object_type


def unique_ordered(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def choose_surface_fixture(text: str, fixtures: dict[str, str]) -> str:
    region_targets = parse_region_targets(text)
    fixture_targets = [target for target in region_targets.values() if target in fixtures]
    preferred = [
        target
        for target in fixture_targets
        if "table" in target or target == "floor"
    ]
    if preferred:
        return preferred[0]
    if fixture_targets:
        return fixture_targets[0]
    if fixtures:
        return next(iter(fixtures))
    raise ValueError("BDDL has no fixture surface for perturbation objects.")


def noncritical_free_objects(
    objects: dict[str, str],
    critical: set[str],
    init_relations: list[tuple[str, str]],
) -> list[str]:
    relation_objects = {name for pair in init_relations for name in pair}
    return [
        name
        for name in objects
        if name not in critical and name not in relation_objects
    ]


def next_numbered_names(objects: dict[str, str], object_type: str, count: int) -> list[str]:
    used = []
    prefix = f"{object_type}_"
    for name in objects:
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        if suffix.isdigit():
            used.append(int(suffix))
    start = max(used, default=0) + 1
    return [f"{object_type}_{idx}" for idx in range(start, start + count)]


def first_numbered_name_after(objects: dict[str, str], object_type: str) -> str:
    return next_numbered_names(objects, object_type, 1)[0]


def bowl_plate_goal_partner(
    objects: dict[str, str],
    target: str,
    goal_relations: list[tuple[str, str]],
) -> str:
    target_type = objects.get(target, "")
    for child, parent in goal_relations:
        if child == target and parent in objects:
            parent_type = objects[parent]
            if (is_bowl_type(target_type) and is_plate_type(parent_type)) or (
                is_plate_type(target_type) and is_bowl_type(parent_type)
            ):
                return parent
        if parent == target and child in objects:
            child_type = objects[child]
            if (is_bowl_type(target_type) and is_plate_type(child_type)) or (
                is_plate_type(target_type) and is_bowl_type(child_type)
            ):
                return child
    return ""


def object_reference_pattern(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?=(_|[^A-Za-z0-9_]|$))")


def line_mentions_removed_object(line: str, removed_objects: set[str]) -> bool:
    return any(object_reference_pattern(name).search(line) for name in removed_objects)


def rewrite_typed_name_section(text: str, section_name: str, removed_objects: set[str]) -> str:
    if not removed_objects:
        return text
    block = section(text, section_name)
    if not block:
        return text
    new_lines: list[str] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(f"(:{section_name}") or " - " not in line:
            new_lines.append(raw_line)
            continue
        lhs, rhs = line.split(" - ", 1)
        kept_names = [name for name in lhs.split() if name not in removed_objects]
        if kept_names:
            indent = raw_line[: len(raw_line) - len(raw_line.lstrip())]
            new_lines.append(f"{indent}{sexpr_list(kept_names)} - {rhs.strip()}")
    return text.replace(block, "\n".join(new_lines), 1)


def remove_object_references_from_simple_section(
    text: str,
    section_name: str,
    removed_objects: set[str],
) -> str:
    if not removed_objects:
        return text
    block = section(text, section_name)
    if not block:
        return text
    new_lines: list[str] = []
    for raw_line in block.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(f"(:{section_name}") or stripped == ")":
            new_lines.append(raw_line)
            continue
        kept_items = [
            item
            for item in stripped.split()
            if not any(item == name or item.startswith(f"{name}_") for name in removed_objects)
        ]
        if kept_items:
            indent = raw_line[: len(raw_line) - len(raw_line.lstrip())]
            new_lines.append(f"{indent}{sexpr_list(kept_items)}")
    return text.replace(block, "\n".join(new_lines), 1)


def remove_predicates_mentioning_objects(
    text: str,
    section_name: str,
    removed_objects: set[str],
) -> str:
    if not removed_objects:
        return text
    block = section(text, section_name)
    if not block:
        return text
    new_lines = [
        raw_line
        for raw_line in block.splitlines()
        if not line_mentions_removed_object(raw_line, removed_objects)
    ]
    return text.replace(block, "\n".join(new_lines), 1)


def target_duplicate_objects_to_remove(text: str) -> set[str]:
    objects = parse_typed_names(text, "objects")
    if not objects:
        return set()
    interests = [item for item in tokens(section(text, "obj_of_interest")) if item != "obj_of_interest"]
    movable_interests = [name for name in interests if name in objects]
    goal_objects = {item for item in tokens(section(text, "goal")) if item in objects}
    critical_objects = set(movable_interests) | goal_objects
    critical_types = {objects[name] for name in critical_objects}
    return {
        name
        for name, object_type in objects.items()
        if object_type in critical_types and name not in critical_objects
    }


def enforce_target_object_uniqueness(text: str) -> tuple[str, list[str]]:
    removed_objects = target_duplicate_objects_to_remove(text)
    if not removed_objects:
        return text, []
    text = rewrite_typed_name_section(text, "objects", removed_objects)
    text = remove_object_references_from_simple_section(text, "obj_of_interest", removed_objects)
    text = remove_predicates_mentioning_objects(text, "init", removed_objects)
    text = remove_predicates_mentioning_objects(text, "goal", removed_objects)
    return text, sorted(removed_objects)


def background_rgba_for_task(suite: str, task_name: str, surface: str) -> tuple[float, float, float, float]:
    task_upper = task_name.upper()
    if suite == "libero_object":
        return (0.35, 0.34, 0.32, 1.0)
    if "LIVING_ROOM" in task_upper:
        return (0.30, 0.20, 0.12, 1.0)
    if "STUDY" in task_upper:
        return (0.50, 0.36, 0.20, 1.0)
    if surface == "floor":
        return (0.35, 0.34, 0.32, 1.0)
    return (0.72, 0.63, 0.50, 1.0)


def parse_task(text: str) -> dict[str, object]:
    objects = parse_typed_names(text, "objects")
    fixtures = parse_typed_names(text, "fixtures")
    scene_targets = {**objects, **fixtures}
    interests = [item for item in tokens(section(text, "obj_of_interest")) if item != "obj_of_interest"]
    if not objects:
        raise ValueError("BDDL has no movable objects.")
    movable_interests = [name for name in interests if name in objects]
    goal_objects = {item for item in tokens(section(text, "goal")) if item in objects}
    region_targets = parse_region_targets(text)
    resolved_interests = unique_ordered(
        [
            resolved
            for item in interests
            for resolved in [goal_reference_target(item, objects, fixtures, region_targets)]
            if resolved is not None and not is_surface_target(resolved, scene_targets.get(resolved, ""))
        ]
    )
    init_relations = parse_relations(text, objects, "init")
    goal_relations = parse_relations(text, objects, "goal")
    goal_targets = parse_goal_destination_targets(text, objects, fixtures)
    critical = set(resolved_interests) | set(goal_targets) | set(movable_interests) | goal_objects
    target = (resolved_interests or goal_targets or movable_interests or [next(iter(objects))])[0]
    critical_order = unique_ordered([*movable_interests, *[name for name in sorted(goal_objects) if name in objects]])
    appearance_targets = parse_goal_destination_targets(text, objects, fixtures)
    if not appearance_targets:
        appearance_targets = critical_order or [target]
    occlusion_reference = target
    occlusion_between_object = bowl_plate_goal_partner(objects, target, goal_relations)
    occlusion_avoid_target = occlusion_between_object
    if not occlusion_avoid_target:
        for name in interests[1:]:
            if name != occlusion_reference:
                occlusion_avoid_target = name
                break
    if any(category == "black_book" for category in objects.values()):
        occluder_type = "yellow_book"
        occluders = next_numbered_names(objects, "yellow_book", 1)
    else:
        occluder_type = "black_book"
        occluders = ["black_book_occluder_1"]
    runtime_target = target
    pose_target = target
    return {
        "objects": objects,
        "fixtures": fixtures,
        "interests": interests,
        "goal_objects": sorted(goal_objects),
        "init_relations": init_relations,
        "goal_relations": goal_relations,
        "critical_objects": sorted(critical),
        "target": target,
        "appearance_targets": appearance_targets,
        "occlusion_reference": occlusion_reference,
        "occlusion_between_object": occlusion_between_object,
        "occlusion_avoid_target": occlusion_avoid_target,
        "occluder_type": occluder_type,
        "runtime_target": runtime_target,
        "pose_target": pose_target,
        "occluders": occluders,
        "surface": choose_surface_fixture(text, fixtures),
    }


def apply_task_overrides(
    task: dict[str, object], suite: str, task_name: str
) -> dict[str, object]:
    key = (suite, task_name)
    objects = dict(task["objects"])

    inactive_target = INACTIVE_TARGET_OVERRIDES.get(key, str(task["target"]))
    if inactive_target not in objects:
        raise ValueError(f"Inactive target override {inactive_target!r} is not a movable object in {key}")
    task["inactive_target"] = inactive_target

    if key in APPEARANCE_TARGET_OVERRIDES:
        appearance_targets = APPEARANCE_TARGET_OVERRIDES[key]
        missing = [name for name in appearance_targets if name not in objects]
        if missing:
            raise ValueError(f"Appearance target overrides do not exist in {key}: {missing}")
        task["active_appearance_targets"] = appearance_targets

    if key in OCCLUSION_BETWEEN_OVERRIDES:
        override = OCCLUSION_BETWEEN_OVERRIDES[key]
        between_object = str(override["object"])
        if between_object not in objects:
            raise ValueError(f"Occlusion between-object {between_object!r} is not in {key}")
        task["occlusion_between_object"] = between_object
        task["occlusion_min_between_distance"] = str(override["min_distance"])
        task["occlusion_between_delta_xy"] = list(override["delta_xy"])
        task["occlusion_between_after_avoid"] = bool(override.get("after_avoid", False))
    else:
        task["occlusion_min_between_distance"] = "0.24"
        task["occlusion_between_delta_xy"] = ["0.16", "0.0"]
        task["occlusion_between_after_avoid"] = False
    return task


def sexpr_list(items: list[str]) -> str:
    return " ".join(items)


def portable_path(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def replace_section_field(lines: list[str], field: str, value: str) -> list[str]:
    prefix = f"      (:{field} "
    return [f"{prefix}{value})" if line.startswith(prefix) else line for line in lines]


def default_config_sections(
    target: str,
    appearance_targets: list[str],
    appearance_rgba_text: str,
    occlusion_reference: str,
    runtime_target: str,
    pose_target: str,
    occluders: list[str],
) -> dict[str, list[str]]:
    return {
        "lighting": [
            "    (:lighting",
            "      (:enabled false)",
            "      (:mode none)",
            "      (:ambient 1.0)",
            "      (:diffuse 1.0)",
            "      (:specular 1.0)",
            "    )",
        ],
        "observation": [
            "    (:observation",
            "      (:enabled false)",
            "      (:mode none)",
            "      (:brightness 1.0)",
            "      (:sigma 0.0)",
            "    )",
        ],
        "camera": [
            "    (:camera",
            "      (:enabled false)",
            "      (:mode none)",
            "      (:name agentview)",
            "      (:pos 0.0 0.0 0.0)",
            "      (:quat 1.0 0.0 0.0 0.0)",
            "    )",
        ],
        "runtime_move": [
            "    (:runtime_move",
            "      (:enabled false)",
            "      (:mode none)",
            f"      (:object {runtime_target})",
            "      (:step 0)",
            "      (:delta_xy 0.0 0.0)",
            "      (:min_delta_norm 0.0)",
            "    )",
        ],
        "object_appearance": [
            "    (:object_appearance",
            "      (:enabled false)",
            "      (:mode none)",
            f"      (:targets {sexpr_list(appearance_targets)})",
            f"      (:rgba {appearance_rgba_text})",
            "      (:override_material_rgba false)",
            "      (:disable_material_texture false)",
            "      (:apply_visible_geom_rgba false)",
            "    )",
        ],
        "object_placement": [
            "    (:object_placement",
            "      (:enabled false)",
            "      (:mode none)",
            f"      (:reference {occlusion_reference})",
            "      (:front_direction 1.0 0.0)",
            f"      (:targets {sexpr_list(occluders)})",
            "      (:offsets_xy (",
            "        (0.0 0.0)",
            "      ))",
            "      (:target_quat 1.0 0.0 0.0 0.0)",
            "      (:z_offset 0.0)",
            "      (:mesh_scale 1.0 1.0 1.0)",
            "      (:min_reference_distance 0.0)",
            "      (:pin_targets false)",
            "    )",
        ],
        "object_shape": [
            "    (:object_shape",
            "      (:enabled false)",
            "      (:mode none)",
            f"      (:target {target})",
            "      (:mesh_scale 1.0 1.0 1.0)",
            "    )",
        ],
        "initial_pose": [
            "    (:initial_pose",
            "      (:enabled false)",
            "      (:mode none)",
            f"      (:target {pose_target})",
            "      (:delta_xy 0.0 0.0)",
            "      (:min_delta_norm 0.0)",
            "      (:yaw 0.0)",
            "    )",
        ],
    }


def build_config(category: Category, suite: str, task_name: str, task: dict[str, object]) -> str:
    target = str(task["target"])
    inactive_target = str(task.get("inactive_target", target))
    appearance_targets = [str(item) for item in task["appearance_targets"]]
    active_appearance_targets = [
        str(item) for item in task.get("active_appearance_targets", appearance_targets)
    ]
    appearance_rgba = background_rgba_for_task(suite, task_name, str(task["surface"]))
    appearance_rgba_text = " ".join(f"{value:.2f}" if idx < 3 else f"{value:.1f}" for idx, value in enumerate(appearance_rgba))
    occlusion_reference = str(task["occlusion_reference"])
    occlusion_between_object = str(task["occlusion_between_object"])
    occlusion_avoid_target = str(task["occlusion_avoid_target"])
    runtime_target = str(task["runtime_target"])
    pose_target = str(task["pose_target"])
    occluders = [str(item) for item in task["occluders"]]
    common = [
        f"    (:case {category.case_name})",
        f"    (:category {category.case_name})",
        f"    (:source_suite {suite})",
        f"    (:source_task {task_name})",
        "    (:bddl_only true)",
    ]

    sections = default_config_sections(
        target,
        appearance_targets,
        appearance_rgba_text,
        occlusion_reference,
        runtime_target,
        pose_target,
        occluders,
    )
    if inactive_target != target:
        if category.config_kind != "runtime_object_move":
            sections["runtime_move"] = replace_section_field(
                sections["runtime_move"], "object", inactive_target
            )
        if category.config_kind != "view_occlusion":
            sections["object_placement"] = replace_section_field(
                sections["object_placement"], "reference", inactive_target
            )
        if category.config_kind not in {"object_shape", "initial_pose_position_angle"}:
            sections["object_shape"] = replace_section_field(
                sections["object_shape"], "target", inactive_target
            )
        if category.config_kind != "initial_pose_position_angle":
            sections["initial_pose"] = replace_section_field(
                sections["initial_pose"], "target", inactive_target
            )
    detail = "load demo BDDL perturbation config"

    if category.config_kind == "visual_noise_glare":
        detail = "mildly_dim_lighting_with_light_policy_observation_noise"
        sections["lighting"] = [
            "    (:lighting",
            "      (:enabled true)",
            "      (:mode dark_lighting)",
            "      (:ambient 0.020)",
            "      (:diffuse 0.16)",
            "      (:specular 0.010)",
            "    )",
        ]
        sections["observation"] = [
            "    (:observation",
            "      (:enabled true)",
            "      (:mode dark_noise)",
            "      (:brightness 0.68)",
            "      (:sigma 12.0)",
            "    )",
        ]
    elif category.config_kind == "camera_view_angle":
        detail = "agentview_camera_shifted_to_an_oblique_view"
        sections["camera"] = [
            "    (:camera",
            "      (:enabled true)",
            "      (:mode agentview_angle)",
            "      (:name agentview)",
            "      (:pos 0.48 0.24 1.62)",
            "      (:quat 0.6099232 0.3594558 0.3594558 0.6099232)",
            "    )",
        ]
    elif category.config_kind == "runtime_object_move":
        detail = "move_the_task_target_object_during_the_rollout"
        sections["runtime_move"] = [
            "    (:runtime_move",
            "      (:enabled true)",
            "      (:mode qpos_delta_xy)",
            f"      (:object {runtime_target})",
            "      (:step 80)",
            "      (:trigger near_grasp)",
            "      (:distance_threshold 0.09)",
            "      (:fallback_step 160)",
            "      (:delta_xy 0.17 0.06)",
            "      (:min_delta_norm 0.18)",
            "    )",
        ]
    elif category.config_kind == "object_texture":
        detail = "make_goal_destination_background_like_without_changing_task_semantics"
        sections["object_appearance"] = [
            "    (:object_appearance",
            "      (:enabled true)",
            "      (:mode table_like_material)",
            f"      (:targets {sexpr_list(active_appearance_targets)})",
            f"      (:rgba {appearance_rgba_text})",
            "      (:override_material_rgba true)",
            "      (:disable_material_texture true)",
            "      (:apply_visible_geom_rgba true)",
            "    )",
        ]
    elif category.config_kind == "view_occlusion":
        detail = "stand_one_book_face_to_agentview_in_front_of_the_target_without_touching_goal_objects"
        side_offset_tasks = {
            "open_the_middle_drawer_of_the_cabinet",
            "open_the_top_drawer_and_put_the_bowl_inside",
            "put_the_bowl_on_the_stove",
            "put_the_bowl_on_top_of_the_cabinet",
            "turn_on_the_stove",
        }
        offsets = [[0.080, 0.090 if task_name in side_offset_tasks else 0.0]]
        offset_lines = ["      (:offsets_xy ("]
        offset_lines.extend(f"        ({x:.3f} {y:.3f})" for x, y in offsets)
        offset_lines.append("      ))")
        separation_lines = []
        if occlusion_between_object:
            min_between_distance = str(task["occlusion_min_between_distance"])
            between_delta_xy = [str(value) for value in task["occlusion_between_delta_xy"]]
            separation_lines = [
                f"      (:between_object {occlusion_between_object})",
                f"      (:min_between_distance {min_between_distance})",
                f"      (:between_delta_xy {between_delta_xy[0]} {between_delta_xy[1]})",
            ]
        avoid_lines = []
        if occlusion_avoid_target:
            avoid_lines = [
                f"      (:avoid_targets {occlusion_avoid_target})",
                "      (:min_avoid_distance 0.08)",
            ]
        relation_lines = (
            [*avoid_lines, *separation_lines]
            if task["occlusion_between_after_avoid"]
            else [*separation_lines, *avoid_lines]
        )
        sections["object_placement"] = [
            "    (:object_placement",
            "      (:enabled true)",
            "      (:mode front_occlusion)",
            f"      (:reference {occlusion_reference})",
            "      (:front_direction 1.0 0.0)",
            *relation_lines,
            f"      (:targets {sexpr_list(occluders)})",
            *offset_lines,
            "      (:target_quat 1.0 0.0 0.0 0.0)",
            "      (:z_offset 0.000)",
            "      (:mesh_scale 1.6 1.6 1.6)",
            "      (:min_reference_distance 0.02)",
            "      (:pin_targets true)",
            "    )",
        ]
    elif category.config_kind == "object_shape":
        detail = "scale_the_target_object_mesh_to_half_size"
        sections["object_shape"] = [
            "    (:object_shape",
            "      (:enabled true)",
            "      (:mode mesh_scale)",
            f"      (:target {target})",
            "      (:mesh_scale 0.5 0.5 0.5)",
            "    )",
        ]
    elif category.config_kind == "initial_pose_position_angle":
        detail = "shift_the_initial_position_and_yaw_of_the_target_object"
        sections["initial_pose"] = [
            "    (:initial_pose",
            "      (:enabled true)",
            "      (:mode delta_xy_yaw)",
            f"      (:target {pose_target})",
            "      (:delta_xy 0.17 -0.06)",
            "      (:min_delta_norm 0.18)",
            "      (:yaw 0.7853981633974483)",
            "    )",
        ]
    else:
        raise ValueError(category.config_kind)

    section_lines = [
        *sections["lighting"],
        *sections["observation"],
        *sections["camera"],
        *sections["runtime_move"],
        *sections["object_appearance"],
        *sections["object_placement"],
        *sections["object_shape"],
        *sections["initial_pose"],
    ]
    return "  (:perturbation_config\n" + "\n".join([*common, f"    (:detail {detail})", *section_lines]) + "\n  )"


def insert_before_section_end(text: str, section_name: str, lines: list[str]) -> str:
    if not lines:
        return text
    block = section(text, section_name)
    replacement = block.rstrip()
    if not replacement.endswith(")"):
        raise ValueError(f"Could not edit :{section_name} section.")
    replacement = replacement[:-1].rstrip() + "\n" + "\n".join(lines) + "\n  )"
    return text.replace(block, replacement, 1)


def add_perturbation_object_declarations(text: str, names: list[str], object_type: str) -> str:
    existing_objects = parse_typed_names(text, "objects")
    new_names = [name for name in names if name not in existing_objects]
    if not new_names:
        return text
    return insert_before_section_end(
        text,
        "objects",
        [f"    {sexpr_list(new_names)} - {object_type}"],
    )


def add_perturbation_init_regions(text: str, names: list[str], surface: str, prefix: str) -> str:
    existing_objects = parse_typed_names(text, "objects")
    new_names = [name for name in names if name in existing_objects]
    if not new_names:
        return text
    base_positions = [(-0.34, -0.30), (-0.26, -0.30), (-0.18, -0.30), (-0.10, -0.30)]
    region_lines: list[str] = []
    init_lines: list[str] = []
    for idx, name in enumerate(new_names):
        x, y = base_positions[idx % len(base_positions)]
        region = f"{prefix}_{idx + 1}_init_region"
        region_lines.extend(
            [
                f"      ({region}",
                f"          (:target {surface})",
                "          (:ranges (",
                f"              ({x} {y} {x + 0.01} {y + 0.01})",
                "            )",
                "          )",
                "          (:yaw_rotation (",
                "              (0.0 0.0)",
                "            )",
                "          )",
                "      )",
            ]
        )
        init_lines.append(f"    (On {name} {surface}_{region})")
    text = insert_before_section_end(text, "regions", region_lines)
    return insert_before_section_end(text, "init", init_lines)


def prepare_task_bddl(text: str, category: Category, task: dict[str, object]) -> str:
    if category.config_kind == "view_occlusion":
        occluders = [str(item) for item in task["occluders"]]
        text = add_perturbation_object_declarations(text, occluders, str(task["occluder_type"]))
        text = add_perturbation_init_regions(text, occluders, str(task["surface"]), "perturb_occ_bottle")
    elif category.config_kind in {"runtime_object_move", "initial_pose_position_angle"}:
        objects = parse_typed_names(text, "objects")
        fixtures = parse_typed_names(text, "fixtures")
        scene_targets = {**objects, **fixtures}
        target_name = str(task["runtime_target"] if category.config_kind == "runtime_object_move" else task["pose_target"])
        if target_name not in scene_targets:
            raise ValueError(f"{category.folder} selected non-existent perturbation target {target_name}")
    return text


def inject_config(text: str, config: str) -> str:
    if ":perturbation_config" in text:
        existing = section(text, "perturbation_config")
        text = text.replace(existing, "")
    stripped = text.rstrip()
    if not stripped.endswith(")"):
        raise ValueError("BDDL does not end with a closing parenthesis.")
    return stripped[:-1].rstrip() + "\n\n" + config + "\n\n)\n"


def clean_category_bddl_dirs(categories: tuple[Category, ...] = CATEGORIES) -> None:
    for category in categories:
        bddl_dir = DEMO_ROOT / category.folder / "bddl"
        if bddl_dir.exists():
            shutil.rmtree(bddl_dir)
        bddl_dir.mkdir(parents=True, exist_ok=True)


def generate(clean: bool, selected_categories: set[str] | None = None) -> dict[str, object]:
    categories = tuple(
        category
        for category in CATEGORIES
        if selected_categories is None
        or category.folder in selected_categories
        or category.case_name in selected_categories
    )
    if selected_categories is not None and not categories:
        raise ValueError(f"No categories matched: {sorted(selected_categories)}")
    if clean:
        clean_category_bddl_dirs(categories)
    manifest_path = MANIFEST_PATH
    source_root_value = portable_path(BDDL_ROOT, ROOT)
    output_root_value = portable_path(DEMO_ROOT, DEMO_ROOT.parent)
    if selected_categories is not None and manifest_path.exists():
        manifest: dict[str, object] = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_bddl_root"] = source_root_value
        manifest["demo_root"] = output_root_value
        manifest.setdefault("categories", {})
    else:
        manifest = {
            "source_bddl_root": source_root_value,
            "demo_root": output_root_value,
            "categories": {},
            "total_files": 0,
        }
    for category in categories:
        category_info = {"folder": category.folder, "suites": {}, "count": 0}
        for suite in SOURCE_SUITES:
            src_dir = BDDL_ROOT / suite
            if not src_dir.exists():
                raise FileNotFoundError(src_dir)
            out_dir = DEMO_ROOT / category.folder / "bddl" / suite
            out_dir.mkdir(parents=True, exist_ok=True)
            suite_files = []
            for src_path in sorted(src_dir.glob("*.bddl")):
                text = src_path.read_text(encoding="utf-8")
                if category.config_kind in PRESERVE_SOURCE_SCENE_CONFIGS:
                    removed_duplicate_objects = []
                else:
                    text, removed_duplicate_objects = enforce_target_object_uniqueness(text)
                task = apply_task_overrides(parse_task(text), suite, src_path.stem)
                text = prepare_task_bddl(text, category, task)
                config = build_config(category, suite, src_path.stem, task)
                out_path = out_dir / src_path.name
                out_path.write_text(inject_config(text, config), encoding="utf-8")
                suite_files.append(
                    {
                        "task": src_path.stem,
                        "path": portable_path(out_path, DEMO_ROOT.parent),
                        "target": task["target"],
                        "appearance_targets": task["appearance_targets"],
                        "appearance_rgba": background_rgba_for_task(suite, src_path.stem, str(task["surface"])),
                        "occlusion_reference": task["occlusion_reference"],
                        "occlusion_between_object": task["occlusion_between_object"],
                        "occlusion_avoid_target": task["occlusion_avoid_target"],
                        "runtime_target": task["runtime_target"],
                        "pose_target": task["pose_target"],
                        "occluders": task["occluders"],
                        "occluder_type": task["occluder_type"],
                        "occlusion_style": "upright_book_face_to_agentview_pinned",
                        "occlusion_offset_xy": [
                            0.08,
                            0.09
                            if src_path.stem
                            in {
                                "open_the_middle_drawer_of_the_cabinet",
                                "open_the_top_drawer_and_put_the_bowl_inside",
                                "put_the_bowl_on_the_stove",
                                "put_the_bowl_on_top_of_the_cabinet",
                                "turn_on_the_stove",
                            }
                            else 0.0,
                        ],
                        "critical_objects": task["critical_objects"],
                        "removed_duplicate_objects": removed_duplicate_objects,
                    }
                )
            category_info["suites"][suite] = suite_files
            category_info["count"] += len(suite_files)
        manifest["categories"][category.folder] = category_info
    manifest["total_files"] = sum(
        int(info.get("count", 0))
        for info in dict(manifest.get("categories", {})).values()
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    global BDDL_ROOT, DEMO_ROOT, MANIFEST_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default=str(DEFAULT_SOURCE_ROOT),
        help="Directory containing the clean LIBERO suite folders.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory that will contain the seven category folders.",
    )
    parser.add_argument(
        "--manifest",
        help="Optional manifest path; defaults to <output-root>/../metadata/.",
    )
    parser.add_argument("--clean", action="store_true", help="Remove old category bddl folders before generating.")
    parser.add_argument(
        "--categories",
        nargs="*",
        help="Optional category folders or case names to regenerate, for example 03_runtime_object_move runtime_object_move.",
    )
    args = parser.parse_args()
    BDDL_ROOT = Path(args.source_root).resolve()
    DEMO_ROOT = Path(args.output_root).resolve()
    MANIFEST_PATH = (
        Path(args.manifest).resolve()
        if args.manifest
        else DEMO_ROOT.parent / "metadata" / "bddl_40task_7perturbation_manifest.json"
    )
    manifest = generate(clean=args.clean, selected_categories=set(args.categories) if args.categories else None)
    if args.categories:
        print(f"updated {', '.join(args.categories)}")
    print(f"manifest total {manifest['total_files']} BDDL files")
    print(MANIFEST_PATH)


if __name__ == "__main__":
    main()
