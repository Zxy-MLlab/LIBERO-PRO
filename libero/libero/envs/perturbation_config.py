from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    from bddl.parsing import scan_tokens
except ModuleNotFoundError:

    def scan_tokens(filename: str) -> list[Any]:
        """Parse enough S-expression syntax for standalone config validation."""
        text = Path(filename).read_text(encoding="utf-8")
        text = re.sub(r";[^\n]*", "", text)
        tokens = re.findall(r"\(|\)|[^\s()]+", text)

        def parse(index: int) -> tuple[Any, int]:
            token = tokens[index]
            if token != "(":
                return token, index + 1
            values = []
            index += 1
            while index < len(tokens) and tokens[index] != ")":
                value, index = parse(index)
                values.append(value)
            if index >= len(tokens):
                raise ValueError(f"Unclosed BDDL list in {filename}")
            return values, index + 1

        parsed, next_index = parse(0)
        if next_index != len(tokens):
            raise ValueError(f"Unexpected trailing BDDL tokens in {filename}")
        return parsed if isinstance(parsed, list) else []


def _normalize_key(key: Any) -> str:
    return str(key).lstrip(":").replace("-", "_")


def _parse_atom(value: Any) -> Any:
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


def _looks_like_field(value: Any) -> bool:
    return (
        isinstance(value, list)
        and value
        and isinstance(value[0], str)
        and value[0].startswith(":")
    )


def _merge_field(target: dict[str, Any], key: str, value: Any) -> None:
    if key not in target:
        target[key] = value
        return
    if not isinstance(target[key], list):
        target[key] = [target[key]]
    target[key].append(value)


def _parse_values(values: list[Any]) -> Any:
    if not values:
        return True
    if len(values) == 1:
        return _parse_node(values[0])
    if all(_looks_like_field(value) for value in values):
        out: dict[str, Any] = {}
        for value in values:
            _merge_field(out, _normalize_key(value[0]), _parse_values(value[1:]))
        return out
    return [_parse_node(value) for value in values]


def _parse_node(node: Any) -> Any:
    if isinstance(node, list):
        if _looks_like_field(node):
            return {_normalize_key(node[0]): _parse_values(node[1:])}
        return [_parse_node(value) for value in node]
    return _parse_atom(node)


def parse_perturbation_config_group(group: list[Any]) -> dict[str, Any]:
    """Convert a BDDL (:perturbation_config ...) group into a plain dictionary."""
    if not group or group[0] != ":perturbation_config":
        return {}
    parsed = _parse_values(group[1:])
    return parsed if isinstance(parsed, dict) else {}


def parse_bddl_perturbation_config(bddl_path: str | Path) -> dict[str, Any]:
    tokens = scan_tokens(filename=str(bddl_path))
    if not isinstance(tokens, list) or not tokens or tokens.pop(0) != "define":
        return {}
    while tokens:
        group = tokens.pop()
        if group and group[0] == ":perturbation_config":
            return parse_perturbation_config_group(group)
    return {}
