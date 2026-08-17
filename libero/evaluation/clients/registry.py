"""Explicit registry for policy transports."""

from typing import Dict, Type

from ..policy_client import PolicyClient

_CLIENTS: Dict[str, Type[PolicyClient]] = {}


def register_client(name: str):
    if not name or not isinstance(name, str):
        raise ValueError("client name must be a non-empty string")

    def decorate(cls: Type[PolicyClient]):
        if name in _CLIENTS:
            raise ValueError("client {!r} is already registered".format(name))
        if not issubclass(cls, PolicyClient):
            raise TypeError("registered client must implement PolicyClient")
        _CLIENTS[name] = cls
        return cls

    return decorate


def available_clients():
    return tuple(sorted(_CLIENTS))


def create_client(policy_cfg) -> PolicyClient:
    name = policy_cfg.get("client")
    if name not in _CLIENTS:
        raise ValueError(
            "unknown client {!r}; available clients: {}".format(
                name, ", ".join(available_clients()) or "none"
            )
        )
    cls = _CLIENTS[name]
    if not hasattr(cls, "from_config"):
        raise TypeError("client {!r} does not provide from_config()".format(name))
    try:
        return cls.from_config(policy_cfg)
    except Exception:
        # A partially constructed transport must be cleaned up by from_config;
        # there is no safe instance to close here.
        raise
