from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from forge.model_cores.base_dit import BaseDiT


_MODEL_CORE_REGISTRY: dict[str, type["BaseDiT"]] = {}


def register_model_core(name: str, model_cls: type["BaseDiT"]) -> None:
    existing = _MODEL_CORE_REGISTRY.get(name)
    if existing is not None and existing is not model_cls:
        raise ValueError(f"Model core '{name}' is already registered with {existing.__name__}.")
    _MODEL_CORE_REGISTRY[name] = model_cls


def get_model_core(name: str) -> type["BaseDiT"]:
    try:
        return _MODEL_CORE_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_MODEL_CORE_REGISTRY))
        raise ValueError(f"Unknown model core '{name}'. Registered cores: {known}") from exc


def list_model_cores() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_CORE_REGISTRY))
