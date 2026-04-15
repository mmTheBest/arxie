"""Registry for built-in Paperbase extraction profile presets."""

from __future__ import annotations

from typing import Any

from paperbase.profiles.sc_regnet import SC_REGNET_PRESET_NAME, sc_regnet_preset


def list_profile_presets() -> list[dict[str, Any]]:
    return [sc_regnet_preset()]


def get_profile_preset(name: str) -> dict[str, Any] | None:
    if name == SC_REGNET_PRESET_NAME:
        return sc_regnet_preset()
    return None
