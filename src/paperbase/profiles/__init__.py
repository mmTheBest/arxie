"""Built-in field profile presets for Paperbase extraction."""

from paperbase.profiles.registry import get_profile_preset, list_profile_presets
from paperbase.profiles.sc_regnet import SC_REGNET_PRESET_NAME, sc_regnet_preset, sc_regnet_schema_payload

__all__ = [
    "SC_REGNET_PRESET_NAME",
    "get_profile_preset",
    "list_profile_presets",
    "sc_regnet_preset",
    "sc_regnet_schema_payload",
]
