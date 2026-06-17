"""Shared version helpers for Arxie/Paperbase runtime surfaces."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

PACKAGE_NAME = "arxie"
DEFAULT_VERSION = "0.2.0"


def get_version() -> str:
    """Return the installed package version, or the release default in source checkouts."""

    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return DEFAULT_VERSION
