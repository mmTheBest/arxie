"""Security-focused input validation helpers."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_LOCAL_HOST_SUFFIXES = (".local", ".internal", ".localhost")


def sanitize_user_text(
    value: str | object,
    *,
    field_name: str,
    max_length: int,
    allow_empty: bool = False,
) -> str:
    """Validate and normalize untrusted text input."""
    text = str(value or "").strip()
    if not text:
        if allow_empty:
            return ""
        raise ValueError(f"{field_name} must not be empty.")
    if len(text) > max_length:
        raise ValueError(f"{field_name} exceeds maximum length ({max_length}).")
    if _CONTROL_CHARS_RE.search(text):
        raise ValueError(f"{field_name} contains unsupported control characters.")
    return text


def sanitize_identifier(
    value: str | object,
    *,
    field_name: str = "identifier",
    max_length: int = 256,
) -> str:
    """Validate an identifier that should not contain embedded whitespace."""
    identifier = sanitize_user_text(value, field_name=field_name, max_length=max_length)
    if any(ch.isspace() for ch in identifier):
        raise ValueError(f"{field_name} must not contain whitespace.")
    return identifier


def validate_public_http_url(
    value: str | object,
    *,
    field_name: str = "url",
    max_length: int = 2048,
) -> str:
    """Validate URL for outbound fetches and block local-network targets."""
    raw_url = sanitize_user_text(value, field_name=field_name, max_length=max_length)
    parsed = urlsplit(raw_url)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"{field_name} must use http or https.")
    if parsed.username or parsed.password:
        raise ValueError(f"{field_name} must not include user credentials.")
    hostname = (parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        raise ValueError(f"{field_name} must include a hostname.")
    if hostname == "localhost" or hostname.endswith(_LOCAL_HOST_SUFFIXES):
        raise ValueError(f"{field_name} host is not allowed.")

    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        ip = None

    if ip is not None and (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        raise ValueError(f"{field_name} host is not allowed.")

    return raw_url
