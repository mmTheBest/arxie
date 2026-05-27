"""Host path policy helpers for local-first and hosted API modes."""

from __future__ import annotations

from pathlib import Path

from paperbase.config import PaperbaseConfig
from services.paperbase_api.errors import PaperbaseAPIError


def ensure_host_path_allowed(
    path: str | Path,
    *,
    config: PaperbaseConfig,
    field_name: str,
) -> Path:
    """Return a resolved path if host-path access is allowed for this request."""

    resolved_path = Path(path).expanduser().resolve()
    if not config.hosted_mode:
        return resolved_path

    allowed_roots = tuple(
        Path(root).expanduser().resolve()
        for root in config.local_path_import_allowed_roots
    )
    if not allowed_roots:
        raise PaperbaseAPIError(
            status_code=403,
            error="path_not_allowed",
            message=f"{field_name} host-path access is disabled in hosted mode.",
        )

    if any(resolved_path == root or root in resolved_path.parents for root in allowed_roots):
        return resolved_path

    raise PaperbaseAPIError(
        status_code=403,
        error="path_not_allowed",
        message=f"{field_name} must be under a configured allowed root.",
    )
