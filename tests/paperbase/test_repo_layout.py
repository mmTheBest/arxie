from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_paperbase_foundation_layout_exists() -> None:
    expected_paths = (
        "src/paperbase/__init__.py",
        "services/paperbase_api/__init__.py",
        "services/paperbase_worker/__init__.py",
        "infra/docker-compose.paperbase.yml",
        "docs/architecture/README.md",
        "docs/architecture/01-system-overview.md",
        "docs/architecture/02-paperbase-core.md",
        "docs/architecture/06-collections-and-annotations.md",
    )

    missing = [
        rel_path
        for rel_path in expected_paths
        if not (REPO_ROOT / rel_path).exists()
    ]

    assert missing == [], f"Missing expected Paperbase foundation paths: {missing}"
