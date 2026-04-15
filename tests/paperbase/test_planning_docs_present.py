from __future__ import annotations

from pathlib import Path


def test_april_14_platform_planning_docs_exist() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    expected = {
        "2026-04-14-arxie-homepage.md",
        "2026-04-14-arxie-platform-implementation-plan.md",
        "2026-04-14-arxie-platform-prd.md",
    }

    plans_dir = repo_root / "docs" / "plans"
    actual = {path.name for path in plans_dir.glob("*.md")}

    assert expected.issubset(actual)
