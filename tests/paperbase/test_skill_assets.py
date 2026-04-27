from __future__ import annotations

from pathlib import Path


def test_paperbase_skills_and_runbooks_exist() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    skill_paths = [
        repo_root / "skills" / "paperbase-postgres" / "SKILL.md",
        repo_root / "skills" / "paperbase-elasticsearch" / "SKILL.md",
        repo_root / "skills" / "paperbase-corpus-ops" / "SKILL.md",
    ]
    runbook_paths = [
        repo_root / "docs" / "runbooks" / "paperbase-ingest.md",
        repo_root / "docs" / "runbooks" / "paperbase-reindex.md",
    ]

    for path in skill_paths + runbook_paths:
        assert path.exists(), f"Missing asset: {path.relative_to(repo_root)}"

    for path in skill_paths:
        content = path.read_text(encoding="utf-8")
        assert content.startswith("---\n"), f"Expected YAML frontmatter in {path.name}"
        assert "\nname:" in content, f"Expected skill name in {path.name}"
        assert "\ndescription:" in content, f"Expected skill description in {path.name}"
