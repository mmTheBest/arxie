from __future__ import annotations

from pathlib import Path


def test_paperbase_compose_uses_env_files_for_platform_services() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    compose_path = repo_root / "infra" / "docker-compose.paperbase.yml"
    compose_text = compose_path.read_text(encoding="utf-8")

    expected_env_files = {
        "infra/env/postgres.env": "env/postgres.env",
        "infra/env/elasticsearch.env": "env/elasticsearch.env",
        "infra/env/minio.env": "env/minio.env",
        "infra/env/redis.env": "env/redis.env",
    }

    for file_path, compose_ref in expected_env_files.items():
        assert (repo_root / file_path).exists()
        assert compose_ref in compose_text

    assert "env_file:" in compose_text
