from __future__ import annotations

from pathlib import Path


def test_env_example_exposes_paperbase_runtime_configuration() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env_example = (repo_root / ".env.example").read_text(encoding="utf-8")

    expected_keys = [
        "PAPERBASE_DATABASE_URL=",
        "PAPERBASE_ELASTICSEARCH_URL=",
        "PAPERBASE_REDIS_URL=",
        "PAPERBASE_OBJECT_STORE_ENDPOINT=",
        "PAPERBASE_OBJECT_STORE_BUCKET=",
        "PAPERBASE_API_HOST=",
        "PAPERBASE_API_PORT=",
        "PAPERBASE_WORKER_POLL_INTERVAL_SECONDS=",
    ]

    for key in expected_keys:
        assert key in env_example


def test_dockerfile_and_compose_include_runtime_services() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dockerfile_text = (repo_root / "Dockerfile").read_text(encoding="utf-8")
    compose_text = (repo_root / "infra" / "docker-compose.paperbase.yml").read_text(encoding="utf-8")

    assert "COPY services ./services" in dockerfile_text
    assert "paperbase-api" in dockerfile_text
    assert "paperbase-api:" in compose_text
    assert "paperbase-worker:" in compose_text
    assert "env/paperbase.env" in compose_text
