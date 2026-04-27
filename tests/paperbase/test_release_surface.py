from __future__ import annotations

from pathlib import Path


def test_env_example_exposes_paperbase_runtime_configuration() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env_example = (repo_root / ".env.example").read_text(encoding="utf-8")

    expected_keys = [
        "PAPERBASE_DATABASE_URL=",
        "PAPERBASE_ELASTICSEARCH_URL=",
        "PAPERBASE_REDIS_URL=",
        "PAPERBASE_WORKER_QUEUE_BACKEND=",
        "PAPERBASE_WORKER_QUEUE_NAME=",
        "PAPERBASE_WORKER_RETRY_LIMIT=",
        "PAPERBASE_WORKER_STALE_JOB_SECONDS=",
        "PAPERBASE_OBJECT_STORE_ENDPOINT=",
        "PAPERBASE_OBJECT_STORE_BACKEND=",
        "PAPERBASE_OBJECT_STORE_BUCKET=",
        "PAPERBASE_OBJECT_STORE_ACCESS_KEY=",
        "PAPERBASE_OBJECT_STORE_SECRET_KEY=",
        "PAPERBASE_OBJECT_STORE_LOCAL_ROOT=",
        "PAPERBASE_DOWNLOAD_CACHE_DIR=",
        "PAPERBASE_DOWNLOAD_CACHE_TTL_SECONDS=",
        "PAPERBASE_EMBEDDING_PROVIDER=",
        "PAPERBASE_EMBEDDING_MODEL=",
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


def test_release_surface_removes_legacy_runtime_and_prd_files() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    assert not (repo_root / "docker-compose.yml").exists()
    assert not (repo_root / "docs" / "PRD-v0.2.md").exists()
    assert not (repo_root / "docs" / "PRE-PRD-v0.2.md").exists()


def test_initial_migration_uses_explicit_revision_steps() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    migration_text = (
        repo_root
        / "src"
        / "paperbase"
        / "db"
        / "migrations"
        / "versions"
        / "20260415_0001_initial_schema.py"
    ).read_text(encoding="utf-8")

    assert "create_all" not in migration_text
    assert "drop_all" not in migration_text
