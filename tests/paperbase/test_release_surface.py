from __future__ import annotations

from pathlib import Path


def test_env_example_exposes_paperbase_runtime_configuration() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env_example = (repo_root / ".env.example").read_text(encoding="utf-8")

    expected_keys = [
        "PAPERBASE_DATABASE_URL=",
        "PAPERBASE_ELASTICSEARCH_URL=",
        "PAPERBASE_REQUIRE_SEARCH_BACKEND=",
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
        "PAPERBASE_UPLOAD_STAGING_DIR=",
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


def test_local_elasticsearch_heap_is_sized_for_single_user_launches() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    elasticsearch_env = (
        repo_root / "infra" / "env" / "elasticsearch.env"
    ).read_text(encoding="utf-8")

    assert "ES_JAVA_OPTS=-Xms512m -Xmx512m" in elasticsearch_env


def test_local_api_and_worker_do_not_hard_depend_on_elasticsearch_service() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    compose_text = (repo_root / "infra" / "docker-compose.paperbase.yml").read_text(encoding="utf-8")

    api_block = compose_text.split("  paperbase-api:\n", maxsplit=1)[1].split("\n  paperbase-worker:\n", maxsplit=1)[0]
    worker_block = compose_text.split("  paperbase-worker:\n", maxsplit=1)[1].split("\n  postgres:\n", maxsplit=1)[0]

    assert "elasticsearch:" not in api_block
    assert "elasticsearch:" not in worker_block


def test_pyproject_exposes_local_launcher_entrypoint() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pyproject_text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")

    assert 'arxie-local = "paperbase.launcher.cli:main"' in pyproject_text


def test_pyproject_pins_production_langchain_stack() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pyproject_text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")

    for dependency in [
        '"langchain==1.2.10"',
        '"langchain-core==1.2.16"',
        '"langchain-openai==1.1.10"',
        '"langchain-community==0.4.1"',
        '"langsmith==0.7.7"',
    ]:
        assert dependency in pyproject_text


def test_pyproject_does_not_pull_sentence_transformers_into_base_runtime() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pyproject_text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")

    dependencies_block = pyproject_text.split("[project.optional-dependencies]", maxsplit=1)[0]

    assert '"sentence-transformers>=' not in dependencies_block
    assert "local-embeddings = [" in pyproject_text


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
