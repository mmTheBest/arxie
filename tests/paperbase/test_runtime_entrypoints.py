from __future__ import annotations

from dataclasses import replace

from paperbase.config import PaperbaseConfig
from services.paperbase_api import main as api_main
from services.paperbase_worker import main as worker_main


def test_api_runtime_app_skips_search_backend_when_local_search_is_disabled(monkeypatch) -> None:
    config = PaperbaseConfig(require_search_backend=False)
    captured: dict[str, object] = {}

    monkeypatch.setattr(api_main, "load_paperbase_config", lambda: config)
    monkeypatch.setattr(api_main, "make_session_factory", lambda database_url=None: object())
    monkeypatch.setattr(api_main, "build_job_queue", lambda config: object())
    monkeypatch.setattr(api_main, "build_embedding_provider", lambda config: object())

    def fake_create_app(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(api_main, "create_app", fake_create_app)

    result = api_main.create_runtime_app()

    assert result is not None
    assert captured["search_backend"] is None


def test_api_runtime_app_builds_search_backend_when_required(monkeypatch) -> None:
    config = PaperbaseConfig(require_search_backend=True, elasticsearch_url="http://search:9200")
    captured: dict[str, object] = {}

    monkeypatch.setattr(api_main, "load_paperbase_config", lambda: config)
    monkeypatch.setattr(api_main, "make_session_factory", lambda database_url=None: object())
    monkeypatch.setattr(api_main, "build_job_queue", lambda config: object())
    monkeypatch.setattr(api_main, "build_embedding_provider", lambda config: object())
    monkeypatch.setattr(api_main, "ElasticsearchSearchBackend", lambda base_url: {"base_url": base_url})

    def fake_create_app(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(api_main, "create_app", fake_create_app)

    api_main.create_runtime_app()

    assert captured["search_backend"] == {"base_url": "http://search:9200"}


def test_worker_skips_search_backend_when_local_search_is_disabled(monkeypatch) -> None:
    config = PaperbaseConfig(require_search_backend=False)
    captured: dict[str, object] = {}

    monkeypatch.setattr(worker_main, "load_paperbase_config", lambda: config)
    monkeypatch.setattr(worker_main, "make_session_factory", lambda database_url=None: object())
    monkeypatch.setattr(worker_main, "build_job_queue", lambda config: object())
    monkeypatch.setattr(worker_main, "build_object_store", lambda config: type("Store", (), {"ensure_bucket": lambda self: None})())
    monkeypatch.setattr(worker_main, "build_embedding_provider", lambda config: object())

    def fake_worker(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(worker_main, "PaperbaseWorker", fake_worker)

    worker_main.build_worker()

    assert captured["search_backend"] is None


def test_worker_builds_search_backend_when_required(monkeypatch) -> None:
    config = PaperbaseConfig(require_search_backend=True, elasticsearch_url="http://search:9200")
    captured: dict[str, object] = {}

    monkeypatch.setattr(worker_main, "load_paperbase_config", lambda: config)
    monkeypatch.setattr(worker_main, "make_session_factory", lambda database_url=None: object())
    monkeypatch.setattr(worker_main, "build_job_queue", lambda config: object())
    monkeypatch.setattr(worker_main, "build_object_store", lambda config: type("Store", (), {"ensure_bucket": lambda self: None})())
    monkeypatch.setattr(worker_main, "build_embedding_provider", lambda config: object())
    monkeypatch.setattr(worker_main, "ElasticsearchSearchBackend", lambda base_url: {"base_url": base_url})

    def fake_worker(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(worker_main, "PaperbaseWorker", fake_worker)

    worker_main.build_worker()

    assert captured["search_backend"] == {"base_url": "http://search:9200"}
