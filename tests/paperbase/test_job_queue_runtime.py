from __future__ import annotations

from paperbase.db.bootstrap import initialize_database
from paperbase.db.repositories import BackgroundJobRepository
from paperbase.db.session import make_session_factory
from services.paperbase_worker.runtime import PaperbaseWorker


class FakeSearchBackend:
    def __init__(self) -> None:
        self.ensure_calls: list[str] = []
        self.bulk_calls: list[str] = []

    def ensure_index(self, index_name: str, template: dict[str, object]) -> None:
        self.ensure_calls.append(index_name)

    def bulk_index(self, index_name: str, documents: list[dict[str, object]]) -> None:
        self.bulk_calls.append(index_name)

    def search(self, index_name: str, query: dict[str, object], size: int) -> list[dict[str, object]]:
        return []


class FakeJobConsumer:
    def __init__(self, job_ids: list[str]) -> None:
        self.job_ids = list(job_ids)

    def receive(self, timeout_seconds: float | None = None) -> str | None:
        del timeout_seconds
        if not self.job_ids:
            return None
        return self.job_ids.pop(0)


def test_worker_processes_dispatched_job_id_without_db_polling_loop(tmp_path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        job = BackgroundJobRepository(session).create(job_type="search_reindex", payload_json={})
        job_id = job.id

    worker = PaperbaseWorker(
        session_factory=session_factory,
        search_backend=FakeSearchBackend(),
        job_consumer=FakeJobConsumer([job_id]),
    )

    processed = worker.process_next_dispatched_job()

    assert processed == job_id
    with session_factory() as session:
        stored_job = BackgroundJobRepository(session).get_by_id(job_id)
        assert stored_job is not None
        assert stored_job.status == "completed"
        assert stored_job.attempt_count == 1
