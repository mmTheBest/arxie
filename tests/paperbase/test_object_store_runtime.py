from __future__ import annotations

import os
import time
from pathlib import Path

from sqlalchemy import select

from paperbase.db.bootstrap import initialize_database
from paperbase.db.models import PaperFile
from paperbase.db.session import make_session_factory
from paperbase.ingest.local_library import import_local_pdf_directory
from paperbase.ingest.models import CanonicalPaperSeed
from paperbase.ingest.provider_identifiers import IdentifierInput, ingest_provider_identifiers
from paperbase.storage import StorageResolver


class FakeObjectStore:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def put_file(self, *, key: str, source_path: Path, content_type: str | None = None) -> str:
        del content_type
        self.files[key] = source_path.read_bytes()
        return f"s3://paperbase/{key}"

    def put_bytes(self, *, key: str, content: bytes, content_type: str | None = None) -> str:
        del content_type
        self.files[key] = content
        return f"s3://paperbase/{key}"

    def download_to_path(self, *, storage_uri: str, destination_path: Path) -> Path:
        key = storage_uri.removeprefix("s3://paperbase/")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(self.files[key])
        return destination_path


class FakeProviderResolver:
    def __init__(self, seed: CanonicalPaperSeed) -> None:
        self.seed = seed

    def fetch_identifier(self, *, kind: str, value: str) -> CanonicalPaperSeed:
        assert kind == "doi"
        assert value == self.seed.doi
        return self.seed


def _write_pdf(path: Path) -> None:
    path.write_bytes(b"%PDF-1.4\n%stub pdf\n")


def test_import_local_pdf_directory_uploads_files_to_object_store(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "SamplePapers"
    corpus_dir.mkdir()
    _write_pdf(corpus_dir / "Alpha Paper.pdf")

    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")
    object_store = FakeObjectStore()

    import_local_pdf_directory(
        source_dir=corpus_dir,
        session_factory=session_factory,
        object_store=object_store,
    )

    with session_factory() as session:
        files = session.execute(select(PaperFile)).scalars().all()

    assert len(files) == 1
    assert files[0].storage_uri.startswith("s3://paperbase/papers/")
    assert len(object_store.files) == 1


def test_provider_identifier_ingest_uploads_remote_pdf_to_object_store(tmp_path: Path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")
    object_store = FakeObjectStore()
    pdf_bytes = b"%PDF-1.4\n%remote provider pdf\n"

    resolver = FakeProviderResolver(
        CanonicalPaperSeed(
            provider="crossref",
            external_id="10.1038/example",
            canonical_title="Provider Paper",
            doi="10.1038/example",
            pdf_url="https://example.org/provider.pdf",
            authors=["Alice Smith"],
        )
    )

    ingest_provider_identifiers(
        identifiers=[IdentifierInput(kind="doi", value="10.1038/example")],
        session_factory=session_factory,
        resolver=resolver,
        object_store=object_store,
        pdf_fetcher=lambda url: pdf_bytes,
    )

    with session_factory() as session:
        files = session.execute(select(PaperFile)).scalars().all()

    assert len(files) == 1
    assert files[0].storage_uri.startswith("s3://paperbase/papers/")
    assert list(object_store.files.values()) == [pdf_bytes]


def test_storage_resolver_can_materialize_object_store_uri_and_prune_cache(tmp_path: Path) -> None:
    object_store = FakeObjectStore()
    storage_uri = object_store.put_bytes(key="papers/paper-1/source.pdf", content=b"%PDF-1.4\n%stored\n")
    cache_dir = tmp_path / "downloads"
    cache_dir.mkdir()
    stale_file = cache_dir / "stale.pdf"
    stale_file.write_bytes(b"old")
    stale_at = time.time() - 3600
    os.utime(stale_file, (stale_at, stale_at))

    resolver = StorageResolver(
        object_store=object_store,
        cache_dir=cache_dir,
        cache_ttl_seconds=60,
    )

    resolved_path = resolver.resolve(storage_uri)
    resolver.cleanup_cache()

    assert resolved_path.exists()
    assert resolved_path.read_bytes() == b"%PDF-1.4\n%stored\n"
    assert not stale_file.exists()
