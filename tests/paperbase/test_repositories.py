from __future__ import annotations

import paperbase.db.models as db_models
import paperbase.db.repositories as db_repositories
from pathlib import Path

from paperbase.db.bootstrap import initialize_database
from paperbase.db.repositories import (
    AnnotationRepository,
    CollectionRepository,
    ExtractionProfileRepository,
    PaperRepository,
)
from paperbase.db.session import make_session_factory


def test_paper_repository_upsert_deduplicates_provider_records(tmp_path: Path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        repository = PaperRepository(session)

        first = repository.upsert(
            provider="semantic_scholar",
            external_id="paper-123",
            canonical_title="Original title",
            abstract="Original abstract",
            publication_year=2024,
            venue="ACL",
            authors=["Alice Smith", "Bob Lee"],
            tags=["single-cell", "benchmark"],
        )
        second = repository.upsert(
            provider="semantic_scholar",
            external_id="paper-123",
            canonical_title="Updated title",
            abstract="Updated abstract",
            publication_year=2025,
            venue="NeurIPS",
            authors=["Alice Smith", "Carol Jones"],
            tags=["benchmark", "foundation-model"],
        )

        assert first.id == second.id
        assert second.canonical_title == "Updated title"
        assert second.publication_year == 2025

        stored = repository.get_by_provider_id("semantic_scholar", "paper-123")
        assert stored is not None
        assert stored.id == first.id
        assert stored.venue == "NeurIPS"
        assert stored.venue_id is not None
        assert repository.list_author_names(stored.id) == ["Alice Smith", "Carol Jones"]
        assert repository.list_tags(stored.id) == ["benchmark", "foundation-model"]


def test_paper_repository_normalizes_venues_across_papers(tmp_path: Path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        repository = PaperRepository(session)

        first = repository.upsert(
            provider="semantic_scholar",
            external_id="paper-venue-1",
            canonical_title="Venue paper 1",
            venue="Nature Methods",
        )
        second = repository.upsert(
            provider="semantic_scholar",
            external_id="paper-venue-2",
            canonical_title="Venue paper 2",
            venue="  nature methods  ",
        )

        assert first.venue_id is not None
        assert second.venue_id == first.venue_id
        assert second.venue == "Nature Methods"


def test_collection_and_annotation_repositories_support_curated_local_workflows(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        paper_repository = PaperRepository(session)
        profile_repository = ExtractionProfileRepository(session)
        collection_repository = CollectionRepository(session)
        annotation_repository = AnnotationRepository(session)

        paper = paper_repository.upsert(
            provider="arxiv",
            external_id="2401.12345",
            canonical_title="Benchmarking field-specific extraction",
            abstract="A study of custom extraction profiles for local paper collections.",
            publication_year=2026,
            venue="arXiv",
        )
        profile = profile_repository.create(
            owner_id="local-user",
            name="field-profile",
            description="Extract datasets, benchmark methods, engineering tricks, and experiment design.",
            schema_payload={
                "datasets": True,
                "benchmark_methods": True,
                "engineering_tricks": True,
                "experiment_design": True,
            },
        )
        collection = collection_repository.create(
            owner_id="local-user",
            title="Field study corpus",
            description="Curated local paper database for one field.",
            tags=["field-study", "benchmarking"],
            extraction_profile_id=profile.id,
        )

        membership = collection_repository.add_paper(
            collection_id=collection.id,
            paper_id=paper.id,
            membership_note="Core benchmark paper.",
        )
        duplicate_membership = collection_repository.add_paper(
            collection_id=collection.id,
            paper_id=paper.id,
            membership_note="Updated membership note.",
        )

        assert membership.id == duplicate_membership.id
        assert duplicate_membership.membership_note == "Updated membership note."

        members = collection_repository.list_papers(collection.id)
        assert [item.paper_id for item in members] == [paper.id]

        annotation = annotation_repository.create(
            author_id="local-user",
            collection_id=collection.id,
            target_type="paper",
            target_id=paper.id,
            body="Track this paper for benchmark design details.",
            tags=["review", "experiment-design"],
            status="active",
        )

        notes = annotation_repository.list_for_target(target_type="paper", target_id=paper.id)
        assert [item.id for item in notes] == [annotation.id]
        assert notes[0].collection_id == collection.id
        assert notes[0].tags_json == ["review", "experiment-design"]


def test_background_job_repository_tracks_pending_running_completed_and_failed_jobs(
    tmp_path: Path,
) -> None:
    background_job_model = getattr(db_models, "BackgroundJob", None)
    background_job_repository_cls = getattr(db_repositories, "BackgroundJobRepository", None)

    assert background_job_model is not None, "BackgroundJob model must be defined."
    assert background_job_repository_cls is not None, "BackgroundJobRepository must be defined."

    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        repository = background_job_repository_cls(session)
        queued = repository.create(
            job_type="search_reindex",
            payload_json={"index_prefix": "paperbase"},
        )

        assert queued.status == "pending"
        assert queued.attempt_count == 0

        claimed = repository.claim_next()
        assert claimed is not None
        assert claimed.id == queued.id
        assert claimed.status == "running"
        assert claimed.attempt_count == 1

        repository.mark_completed(
            claimed.id,
            result_json={"indexed": {"papers": 1, "chunks": 2, "figures": 0}},
        )

        completed = repository.get_by_id(claimed.id)
        assert completed is not None
        assert completed.status == "completed"
        assert completed.result_json == {"indexed": {"papers": 1, "chunks": 2, "figures": 0}}
        assert completed.finished_at is not None

        failed = repository.create(
            job_type="collection_extract",
            payload_json={"collection_id": "collection-123"},
        )
        repository.claim_next()
        repository.mark_failed(failed.id, error_message="backend unavailable")

        stored_failed = repository.get_by_id(failed.id)
        assert stored_failed is not None
        assert stored_failed.status == "failed"
        assert stored_failed.error_message == "backend unavailable"
        assert stored_failed.finished_at is not None
