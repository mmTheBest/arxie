from __future__ import annotations

from paperbase.db.models import Base


def test_paperbase_metadata_contains_core_tables() -> None:
    expected_tables = {
        "papers",
        "paper_sources",
        "paper_files",
        "sections",
        "chunks",
        "figures",
        "glossary_terms",
        "datasets",
        "methods",
        "metrics",
        "result_rows",
        "findings",
        "engineering_tricks",
        "evidence_spans",
        "extraction_runs",
        "extraction_profiles",
        "collections",
        "collection_papers",
        "annotations",
    }

    assert expected_tables.issubset(set(Base.metadata.tables))


def test_schema_tracks_expandable_ownership_and_provenance_fields() -> None:
    collections = Base.metadata.tables["collections"]
    extraction_runs = Base.metadata.tables["extraction_runs"]
    annotations = Base.metadata.tables["annotations"]
    papers = Base.metadata.tables["papers"]

    assert "owner_id" in collections.c
    assert "scope_type" in collections.c

    assert "provider" in papers.c
    assert "external_id" in papers.c

    assert "model_name" in extraction_runs.c
    assert "prompt_version" in extraction_runs.c
    assert "schema_version" in extraction_runs.c

    assert "author_id" in annotations.c
    assert "target_type" in annotations.c
    assert "target_id" in annotations.c
    assert "body" in annotations.c
