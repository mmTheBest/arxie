from __future__ import annotations

from paperbase.db.models import Base


def test_paperbase_metadata_contains_core_tables() -> None:
    expected_tables = {
        "papers",
        "venues",
        "authors",
        "paper_authors",
        "tags",
        "paper_tags",
        "paper_sources",
        "paper_files",
        "sections",
        "chunks",
        "figures",
        "tables",
        "glossary_terms",
        "datasets",
        "methods",
        "metrics",
        "result_rows",
        "findings",
        "limitations",
        "engineering_tricks",
        "research_design_elements",
        "evidence_spans",
        "extraction_runs",
        "extraction_profiles",
        "collections",
        "collection_papers",
        "workspaces",
        "research_threads",
        "research_messages",
        "research_artifacts",
        "paper_research_labels",
        "annotations",
    }

    assert expected_tables.issubset(set(Base.metadata.tables))


def test_schema_tracks_expandable_ownership_and_provenance_fields() -> None:
    collections = Base.metadata.tables["collections"]
    workspaces = Base.metadata.tables["workspaces"]
    research_threads = Base.metadata.tables["research_threads"]
    research_messages = Base.metadata.tables["research_messages"]
    research_artifacts = Base.metadata.tables["research_artifacts"]
    paper_research_labels = Base.metadata.tables["paper_research_labels"]
    extraction_runs = Base.metadata.tables["extraction_runs"]
    annotations = Base.metadata.tables["annotations"]
    papers = Base.metadata.tables["papers"]
    venues = Base.metadata.tables["venues"]
    authors = Base.metadata.tables["authors"]
    paper_authors = Base.metadata.tables["paper_authors"]
    paper_tags = Base.metadata.tables["paper_tags"]
    limitations = Base.metadata.tables["limitations"]
    research_design_elements = Base.metadata.tables["research_design_elements"]
    tables = Base.metadata.tables["tables"]

    assert "owner_id" in collections.c
    assert "scope_type" in collections.c
    assert "owner_id" in workspaces.c
    assert "collection_id" in workspaces.c
    assert "saved_query" in workspaces.c
    assert "focus_note" in workspaces.c
    assert "active_filters_json" in workspaces.c
    assert "pinned_paper_ids_json" in workspaces.c
    assert "collection_id" in research_threads.c
    assert "selected_paper_ids_json" in research_threads.c
    assert "thread_id" in research_messages.c
    assert "artifact_id" in research_messages.c
    assert "artifact_type" in research_artifacts.c
    assert "evidence_payload_json" in research_artifacts.c
    assert "user_label" in paper_research_labels.c
    assert "inferred_signals_json" in paper_research_labels.c

    assert "provider" in papers.c
    assert "external_id" in papers.c
    assert "venue_id" in papers.c
    assert "normalized_name" in venues.c
    assert "display_name" in venues.c

    assert "model_name" in extraction_runs.c
    assert "prompt_version" in extraction_runs.c
    assert "schema_version" in extraction_runs.c

    assert "author_id" in annotations.c
    assert "target_type" in annotations.c
    assert "target_id" in annotations.c
    assert "body" in annotations.c
    assert "display_name" in authors.c
    assert "paper_id" in paper_authors.c
    assert "tag_id" in paper_tags.c
    assert "statement" in limitations.c
    assert "element_type" in research_design_elements.c
    assert "title" in research_design_elements.c
    assert "description" in research_design_elements.c
    assert "metadata_json" in research_design_elements.c
    assert "caption" in tables.c
    assert "structured_payload_json" in tables.c
