"""Document builders for Paperbase search/read models."""

from __future__ import annotations


def build_paper_document(
    *,
    paper_id: str,
    title: str,
    abstract: str | None = None,
    year: int | None = None,
    venue: str | None = None,
    provider: str,
    external_id: str,
    doi: str | None = None,
    arxiv_id: str | None = None,
    authors: list[str] | None = None,
    tags: list[str] | None = None,
    datasets: list[str] | None = None,
    methods: list[str] | None = None,
    metrics: list[str] | None = None,
    extraction_state: str = "unextracted",
    embedding_vector: list[float] | None = None,
) -> dict[str, object]:
    return {
        "paper_id": paper_id,
        "provider": provider,
        "external_id": external_id,
        "title": title,
        "abstract": abstract or "",
        "publication_year": year,
        "venue": venue,
        "doi": doi,
        "arxiv_id": arxiv_id,
        "authors": authors or [],
        "tags": tags or [],
        "datasets": datasets or [],
        "methods": methods or [],
        "metrics": metrics or [],
        "extraction_state": extraction_state,
        "embedding": embedding_vector or [],
    }


def build_chunk_document(
    *,
    chunk_id: str,
    paper_id: str,
    title: str,
    section_title: str | None,
    text: str,
    embedding_vector: list[float] | None = None,
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "paper_id": paper_id,
        "title": title,
        "section_title": section_title or "",
        "text": text,
        "embedding": embedding_vector or [],
    }


def build_figure_document(
    *,
    figure_id: str,
    paper_id: str,
    title: str,
    figure_label: str | None,
    caption: str | None,
    embedding_vector: list[float] | None = None,
) -> dict[str, object]:
    return {
        "figure_id": figure_id,
        "paper_id": paper_id,
        "title": title,
        "figure_label": figure_label or "",
        "caption": caption or "",
        "embedding": embedding_vector or [],
    }


def build_table_document(
    *,
    table_id: str,
    paper_id: str,
    title: str,
    table_label: str | None,
    caption: str | None,
    structured_payload: dict[str, object] | None = None,
    embedding_vector: list[float] | None = None,
) -> dict[str, object]:
    return {
        "table_id": table_id,
        "paper_id": paper_id,
        "title": title,
        "table_label": table_label or "",
        "caption": caption or "",
        "structured_payload": structured_payload or {},
        "embedding": embedding_vector or [],
    }
