from __future__ import annotations


def test_index_templates_define_expected_mappings() -> None:
    from paperbase.search.index_templates import (
        chunk_index_template,
        figure_index_template,
        paper_index_template,
    )

    assert "embedding" in paper_index_template()["mappings"]["properties"]
    assert "paper_id" in chunk_index_template()["mappings"]["properties"]
    assert "caption" in figure_index_template()["mappings"]["properties"]


def test_indexer_builds_documents_for_papers_chunks_and_figures() -> None:
    from paperbase.search.indexer import (
        build_chunk_document,
        build_figure_document,
        build_paper_document,
    )

    paper_doc = build_paper_document(
        paper_id="paper-1",
        title="scLong",
        abstract="Long-range gene context modeling.",
        year=2026,
        venue="Nature",
        authors=["Alice Smith", "Bob Lee"],
        tags=["scRegNet"],
    )
    chunk_doc = build_chunk_document(
        chunk_id="chunk-1",
        paper_id="paper-1",
        title="scLong",
        section_title="Methods",
        text="We evaluate AUROC on scRegNetBench.",
    )
    figure_doc = build_figure_document(
        figure_id="figure-1",
        paper_id="paper-1",
        title="scLong",
        figure_label="Figure 1",
        caption="Model architecture.",
    )

    assert paper_doc["paper_id"] == "paper-1"
    assert chunk_doc["section_title"] == "Methods"
    assert figure_doc["figure_label"] == "Figure 1"


def test_query_builder_supports_text_filters_and_vector_query() -> None:
    from paperbase.search.query_builder import build_search_query

    query = build_search_query(
        query_text="gene regulatory",
        filters={"year_gte": 2024, "venue": ["Nature"], "tags": ["scRegNet"]},
        embedding_vector=[0.1, 0.2, 0.3],
    )

    assert query["bool"]["must"][0]["multi_match"]["query"] == "gene regulatory"
    assert {"range": {"publication_year": {"gte": 2024}}} in query["bool"]["filter"]
    assert {"terms": {"venue.keyword": ["Nature"]}} in query["bool"]["filter"]
    assert query["knn"]["field"] == "embedding"
