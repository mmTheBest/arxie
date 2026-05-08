from __future__ import annotations

from fastapi.testclient import TestClient

from paperbase.db.bootstrap import initialize_database
from paperbase.db.models import (
    BackgroundJob,
    Dataset,
    EngineeringTrick,
    ExtractionRun,
    Figure,
    GlossaryTerm,
    Limitation,
    Method,
    Metric,
    ResearchDesignElement,
    ResultRow,
    Section,
    TableArtifact,
)
from paperbase.db.repositories import CollectionRepository, PaperRepository
from paperbase.db.session import make_session_factory
from services.paperbase_api.app import create_app


def test_paperbase_api_exposes_collection_structured_summary(tmp_path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        paper = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id="paper-1",
            canonical_title="scLong",
            publication_year=2026,
            venue="Nature",
        )
        collection = CollectionRepository(session).create(
            owner_id="local-user",
            title="SamplePapers",
            description="scRegNet field corpus.",
        )
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=paper.id)

        dataset = Dataset(paper_id=paper.id, normalized_name="scregnetbench", display_name="scRegNetBench")
        method = Method(paper_id=paper.id, normalized_name="sclong", display_name="scLong")
        metric = Metric(paper_id=paper.id, normalized_name="auroc", display_name="AUROC")
        session.add_all(
            [
                Section(
                    paper_id=paper.id,
                    title="Methods",
                    ordinal=1,
                    page_start=1,
                    page_end=2,
                    text="Full text method section.",
                ),
                dataset,
                method,
                metric,
                GlossaryTerm(
                    paper_id=paper.id,
                    term="GRN",
                    definition="Gene regulatory network.",
                ),
                Limitation(
                    paper_id=paper.id,
                    statement="The benchmark covers a limited number of perturbation settings.",
                ),
                Figure(
                    paper_id=paper.id,
                    page_number=2,
                    figure_label="Figure 1",
                    caption="Model architecture.",
                ),
                TableArtifact(
                    paper_id=paper.id,
                    page_number=4,
                    table_label="Table 1",
                    caption="Benchmark comparison.",
                    structured_payload_json={"columns": ["Method", "AUROC"], "rows": 1},
                ),
                EngineeringTrick(
                    paper_id=paper.id,
                    title="Long-context packing",
                    description="Pack distal gene context into a long sequence window.",
                ),
                ResearchDesignElement(
                    paper_id=paper.id,
                    element_type="evaluation_protocol",
                    title="Held-out perturbation benchmark",
                    description="Evaluate on a held-out perturbation split with AUROC.",
                    metadata_json={"reproducibility_signal": "fixed split"},
                ),
                ExtractionRun(
                    paper_id=paper.id,
                    model_name="fake-extractor",
                    prompt_version="paperbase-v1",
                    schema_version="schema-v1",
                    status="completed",
                ),
                BackgroundJob(
                    job_type="collection_parse",
                    status="completed",
                    payload_json={"collection_id": collection.id},
                    result_json={"collection_id": collection.id},
                ),
                BackgroundJob(
                    job_type="collection_extract",
                    status="failed",
                    payload_json={"collection_id": collection.id},
                    result_json=None,
                    error_message="Extractor timed out.",
                ),
            ]
        )
        session.flush()
        session.add(
            ResultRow(
                paper_id=paper.id,
                dataset_id=dataset.id,
                method_id=method.id,
                metric_id=metric.id,
                value_numeric=0.91,
            )
        )
        session.commit()
        collection_id = collection.id

    client = TestClient(create_app(session_factory=session_factory))

    response = client.get(f"/api/v1/collections/{collection_id}/structured-summary")

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["collection_id"] == collection_id
    assert payload["paper_count"] == 1
    assert payload["parsed_paper_count"] == 1
    assert payload["extracted_paper_count"] == 1
    assert payload["latest_parse_job_status"] == "completed"
    assert payload["latest_extraction_job_status"] == "failed"
    assert payload["failed_job_count"] == 1
    assert payload["datasets"][0]["display_name"] == "scRegNetBench"
    assert payload["methods"][0]["display_name"] == "scLong"
    assert payload["metrics"][0]["display_name"] == "AUROC"
    assert payload["figures"][0]["figure_label"] == "Figure 1"
    assert payload["tables"][0]["table_label"] == "Table 1"
    assert payload["glossary_terms"][0]["term"] == "GRN"
    assert payload["limitations"][0]["statement"] == "The benchmark covers a limited number of perturbation settings."
    assert payload["engineering_tricks"][0]["title"] == "Long-context packing"
    assert payload["research_design_elements"][0]["element_type"] == "evaluation_protocol"
    assert payload["top_result_rows"][0]["value_numeric"] == 0.91


def test_paperbase_api_collection_summary_deduplicates_metric_aliases(tmp_path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        paper = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id="paper-1",
            canonical_title="scLong",
            publication_year=2026,
            venue="Nature",
        )
        collection = CollectionRepository(session).create(
            owner_id="local-user",
            title="SamplePapers",
            description="scRegNet field corpus.",
        )
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=paper.id)

        method = Method(paper_id=paper.id, normalized_name="sclong", display_name="scLong")
        metric_short = Metric(paper_id=paper.id, normalized_name="auroc", display_name="AUROC")
        metric_long = Metric(
            paper_id=paper.id,
            normalized_name="area-under-the-receiver-operating-characteristic-curve",
            display_name="Area Under the Receiver Operating Characteristic Curve",
        )
        session.add_all(
            [
                method,
                metric_short,
                metric_long,
                ExtractionRun(
                    paper_id=paper.id,
                    model_name="fake-extractor",
                    prompt_version="paperbase-v1",
                    schema_version="schema-v1",
                    status="completed",
                ),
            ]
        )
        session.flush()
        session.add_all(
            [
                ResultRow(
                    paper_id=paper.id,
                    method_id=method.id,
                    metric_id=metric_short.id,
                    value_numeric=0.91,
                ),
                ResultRow(
                    paper_id=paper.id,
                    method_id=method.id,
                    metric_id=metric_long.id,
                    value_numeric=0.89,
                ),
            ]
        )
        session.commit()
        collection_id = collection.id

    client = TestClient(create_app(session_factory=session_factory))

    response = client.get(f"/api/v1/collections/{collection_id}/structured-summary")

    assert response.status_code == 200
    payload = response.json()["data"]
    assert [metric["display_name"] for metric in payload["metrics"]] == ["AUROC"]
    assert payload["top_result_rows"][0]["metric"] == "AUROC"
    assert payload["top_result_rows"][1]["metric"] == "AUROC"
