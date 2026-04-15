from __future__ import annotations

from fastapi.testclient import TestClient

from paperbase.db.bootstrap import initialize_database
from paperbase.db.models import (
    Dataset,
    EngineeringTrick,
    EvidenceSpan,
    Figure,
    Method,
    Metric,
    ResultRow,
    TableArtifact,
)
from paperbase.db.repositories import CollectionRepository, PaperRepository
from paperbase.db.session import make_session_factory
from services.paperbase_api.app import create_app


def test_compare_results_can_include_evidence_and_metric_aliases(tmp_path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        collection = CollectionRepository(session).create(
            owner_id="local-user",
            title="scRegNet",
            description="Comparison corpus.",
        )

        paper_one = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id="paper-1",
            canonical_title="scLong",
            publication_year=2026,
            venue="Nature",
        )
        paper_two = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id="paper-2",
            canonical_title="scNET",
            publication_year=2025,
            venue="Nature Methods",
        )
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=paper_one.id)
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=paper_two.id)

        dataset_one = Dataset(
            paper_id=paper_one.id,
            normalized_name="scregnetbench",
            display_name="scRegNetBench",
        )
        dataset_two = Dataset(
            paper_id=paper_two.id,
            normalized_name="scregnetbench",
            display_name="scRegNetBench",
        )
        method_one = Method(paper_id=paper_one.id, normalized_name="sclong", display_name="scLong")
        method_two = Method(paper_id=paper_two.id, normalized_name="scnet", display_name="scNET")
        metric_one = Metric(paper_id=paper_one.id, normalized_name="auroc", display_name="AUROC")
        metric_two = Metric(
            paper_id=paper_two.id,
            normalized_name="area-under-the-receiver-operating-characteristic-curve",
            display_name="Area Under the Receiver Operating Characteristic Curve",
        )
        session.add_all(
            [dataset_one, dataset_two, method_one, method_two, metric_one, metric_two]
        )
        session.flush()

        result_one = ResultRow(
            paper_id=paper_one.id,
            dataset_id=dataset_one.id,
            method_id=method_one.id,
            metric_id=metric_one.id,
            value_numeric=0.91,
            value_text="0.91",
        )
        result_two = ResultRow(
            paper_id=paper_two.id,
            dataset_id=dataset_two.id,
            method_id=method_two.id,
            metric_id=metric_two.id,
            value_numeric=0.87,
            value_text="0.87",
        )
        session.add_all([result_one, result_two])
        session.flush()
        session.add_all(
            [
                EvidenceSpan(
                    paper_id=paper_one.id,
                    target_type="result_row",
                    target_id=result_one.id,
                    page_number=3,
                    quote_text="scLong reaches AUROC 0.91 on scRegNetBench.",
                ),
                EvidenceSpan(
                    paper_id=paper_two.id,
                    target_type="result_row",
                    target_id=result_two.id,
                    page_number=4,
                    quote_text="scNET reports AUROC 0.87 on scRegNetBench.",
                ),
            ]
        )
        session.commit()
        collection_id = collection.id

    client = TestClient(create_app(session_factory=session_factory))

    response = client.post(
        "/api/v1/compare/results",
        json={
            "collection_id": collection_id,
            "dataset": "scRegNetBench",
            "metric": "Area Under the Receiver Operating Characteristic Curve",
            "include_evidence": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()["data"]
    assert [item["paper_title"] for item in payload] == ["scLong", "scNET"]
    assert [item["metric"] for item in payload] == ["AUROC", "AUROC"]
    assert payload[0]["evidence_spans"][0]["quote_text"] == "scLong reaches AUROC 0.91 on scRegNetBench."


def test_compare_methods_summarizes_collection_slice(tmp_path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        collection = CollectionRepository(session).create(
            owner_id="local-user",
            title="scRegNet",
            description="Comparison corpus.",
        )

        paper_one = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id="paper-1",
            canonical_title="scLong",
        )
        paper_two = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id="paper-2",
            canonical_title="scNET",
        )
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=paper_one.id)
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=paper_two.id)

        dataset_one = Dataset(paper_id=paper_one.id, normalized_name="scregnetbench", display_name="scRegNetBench")
        dataset_two = Dataset(paper_id=paper_two.id, normalized_name="scregnetbench", display_name="scRegNetBench")
        method_one = Method(paper_id=paper_one.id, normalized_name="sclong", display_name="scLong")
        method_two = Method(paper_id=paper_two.id, normalized_name="scnet", display_name="scNET")
        metric_one = Metric(paper_id=paper_one.id, normalized_name="auroc", display_name="AUROC")
        metric_two = Metric(paper_id=paper_two.id, normalized_name="auroc", display_name="AUROC")
        session.add_all([dataset_one, dataset_two, method_one, method_two, metric_one, metric_two])
        session.flush()
        session.add_all(
            [
                ResultRow(
                    paper_id=paper_one.id,
                    dataset_id=dataset_one.id,
                    method_id=method_one.id,
                    metric_id=metric_one.id,
                    value_numeric=0.91,
                    value_text="0.91",
                ),
                ResultRow(
                    paper_id=paper_two.id,
                    dataset_id=dataset_two.id,
                    method_id=method_two.id,
                    metric_id=metric_two.id,
                    value_numeric=0.87,
                    value_text="0.87",
                ),
            ]
        )
        session.commit()
        collection_id = collection.id

    client = TestClient(create_app(session_factory=session_factory))

    response = client.post(
        "/api/v1/compare/methods",
        json={
            "collection_id": collection_id,
            "dataset": "scRegNetBench",
            "metric": "AUROC",
        },
    )

    assert response.status_code == 200
    payload = response.json()["data"]
    assert [item["method"] for item in payload] == ["scLong", "scNET"]
    assert payload[0]["paper_count"] == 1
    assert payload[0]["result_count"] == 1
    assert payload[0]["datasets"] == ["scRegNetBench"]
    assert payload[0]["metrics"] == ["AUROC"]
    assert payload[0]["best_result"]["paper_title"] == "scLong"
    assert payload[0]["best_result"]["value_numeric"] == 0.91


def test_compare_engineering_tricks_can_filter_by_method_family(tmp_path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        collection = CollectionRepository(session).create(
            owner_id="local-user",
            title="scRegNet",
            description="Comparison corpus.",
        )

        paper_one = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id="paper-1",
            canonical_title="scLong",
        )
        paper_two = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id="paper-2",
            canonical_title="scNET",
        )
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=paper_one.id)
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=paper_two.id)

        session.add_all(
            [
                Method(paper_id=paper_one.id, normalized_name="sclong", display_name="scLong"),
                Method(paper_id=paper_two.id, normalized_name="scnet", display_name="scNET"),
                EngineeringTrick(
                    paper_id=paper_one.id,
                    title="Long-context packing",
                    description="Pack distant gene context into one sequence window.",
                ),
                EngineeringTrick(
                    paper_id=paper_two.id,
                    title="Edge pruning",
                    description="Prune weak candidate edges before scoring.",
                ),
            ]
        )
        session.commit()
        collection_id = collection.id

    client = TestClient(create_app(session_factory=session_factory))

    response = client.post(
        "/api/v1/compare/engineering-tricks",
        json={
            "collection_id": collection_id,
            "method": "scLong",
        },
    )

    assert response.status_code == 200
    payload = response.json()["data"]
    assert len(payload) == 1
    assert payload[0]["title"] == "Long-context packing"
    assert payload[0]["paper_count"] == 1
    assert payload[0]["papers"][0]["paper_title"] == "scLong"


def test_compare_figures_and_tables_can_filter_by_collection_and_method(tmp_path) -> None:
    database_path = tmp_path / "paperbase.sqlite3"
    initialize_database(f"sqlite:///{database_path}")
    session_factory = make_session_factory(f"sqlite:///{database_path}")

    with session_factory() as session:
        collection = CollectionRepository(session).create(
            owner_id="local-user",
            title="scRegNet",
            description="Artifact corpus.",
        )

        paper_one = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id="paper-1",
            canonical_title="scLong",
        )
        paper_two = PaperRepository(session).upsert(
            provider="local_filesystem",
            external_id="paper-2",
            canonical_title="scNET",
        )
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=paper_one.id)
        CollectionRepository(session).add_paper(collection_id=collection.id, paper_id=paper_two.id)

        session.add_all(
            [
                Method(paper_id=paper_one.id, normalized_name="sclong", display_name="scLong"),
                Method(paper_id=paper_two.id, normalized_name="scnet", display_name="scNET"),
                Figure(
                    paper_id=paper_one.id,
                    page_number=2,
                    figure_label="Figure 2",
                    caption="Ablation on long-context packing.",
                ),
                Figure(
                    paper_id=paper_two.id,
                    page_number=5,
                    figure_label="Figure 4",
                    caption="Edge pruning overview.",
                ),
                TableArtifact(
                    paper_id=paper_one.id,
                    page_number=3,
                    table_label="Table 1",
                    caption="scLong benchmark table.",
                    structured_payload_json={"rows": 4},
                ),
                TableArtifact(
                    paper_id=paper_two.id,
                    page_number=6,
                    table_label="Table 3",
                    caption="scNET benchmark table.",
                    structured_payload_json={"rows": 2},
                ),
            ]
        )
        session.commit()
        collection_id = collection.id

    client = TestClient(create_app(session_factory=session_factory))

    figures_response = client.post(
        "/api/v1/compare/figures",
        json={"collection_id": collection_id, "method": "scLong"},
    )
    tables_response = client.post(
        "/api/v1/compare/tables",
        json={"collection_id": collection_id, "method": "scLong"},
    )

    assert figures_response.status_code == 200
    assert tables_response.status_code == 200
    assert figures_response.json()["data"][0]["figure_label"] == "Figure 2"
    assert figures_response.json()["data"][0]["paper_title"] == "scLong"
    assert tables_response.json()["data"][0]["table_label"] == "Table 1"
    assert tables_response.json()["data"][0]["paper_title"] == "scLong"
