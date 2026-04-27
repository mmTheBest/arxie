from __future__ import annotations

from pathlib import Path

from paperbase.schemas.extraction import (
    DatasetExtraction,
    EvidenceSpanPayload,
    FindingExtraction,
    MethodExtraction,
    ResultExtraction,
)


def test_extraction_schema_models_capture_structured_outputs() -> None:
    evidence = EvidenceSpanPayload(
        target_type="result_row",
        quote_text="We improve AUROC from 0.84 to 0.91 on the held-out set.",
        page_number=4,
    )
    dataset = DatasetExtraction(display_name="scRegNetBench", evidence_spans=[evidence])
    method = MethodExtraction(display_name="scLong", evidence_spans=[evidence])
    result = ResultExtraction(
        dataset_name="scRegNetBench",
        method_name="scLong",
        metric_name="AUROC",
        value_numeric=0.91,
        evidence_spans=[evidence],
    )
    finding = FindingExtraction(statement="scLong improves AUROC on scRegNetBench.", evidence_spans=[evidence])

    assert dataset.display_name == "scRegNetBench"
    assert method.display_name == "scLong"
    assert result.metric_name == "AUROC"
    assert finding.evidence_spans[0].quote_text.startswith("We improve AUROC")


def test_migration_scaffold_exists() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    migrations_dir = repo_root / "src" / "paperbase" / "db" / "migrations"

    assert (repo_root / "alembic.ini").exists()
    assert (migrations_dir / "env.py").exists()
    assert (migrations_dir / "script.py.mako").exists()
    version_files = list((migrations_dir / "versions").glob("*.py"))
    assert version_files
