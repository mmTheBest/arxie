from __future__ import annotations

from paperbase.extract.contracts import StructuredExtractionBundle


def test_structured_extraction_bundle_accepts_sc_regnet_style_alias_fields() -> None:
    bundle = StructuredExtractionBundle.model_validate(
        {
            "datasets": [
                {
                    "dataset_name": "scRegNetBench",
                    "benchmark_role": "evaluation",
                    "modality": "scRNA-seq",
                }
            ],
            "methods": [
                {
                    "canonical_name": "scRegNet",
                    "model_family": "foundation-model-plus-graph-learning",
                    "target_task": "TF-gene link prediction",
                }
            ],
            "metrics": [
                {
                    "metric_name": "AUROC",
                }
            ],
        }
    )

    assert bundle.datasets[0].display_name == "scRegNetBench"
    assert bundle.methods[0].display_name == "scRegNet"
    assert bundle.metrics[0].display_name == "AUROC"


def test_structured_extraction_bundle_drops_malformed_scalar_entries_from_entity_lists() -> None:
    bundle = StructuredExtractionBundle.model_validate(
        {
            "datasets": [
                {"dataset_name": "scRegNetBench"},
                "results':[{",
            ],
            "methods": [
                {"canonical_name": "scRegNet"},
                None,
            ],
            "metrics": [
                {"metric_name": "AUROC"},
                "results[]:[],",
            ],
        }
    )

    assert [dataset.display_name for dataset in bundle.datasets] == ["scRegNetBench"]
    assert [method.display_name for method in bundle.methods] == ["scRegNet"]
    assert [metric.display_name for metric in bundle.metrics] == ["AUROC"]
