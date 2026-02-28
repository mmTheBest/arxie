from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.eval.harness import EvalHarness


class _MockAgent:
    def __init__(self, responses: dict[str, dict[str, object]], usage_log_path: Path) -> None:
        self._responses = responses
        self.usage_logger = SimpleNamespace(path=usage_log_path)

    def run(self, query: str) -> str:
        payload = self._responses[query]
        raw_tool_entries = payload.get("tool_entries")
        if raw_tool_entries is None:
            raw_tool_entries = [
                {"status": int(status)}
                for status in payload.get("tool_statuses", [])
            ]
        usage_path = Path(self.usage_logger.path)
        usage_path.parent.mkdir(parents=True, exist_ok=True)
        with usage_path.open("a", encoding="utf-8") as f:
            for entry in raw_tool_entries:
                record = {
                    "endpoint": "tool:search_papers",
                    "response_time_ms": 5,
                }
                if isinstance(entry, dict):
                    record.update(entry)
                f.write(
                    json.dumps(record)
                    + "\n"
                )
        return str(payload["answer"])


def _write_dataset(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def test_load_questions_raises_for_missing_required_fields(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    _write_dataset(
        dataset_path,
        [
            {
                "query": "Who proposed transformers?",
                "difficulty_tier": "tier_1",
                "min_citations": 1,
                "max_citations": 3,
            }
        ],
    )

    harness = EvalHarness(dataset_path=dataset_path, output_dir=tmp_path / "out")
    with pytest.raises(ValueError, match="expected_keywords"):
        harness.load_questions()


def test_run_computes_metrics_and_writes_outputs(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    usage_log_path = tmp_path / "api-usage.jsonl"
    _write_dataset(
        dataset_path,
        [
            {
                "query": "Q1",
                "expected_keywords": ["attention", "transformer"],
                "difficulty_tier": "tier_1",
                "min_citations": 1,
                "max_citations": 4,
            },
            {
                "query": "Q2",
                "expected_keywords": ["retrieval"],
                "difficulty_tier": "tier_2",
                "min_citations": 1,
                "max_citations": 4,
            },
        ],
    )

    responses = {
        "Q1": {
            "answer": (
                "## Answer\n"
                "Transformers replaced recurrence and improved parallelization for long-context "
                "sequence modeling in modern NLP systems.\n\n"
                "## References\n"
                "1. https://arxiv.org/abs/1706.03762\n"
                "2. https://doi.org/10.48550/arXiv.1706.03762"
            ),
            "tool_statuses": [200, 200],
        },
        "Q2": {
            "answer": (
                "## Answer\n"
                "Short answer.\n\n"
                "## References\n"
                "1. https:///paper\n"
                "2. https://example.com/paper"
            ),
            "tool_statuses": [200, 500],
        },
    }

    times = iter([1.0, 1.2, 2.0, 2.4])

    def _clock() -> float:
        return next(times)

    harness = EvalHarness(
        dataset_path=dataset_path,
        output_dir=tmp_path / "results",
        agent_factory=lambda: _MockAgent(responses, usage_log_path),
        time_fn=_clock,
    )

    report = harness.run()

    assert report["metrics"]["citation_precision"] == pytest.approx(0.75, abs=1e-6)
    assert report["metrics"]["claim_support_ratio"] == pytest.approx(0.5, abs=1e-6)
    assert report["metrics"]["tool_success_rate"] == pytest.approx(0.75, abs=1e-6)
    assert report["metrics"]["latency_p95"] == pytest.approx(0.4, abs=1e-6)
    assert report["total_questions"] == 2

    json_path = Path(report["artifacts"]["json"])
    md_path = Path(report["artifacts"]["markdown"])
    assert json_path.exists()
    assert md_path.exists()

    written = json.loads(json_path.read_text(encoding="utf-8"))
    assert written["metrics"]["tool_success_rate"] == pytest.approx(0.75, abs=1e-6)

    markdown = md_path.read_text(encoding="utf-8")
    assert "citation_precision" in markdown
    assert "Tier Breakdown" in markdown


def test_run_counts_missing_status_without_error_as_success(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    usage_log_path = tmp_path / "api-usage.jsonl"
    _write_dataset(
        dataset_path,
        [
            {
                "query": "Q",
                "expected_keywords": ["science"],
                "difficulty_tier": "tier_1",
                "min_citations": 1,
                "max_citations": 4,
            }
        ],
    )

    responses = {
        "Q": {
            "answer": (
                "## Answer\n"
                "This answer contains enough substantive detail to exceed fifty characters "
                "for coverage checks.\n\n"
                "## References\n"
                "1. https://example.org/paper"
            ),
            "tool_entries": [
                {"endpoint": "tool:search_papers"},
                {"endpoint": "tool:get_paper_details", "status": 200, "error": ""},
                {"endpoint": "tool:get_paper_citations", "status": 200, "error": "timeout"},
                {"endpoint": "tool:get_paper_citations", "status": 500},
            ],
        }
    }

    times = iter([1.0, 1.1])

    def _clock() -> float:
        return next(times)

    harness = EvalHarness(
        dataset_path=dataset_path,
        output_dir=tmp_path / "results",
        agent_factory=lambda: _MockAgent(responses, usage_log_path),
        time_fn=_clock,
    )

    report = harness.run()
    assert report["metrics"]["tool_success_rate"] == pytest.approx(0.5, abs=1e-6)
