from __future__ import annotations

from paperbase.extract.contracts import StructuredExtractionBundle
from paperbase.schemas.extraction import (
    DatasetExtraction,
    EvidenceSpanPayload,
    GlossaryTermExtraction,
    MethodExtraction,
    MetricExtraction,
)


def test_build_extraction_messages_includes_schema_and_paper_text() -> None:
    from paperbase.extract.prompts import build_extraction_messages

    messages = build_extraction_messages(
        paper_text="Methods\nWe evaluate scLong on scRegNetBench using AUROC.",
        schema_payload={
            "domain": "scRegNet",
            "targets": ["datasets", "methods", "metrics", "glossary_terms"],
        },
    )

    assert len(messages) == 2
    assert "evidence spans" in messages[0].content.lower()
    assert "scRegNet" in messages[1].content
    assert "glossary_terms" in messages[1].content
    assert "scLong" in messages[1].content


def test_openai_extraction_client_returns_structured_bundle_from_llm() -> None:
    from paperbase.extract.client import OpenAIExtractionClient

    evidence = EvidenceSpanPayload(
        target_type="result_row",
        quote_text="scLong improves AUROC on scRegNetBench.",
        page_number=3,
    )
    expected_bundle = StructuredExtractionBundle(
        datasets=[DatasetExtraction(display_name="scRegNetBench", evidence_spans=[evidence])],
        methods=[MethodExtraction(display_name="scLong", evidence_spans=[evidence])],
        metrics=[MetricExtraction(display_name="AUROC", evidence_spans=[evidence])],
        glossary_terms=[
            GlossaryTermExtraction(
                term="AUROC",
                definition="Area under the receiver operating characteristic curve.",
                evidence_spans=[evidence],
            )
        ],
    )

    class FakeRunnable:
        def __init__(self, response: StructuredExtractionBundle) -> None:
            self.response = response
            self.messages: list[object] = []

        def invoke(self, messages: object) -> StructuredExtractionBundle:
            self.messages.append(messages)
            return self.response

    class FakeLLM:
        def __init__(self, response: StructuredExtractionBundle) -> None:
            self.response = response
            self.calls: list[tuple[object, str, bool | None]] = []
            self.runnable = FakeRunnable(response)

        def with_structured_output(
            self,
            schema: object,
            *,
            method: str = "json_schema",
            strict: bool | None = None,
        ) -> FakeRunnable:
            self.calls.append((schema, method, strict))
            return self.runnable

    llm = FakeLLM(expected_bundle)
    client = OpenAIExtractionClient(llm=llm, model="gpt-test")

    bundle = client.extract(
        paper_text="Methods\nWe evaluate scLong on scRegNetBench using AUROC.",
        schema_payload={"domain": "scRegNet"},
    )

    assert bundle == expected_bundle
    assert llm.calls == [(StructuredExtractionBundle, "function_calling", None)]
    assert len(llm.runnable.messages) == 1
