from __future__ import annotations

from fastapi.testclient import TestClient

from ra.api import create_app
from ra.proposal import (
    ClaimEvidenceLabel,
    ProposalAssembler,
    ProposalExporter,
    ProposalExportFormat,
    ProposalStage,
    ProposalStageEngine,
)


class _StubRetriever:
    async def search(self, query: str, limit: int, sources: tuple[str, ...]):  # noqa: ARG002
        return []

    async def get_paper(self, identifier: str):  # noqa: ARG002
        return None

    async def close(self) -> None:
        return None


def _mk_app():
    return create_app(retriever_factory=lambda: _StubRetriever())


def _build_completed_state():
    engine = ProposalStageEngine()
    state = engine.create_initial_state()

    updates = {
        ProposalStage.IDEA_INTAKE: {
            "problem": "Citation traceability is inconsistent.",
            "target_population": "Computational social scientists.",
            "mechanism": "Stage-gated drafting with evidence mapping.",
            "expected_outcome": "Higher traceability confidence.",
        },
        ProposalStage.LOGIC_REFINEMENT: {
            "problem_gap_chain": "Current tools hide evidence provenance.",
            "core_assumptions": "Researchers prefer transparent drafts.",
            "testable_hypothesis": "Provenance tags improve trust.",
        },
        ProposalStage.EVIDENCE_MAPPING: {
            "supporting_evidence": [
                {
                    "paper_id": "p1",
                    "title": "Transparent citations improve trust",
                    "provenance_link": "https://doi.org/10.1000/xyz123",
                }
            ],
            "contradicting_evidence": [
                {
                    "paper_id": "p2",
                    "title": "Citation UX can overwhelm users",
                    "provenance_link": "https://example.org/paper2",
                }
            ],
            "landscape_summary": "Evidence is favorable but highlights usability tradeoffs.",
        },
        ProposalStage.HYPOTHESIS_RESHAPING: {
            "candidate_hypotheses": ["h1", "h2"],
            "falsification_criteria": "No trust increase in blinded evaluation.",
            "selected_primary_hypothesis": "Evidence-tagged drafts improve reviewer trust.",
        },
        ProposalStage.DATA_FEASIBILITY_PLANNING: {
            "candidate_datasets": ["peer_review_2024"],
            "feasibility_constraints": "Limited annotation budget.",
            "selected_data_strategy": "Stratified sample across disciplines.",
        },
        ProposalStage.EXPERIMENT_ANALYSIS_DESIGN: {
            "experiment_design": "A/B compare baseline vs evidence-tagged drafts.",
            "analysis_plan": "Mixed-effects regression over trust ratings.",
            "controls_and_confounders": "Reviewer expertise, domain, and draft length.",
        },
        ProposalStage.PROPOSAL_ASSEMBLY: {
            "background_and_gap": "Provenance visibility is missing in current drafting tools.",
            "hypothesis_statement": "Evidence-backed claim labels increase trust.",
            "method_summary": "Stage-gated workflow with provenance-aware synthesis.",
            "analysis_summary": "Evaluate trust and correctness with blinded reviewers.",
            "expected_outcomes": "Higher trust and citation precision.",
            "risks_and_limitations": "Potential UI overload in dense evidence scenarios.",
        },
    }

    for stage, payload in updates.items():
        state = engine.update_stage_payload(state, stage, payload)

    return state


def _seed_session_for_export(client: TestClient, session_id: str = "session-1") -> None:
    created = client.post("/api/proposal/sessions", json={"session_id": session_id})
    assert created.status_code == 201

    version = created.json()["version"]
    state = _build_completed_state()

    for stage in ProposalStageEngine().stage_sequence:
        payload = state.stage_states[stage].payload
        updated = client.patch(
            f"/api/proposal/sessions/{session_id}/stages/{stage.value}",
            json={"expected_version": version, "payload": payload},
        )
        assert updated.status_code == 200
        version = updated.json()["version"]


def test_proposal_assembler_outputs_required_sections_with_evidence_labels() -> None:
    assembler = ProposalAssembler()

    draft = assembler.assemble(session_id="session-1", state=_build_completed_state())

    section_ids = [section.section_id for section in draft.sections]
    assert section_ids == [
        "framing",
        "evidence",
        "hypothesis",
        "method",
        "outcomes",
        "risks",
    ]

    all_claims = [claim for section in draft.sections for claim in section.claims]
    assert any(claim.label is ClaimEvidenceLabel.EVIDENCE_BACKED for claim in all_claims)

    evidence_claim = next(
        claim for claim in all_claims if claim.label is ClaimEvidenceLabel.EVIDENCE_BACKED
    )
    assert "p1" in evidence_claim.reference_ids
    assert evidence_claim.provenance_links["p1"] == "https://doi.org/10.1000/xyz123"


def test_proposal_exporter_renders_markdown_and_pdf_outputs() -> None:
    draft = ProposalAssembler().assemble(session_id="session-1", state=_build_completed_state())
    exporter = ProposalExporter()

    markdown_export = exporter.render(draft, export_format=ProposalExportFormat.MARKDOWN)
    assert markdown_export.content_type == "text/markdown"
    assert markdown_export.filename == "session-1-proposal.md"
    assert "## Framing" in markdown_export.content
    assert "[Evidence-backed]" in markdown_export.content
    assert "[p1](https://doi.org/10.1000/xyz123)" in markdown_export.content

    pdf_export = exporter.render(draft, export_format=ProposalExportFormat.PDF)
    assert pdf_export.content_type == "application/pdf"
    assert pdf_export.filename == "session-1-proposal.pdf"
    assert pdf_export.content.startswith("%PDF-1.4")
    assert "## Framing" in pdf_export.content


def test_export_proposal_session_endpoint_returns_markdown_and_pdf_contract() -> None:
    client = TestClient(_mk_app())
    _seed_session_for_export(client, session_id="session-1")

    markdown = client.get("/api/proposal/sessions/session-1/export", params={"format": "markdown"})
    assert markdown.status_code == 200
    markdown_payload = markdown.json()
    assert markdown_payload["session_id"] == "session-1"
    assert markdown_payload["format"] == "markdown"
    assert markdown_payload["content_type"] == "text/markdown"
    assert markdown_payload["filename"] == "session-1-proposal.md"
    assert "## Evidence" in markdown_payload["content"]

    pdf = client.get("/api/proposal/sessions/session-1/export", params={"format": "pdf"})
    assert pdf.status_code == 200
    pdf_payload = pdf.json()
    assert pdf_payload["format"] == "pdf"
    assert pdf_payload["content_type"] == "application/pdf"
    assert pdf_payload["filename"] == "session-1-proposal.pdf"
    assert pdf_payload["content"].startswith("%PDF-1.4")
