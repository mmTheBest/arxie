"""Quality harnesses for Paperbase research-agent outputs."""

from __future__ import annotations

from typing import Any


def validate_research_output(
    *,
    context: dict[str, Any],
    output_payload: dict[str, Any],
    readiness_warnings: list[str],
    forced_status: str | None = None,
) -> dict[str, Any]:
    papers = [paper for paper in context.get("papers", []) if isinstance(paper, dict)]
    references = [
        reference
        for reference in output_payload.get("evidence_references", [])
        if isinstance(reference, dict)
    ]
    readiness_blockers = list(output_payload.get("readiness_blockers") or [])
    missing_evidence: list[str] = []
    unsupported_claims: list[str] = []

    if not papers:
        missing_evidence.append("No paper evidence was available.")
    if papers and not references and forced_status != "blocked":
        missing_evidence.append("No explicit evidence references were provided.")
    if output_payload.get("summary") and not papers:
        unsupported_claims.append(str(output_payload["summary"]))

    if forced_status == "blocked" or readiness_blockers:
        harness_status = "blocked"
    elif missing_evidence or unsupported_claims:
        harness_status = "needs_attention"
    else:
        harness_status = "passed"

    return {
        "harness_status": harness_status,
        "missing_evidence": missing_evidence,
        "unsupported_claims": unsupported_claims,
        "readiness_blockers": readiness_blockers,
        "readiness_warnings": readiness_warnings,
        "evidence_reference_count": len(references),
        "evidence_paper_count": len(papers),
        "source_context_count": len([source for source in context.get("sources", []) if isinstance(source, dict)]),
    }
