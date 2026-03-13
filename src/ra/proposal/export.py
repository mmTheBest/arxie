"""Proposal draft export primitives."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ra.proposal.assembler import ClaimEvidenceLabel, ProposalClaim, ProposalDraft


class ProposalExportFormat(str, Enum):
    """Supported proposal export formats."""

    MARKDOWN = "markdown"
    PDF = "pdf"


@dataclass(frozen=True, slots=True)
class ProposalExportDocument:
    """Rendered proposal export document."""

    export_format: ProposalExportFormat
    content_type: str
    filename: str
    content: str


class ProposalExporter:
    """Render proposal drafts as markdown or a lightweight PDF wrapper."""

    def render(
        self,
        draft: ProposalDraft,
        *,
        export_format: ProposalExportFormat,
    ) -> ProposalExportDocument:
        if export_format is ProposalExportFormat.PDF:
            markdown = self._render_markdown(draft)
            return ProposalExportDocument(
                export_format=ProposalExportFormat.PDF,
                content_type="application/pdf",
                filename=f"{draft.session_id}-proposal.pdf",
                content=self._wrap_markdown_as_pdf(markdown),
            )

        markdown = self._render_markdown(draft)
        return ProposalExportDocument(
            export_format=ProposalExportFormat.MARKDOWN,
            content_type="text/markdown",
            filename=f"{draft.session_id}-proposal.md",
            content=markdown,
        )

    def _render_markdown(self, draft: ProposalDraft) -> str:
        lines: list[str] = [
            "# Proposal Draft",
            "",
            f"Session: `{draft.session_id}`",
            "",
        ]

        for section in draft.sections:
            lines.append(f"## {section.title}")
            lines.append(section.summary)
            lines.append("")

            if section.claims:
                lines.append("### Claims")
                for claim in section.claims:
                    label = (
                        "Evidence-backed"
                        if claim.label is ClaimEvidenceLabel.EVIDENCE_BACKED
                        else "Speculative"
                    )
                    claim_line = f"- [{label}] {claim.text}"
                    refs = self._render_claim_references(claim)
                    if refs:
                        claim_line = f"{claim_line} (Refs: {refs})"
                    lines.append(claim_line)
                lines.append("")

        if draft.references:
            lines.append("## References")
            for reference in draft.references:
                if reference.provenance_link:
                    lines.append(
                        f"- [{reference.reference_id}]({reference.provenance_link}) "
                        f"{reference.title}"
                    )
                else:
                    lines.append(f"- {reference.reference_id}: {reference.title}")
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _render_claim_references(claim: ProposalClaim) -> str:
        if not claim.reference_ids:
            return ""

        references: list[str] = []
        provenance_links = claim.provenance_links or {}
        for reference_id in claim.reference_ids:
            link = provenance_links.get(reference_id)
            if link:
                references.append(f"[{reference_id}]({link})")
            else:
                references.append(reference_id)
        return ", ".join(references)

    @staticmethod
    def _wrap_markdown_as_pdf(markdown: str) -> str:
        """Create a deterministic text-based PDF wrapper for v0.2 export."""

        payload = markdown.replace("\r\n", "\n")
        return (
            "%PDF-1.4\n"
            "% Arxie v0.2 placeholder PDF export\n"
            "1 0 obj\n"
            "<< /Type /Catalog >>\n"
            "endobj\n"
            "% Markdown payload follows for v0.2 compatibility\n"
            f"{payload}\n"
            "%%EOF\n"
        )
