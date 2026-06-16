import type { CollectionExtractionRecoveryAction } from "../api/client";

export function RecoveryActionSampleList({
  action
}: {
  action: CollectionExtractionRecoveryAction;
}) {
  const hasSamples = action.unresolved_evidence_span_samples.length > 0;
  if (action.action_type !== "review_evidence" || !hasSamples) {
    return null;
  }

  return (
    <ol
      className="recovery-action-samples"
      data-contract="Span samples are API-bounded for review actions"
    >
      {action.unresolved_evidence_span_samples.map((sample, index) => {
        const sampleKey = `${sample.paper_id}:${index}:${sample.target_id ?? "unresolved"}`;
        const metadata = [
          sample.page_number !== null ? `page ${sample.page_number}` : null,
          sample.target_id ? `target ${sample.target_id}` : null,
          sample.reason ? humanizeSampleReason(sample.reason) : null
        ].filter(Boolean);

        return (
          <li key={sampleKey}>
            <strong>{sample.paper_title}</strong>
            {metadata.length > 0 ? <small>{metadata.join(" · ")}</small> : null}
            {sample.quote_preview ? <p>{sample.quote_preview}</p> : null}
          </li>
        );
      })}
    </ol>
  );
}

function humanizeSampleReason(reason: string) {
  return reason.replaceAll("_", " ");
}
