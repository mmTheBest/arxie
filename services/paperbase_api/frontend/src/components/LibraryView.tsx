import { useEffect, useRef, type FormEvent } from "react";

import type {
  CollectionExtractionRecoveryAction,
  CollectionPaperMembership,
  CollectionSummary
} from "../api/client";
import { RecoveryActionSampleList } from "./RecoveryActionSamples";

export interface LibraryViewProps {
  collections: CollectionSummary[];
  selectedLibrary: CollectionSummary | null;
  papers: CollectionPaperMembership[];
  recoveryActions: CollectionExtractionRecoveryAction[];
  onSelectCollection: (collectionId: string) => void;
  onNewLibrary: () => void;
  onUploadSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onPrepareLibrary: () => void;
  onQueueRecoveryAction: (action: CollectionExtractionRecoveryAction) => void;
}

export function LibraryView({
  collections,
  selectedLibrary,
  papers,
  recoveryActions,
  onSelectCollection,
  onNewLibrary,
  onUploadSubmit,
  onPrepareLibrary,
  onQueueRecoveryAction
}: LibraryViewProps) {
  const directoryInputRef = useRef<HTMLInputElement>(null);
  const uploadFormRef = useRef<HTMLFormElement>(null);
  const uploadTitleInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    directoryInputRef.current?.setAttribute("webkitdirectory", "");
    directoryInputRef.current?.setAttribute("directory", "");
  }, []);

  function handleNewLibraryClick() {
    onNewLibrary();
    window.requestAnimationFrame(() => {
      uploadFormRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      uploadTitleInputRef.current?.focus();
    });
  }

  const readiness = selectedLibrary
    ? `${selectedLibrary.extracted_paper_count ?? 0}/${selectedLibrary.paper_count ?? 0} evidence ready`
    : "No library selected";

  return (
    <main className="workspace-grid library-grid">
      <aside className="sidebar" aria-label="Libraries">
        <button type="button" className="action-button" onClick={handleNewLibraryClick}>
          + New Library
        </button>
        {collections.length > 0 ? (
          <ul className="plain-list">
            {collections.map((collection) => (
              <li key={collection.id}>
                <button type="button" onClick={() => onSelectCollection(collection.id)}>
                  <span>{collection.title}</span>
                  <small>{collection.extracted_paper_count ?? 0} evidence ready</small>
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="muted">No libraries yet. Add PDFs to create one.</p>
        )}
      </aside>
      <section className="library-panel">
        <div className="module-heading">
          <div>
            <p className="section-label">Library</p>
            <h2>{selectedLibrary?.title ?? "Create or select a library"}</h2>
          </div>
          <span className="status-pill">{readiness}</span>
        </div>
        <form ref={uploadFormRef} className="upload-form" onSubmit={onUploadSubmit}>
          <input
            ref={directoryInputRef}
            type="file"
            name="files"
            accept="application/pdf,.pdf"
            multiple
            aria-label="Add PDF papers"
          />
          <input ref={uploadTitleInputRef} name="collection_title" placeholder="Library title" />
          <input name="collection_description" placeholder="Description" />
          <button type="submit">Add Papers</button>
        </form>
        <div className="button-row">
          <button type="button" onClick={onPrepareLibrary} disabled={!selectedLibrary}>
            Prepare Library
          </button>
        </div>
        <RecoveryActionsPanel
          actions={recoveryActions}
          canQueue={Boolean(selectedLibrary)}
          onQueueRecoveryAction={onQueueRecoveryAction}
        />
        <PaperReadinessTable papers={papers} />
      </section>
    </main>
  );
}

export function RecoveryActionsPanel({
  actions,
  canQueue,
  onQueueRecoveryAction
}: {
  actions: CollectionExtractionRecoveryAction[];
  canQueue: boolean;
  onQueueRecoveryAction: (action: CollectionExtractionRecoveryAction) => void;
}) {
  if (actions.length === 0) {
    return null;
  }

  return (
    <section
      className="recovery-action-list"
      aria-label="Evidence recovery"
      data-contract="Paper ids are bounded by the API"
    >
      <div className="recovery-action-heading">
        <p className="section-label">Evidence recovery</p>
      </div>
      {actions.map((action) => {
        const canQueueAction = canQueue && action.can_queue_job && action.paper_ids.length > 0;
        return (
          <article key={action.action_id} className="recovery-action-item">
            <div>
              <strong>{action.label}</strong>
              <p>{action.description}</p>
              <small>
                {action.paper_count} paper{action.paper_count === 1 ? "" : "s"}
                {action.truncated ? " · first batch shown" : ""}
                {action.action_type === "review_evidence"
                  ? ` · ${action.unresolved_evidence_span_count} span${
                      action.unresolved_evidence_span_count === 1 ? "" : "s"
                    }`
                  : ""}
              </small>
              <RecoveryActionSampleList action={action} />
            </div>
            {action.can_queue_job ? (
              <button
                type="button"
                aria-label={`Queue ${action.label}`}
                onClick={() => onQueueRecoveryAction(action)}
                disabled={!canQueueAction}
              >
                Queue action
              </button>
            ) : (
              <span className="muted">Review evidence</span>
            )}
          </article>
        );
      })}
    </section>
  );
}

export function PaperReadinessTable({ papers }: { papers: CollectionPaperMembership[] }) {
  return (
    <table>
      <thead>
        <tr>
          <th>Title</th>
          <th>Parsed</th>
          <th>Evidence</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {papers.length > 0 ? (
          papers.map((membership) => (
            <tr key={membership.id}>
              <td>{membership.paper.title}</td>
              <td>{membership.is_parsed ? "Yes" : "No"}</td>
              <td>{membership.is_extracted ? "Ready" : "Missing"}</td>
              <td>
                {membership.latest_job_error ||
                  membership.latest_extraction_job_status ||
                  membership.latest_parse_job_status ||
                  "Idle"}
              </td>
            </tr>
          ))
        ) : (
          <tr>
            <td colSpan={4}>No papers loaded for this library yet.</td>
          </tr>
        )}
      </tbody>
    </table>
  );
}
