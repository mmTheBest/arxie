import type { ResearchArtifactSummary, StudySource } from "../api/client";

type ArtifactCategoryKey = "drafts" | "notes" | "code" | "outputs";

const artifactCategoryLabels: Record<ArtifactCategoryKey, string> = {
  drafts: "Drafts",
  notes: "Notes",
  code: "Code",
  outputs: "Outputs"
};

function artifactCategoryForSource(source: StudySource): ArtifactCategoryKey {
  switch (source.source_type) {
    case "draft_path":
      return "drafts";
    case "code_path":
      return "code";
    case "results_path":
      return "outputs";
    case "text":
    default:
      return "notes";
  }
}

export interface ArtifactListProps {
  studySources: StudySource[];
  activeStudySourceIds: string[];
  savedArtifacts: ResearchArtifactSummary[];
  onToggleStudySource: (sourceId: string) => void;
}

export function ArtifactList({
  studySources,
  activeStudySourceIds,
  savedArtifacts,
  onToggleStudySource
}: ArtifactListProps) {
  const categoryKeys = Object.keys(artifactCategoryLabels) as ArtifactCategoryKey[];
  const hasArtifacts = studySources.length > 0 || savedArtifacts.length > 0;

  return (
    <div className="artifact-browser" aria-label="selectable context">
      {!hasArtifacts ? (
        <p className="muted">
          No registered project context yet. Add a note or path source to include your own work
          in this study.
        </p>
      ) : null}
      {categoryKeys.map((categoryKey) => {
        const categorySources = studySources.filter(
          (source) => artifactCategoryForSource(source) === categoryKey
        );
        const categorySavedArtifacts = categoryKey === "outputs" ? savedArtifacts : [];
        const hasCategoryItems =
          categorySources.length > 0 || categorySavedArtifacts.length > 0;

        return (
          <section key={categoryKey} className="artifact-category">
            <div className="artifact-category-title">
              <h3>{artifactCategoryLabels[categoryKey]}</h3>
              <small>{categorySources.length + categorySavedArtifacts.length}</small>
            </div>
            {categorySources.length > 0 ? (
              <p className="artifact-section-label">Attach to next chat</p>
            ) : null}
            {hasCategoryItems ? (
              <ul className="artifact-tree">
                {categorySources.map((source) => (
                  <li key={source.id}>
                    <label className="artifact-selector">
                      <input
                        type="checkbox"
                        checked={activeStudySourceIds.includes(source.id)}
                        onChange={() => onToggleStudySource(source.id)}
                        aria-label={`Attach ${source.title} to next chat`}
                      />
                      <span>
                        <strong>{source.source_type}</strong>
                        <span>{source.title}</span>
                        {source.is_stale ? <small>stale</small> : null}
                        {source.source_size_bytes ? (
                          <small>{Math.ceil(source.source_size_bytes / 1024)} KB</small>
                        ) : null}
                      </span>
                    </label>
                  </li>
                ))}
                {categorySavedArtifacts.map((artifact) => (
                  <li key={artifact.id} className="saved-artifact-display">
                    <strong>saved output</strong>
                    <span>{artifact.saved_title || artifact.title}</span>
                    <small>display only</small>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="muted artifact-category-empty">No items yet.</p>
            )}
          </section>
        );
      })}
    </div>
  );
}
