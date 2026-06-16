import { useEffect, useRef, type FormEvent } from "react";

import type { ArtifactFolderListing, StudySource } from "../api/client";

export interface PendingArtifactFileItem {
  id: string;
  displayPath: string;
  sourceType: StudySource["source_type"];
}

export interface ArtifactIntakeProps {
  disabled: boolean;
  artifactFolderPath: string;
  artifactFolderListing: ArtifactFolderListing | null;
  selectedArtifactFilePaths: string[];
  pendingArtifactFiles: PendingArtifactFileItem[];
  onArtifactFolderPathChange: (path: string) => void;
  onBrowseArtifactFolder: (event: FormEvent<HTMLFormElement>) => void;
  onOpenArtifactSubfolder: (relativePath: string) => void;
  onToggleArtifactFolderFile: (path: string) => void;
  onRegisterSelectedArtifactFiles: () => void;
  onArtifactFileSelection: (files: FileList | null) => void;
  onRegisterPendingArtifactFiles: () => void;
}

export function ArtifactIntake({
  disabled,
  artifactFolderPath,
  artifactFolderListing,
  selectedArtifactFilePaths,
  pendingArtifactFiles,
  onArtifactFolderPathChange,
  onBrowseArtifactFolder,
  onOpenArtifactSubfolder,
  onToggleArtifactFolderFile,
  onRegisterSelectedArtifactFiles,
  onArtifactFileSelection,
  onRegisterPendingArtifactFiles
}: ArtifactIntakeProps) {
  const artifactFileInputRef = useRef<HTMLInputElement>(null);
  const artifactDirectoryInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    artifactDirectoryInputRef.current?.setAttribute("webkitdirectory", "");
    artifactDirectoryInputRef.current?.setAttribute("directory", "");
  }, []);

  return (
    <>
      <form className="artifact-folder-form" onSubmit={onBrowseArtifactFolder}>
        <label>
          Local folder
          <input
            value={artifactFolderPath}
            onChange={(event) => onArtifactFolderPathChange(event.target.value)}
            disabled={disabled}
            placeholder="/path/to/project"
          />
        </label>
        <button type="submit" disabled={disabled}>
          Open folder
        </button>
      </form>
      <div className="artifact-picker-actions">
        <button
          type="button"
          onClick={() => artifactFileInputRef.current?.click()}
          disabled={disabled}
        >
          Select files
        </button>
        <button
          type="button"
          onClick={() => artifactDirectoryInputRef.current?.click()}
          disabled={disabled}
        >
          Select folder files
        </button>
      </div>
      <input
        ref={artifactFileInputRef}
        className="visually-hidden"
        type="file"
        multiple
        onChange={(event) => onArtifactFileSelection(event.currentTarget.files)}
      />
      <input
        ref={artifactDirectoryInputRef}
        className="visually-hidden"
        type="file"
        multiple
        onChange={(event) => onArtifactFileSelection(event.currentTarget.files)}
      />
      {artifactFolderListing ? (
        <section className="artifact-folder-panel" aria-label="Local artifact folder">
          <div className="artifact-folder-heading">
            <h3>{artifactFolderListing.relative_path || "Folder root"}</h3>
            {artifactFolderListing.relative_path ? (
              <button
                type="button"
                onClick={() =>
                  onOpenArtifactSubfolder(
                    artifactFolderListing.relative_path.split("/").slice(0, -1).join("/")
                  )
                }
              >
                Up
              </button>
            ) : null}
          </div>
          <ul className="artifact-folder-tree">
            {artifactFolderListing.entries.map((entry) => (
              <li key={entry.path}>
                {entry.entry_type === "directory" ? (
                  <button
                    type="button"
                    className="artifact-folder-entry"
                    onClick={() => onOpenArtifactSubfolder(entry.relative_path)}
                  >
                    <strong>folder</strong>
                    <span>{entry.name}</span>
                  </button>
                ) : (
                  <label className="artifact-selector artifact-folder-entry">
                    <input
                      type="checkbox"
                      checked={selectedArtifactFilePaths.includes(entry.path)}
                      onChange={() => onToggleArtifactFolderFile(entry.path)}
                      aria-label={`Register ${entry.name} from My Artifacts`}
                    />
                    <span>
                      <strong>{entry.source_type}</strong>
                      <span>{entry.name}</span>
                    </span>
                  </label>
                )}
              </li>
            ))}
          </ul>
          {artifactFolderListing.entries.length === 0 ? (
            <p className="muted">No supported artifact files in this folder.</p>
          ) : null}
          <button
            type="button"
            onClick={onRegisterSelectedArtifactFiles}
            disabled={selectedArtifactFilePaths.length === 0}
          >
            Register selected
          </button>
          {artifactFolderListing.ignored_count > 0 || artifactFolderListing.truncated ? (
            <p className="muted">
              {artifactFolderListing.ignored_count} ignored
              {artifactFolderListing.truncated ? "; list truncated" : ""}
            </p>
          ) : null}
        </section>
      ) : null}
      {pendingArtifactFiles.length > 0 ? (
        <section className="pending-artifact-files" aria-label="Pending selected files">
          <p className="artifact-section-label">Selected files</p>
          <ul className="artifact-tree">
            {pendingArtifactFiles.map((pendingFile) => (
              <li key={pendingFile.id}>
                <strong>{pendingFile.sourceType}</strong>
                <span>{pendingFile.displayPath}</span>
              </li>
            ))}
          </ul>
          <button type="button" onClick={onRegisterPendingArtifactFiles}>
            Register selected
          </button>
        </section>
      ) : null}
    </>
  );
}
