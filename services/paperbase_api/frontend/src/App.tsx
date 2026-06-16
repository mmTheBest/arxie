import { useEffect, useMemo, useRef, useState } from "react";

import type { PendingArtifactFileItem } from "./components/ArtifactIntake";
import { LibraryView } from "./components/LibraryView";
import { SettingsModal } from "./components/SettingsModal";
import {
  StudyView,
  suggestionHasReadyModelProvider,
  suggestionHasRequiredContext,
  type ChatEntry,
  type ChatSuggestion,
  type StudyBriefDraftText,
  type StudyBriefListField
} from "./components/StudyView";
import {
  acceptStudyBriefProposal,
  browseArtifactFolder,
  createResearchThread,
  createStudy,
  createStudySource,
  currentProjectId,
  fetchCollectionStructuredSummary,
  fetchRuntimeStatus,
  fetchResearchArtifact,
  fetchResearchArtifactRun,
  fetchResearchThreadDetail,
  fetchStudyBrief,
  fetchStudyBriefProposal,
  listBackgroundJobs,
  listCollectionPapers,
  listCollections,
  listResearchThreads,
  listSavedResearchArtifacts,
  listStudies,
  listStudySources,
  postResearchMessage,
  queueCollectionExtraction,
  queueCollectionParse,
  retryResearchArtifact,
  runModelProviderSmokeTest,
  saveStudyBrief,
  uploadLocalLibraryFiles,
  type ArtifactFolderListing,
  type BackgroundJobSummary,
  type CollectionExtractionRecoveryAction,
  type CollectionPaperMembership,
  type CollectionSummary,
  type JsonRecord,
  type ModelProviderStatus,
  type ProjectDataPathStatus,
  type ResearchAgentRunSummary,
  type ResearchArtifactSummary,
  type ResearchThreadSummary,
  type RuntimeModelProviderSmokeResult,
  type StudyBrief,
  type StudyBriefContent,
  type StudyBriefItem,
  type StudyBriefProposal,
  type StudySource,
  type WorkerHeartbeatStatus,
  type WorkerModelProviderStatus,
  type WorkspaceSummary
} from "./api/client";

type PrimaryView = "study" | "library";
type PendingArtifactFile = PendingArtifactFileItem & {
  file: File;
  title: string;
};
type ThreadLoadMode = "select" | "refresh";

const ACTIVE_STATUSES = new Set(["pending", "queued", "running"]);

function isActiveStatus(status: string | null | undefined): boolean {
  return Boolean(status && ACTIVE_STATUSES.has(status));
}

function isJsonRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringifyEvidenceValue(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (isJsonRecord(value)) {
    const label = value.label || value.title || value.paper_title || value.citation;
    const detail = value.text || value.snippet || value.summary || value.content;
    return [label, detail].filter(Boolean).map(String).join(" - ") || JSON.stringify(value);
  }
  return JSON.stringify(value);
}

function evidenceArray(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map(stringifyEvidenceValue).filter(Boolean);
  }
  if (value) {
    return [stringifyEvidenceValue(value)];
  }
  return [];
}

function renderArtifactAnswer(artifact: ResearchArtifactSummary): string {
  if (isActiveStatus(artifact.status)) {
    return artifact.title;
  }
  const output = artifact.output_payload || {};
  const sections: string[] = [];
  for (const key of ["summary", "answer", "content"]) {
    const value = output[key];
    if (typeof value === "string" && value.trim()) {
      sections.push(value.trim());
      break;
    }
  }

  const recommendations = evidenceArray(output.recommendations);
  if (recommendations.length > 0) {
    sections.push(`Recommendations:\n${recommendations.map((item) => `- ${item}`).join("\n")}`);
  }

  const themes = evidenceArray(output.themes);
  if (themes.length > 0) {
    sections.push(`Themes:\n${themes.map((item) => `- ${item}`).join("\n")}`);
  }

  if (sections.length > 0) {
    return sections.join("\n\n");
  }
  if (Object.keys(output).length > 0) {
    return JSON.stringify(output, null, 2);
  }
  return isActiveStatus(artifact.status) ? artifact.title : artifact.error_message || artifact.title;
}

function renderEvidenceLines(artifact: ResearchArtifactSummary): string[] {
  const output = artifact.output_payload || {};
  const evidencePayload = artifact.evidence_payload || {};
  const evidenceLines = [
    ...evidenceArray(output.evidence_references),
    ...evidenceArray(output.evidence),
    ...evidenceArray(evidencePayload.evidence_references),
    ...evidenceArray(evidencePayload.references),
    ...evidenceArray(evidencePayload.sources),
    ...evidenceArray(evidencePayload.items),
  ];

  if (evidenceLines.length > 0) {
    return evidenceLines;
  }
  return Object.keys(evidencePayload).length > 0
    ? [JSON.stringify(evidencePayload, null, 2)]
    : [];
}

function studyBelongsToCollection(
  study: WorkspaceSummary | null | undefined,
  collectionId: string
): study is WorkspaceSummary {
  return study?.collection_id === collectionId;
}

function threadBelongsToCollection(
  thread: ResearchThreadSummary | null | undefined,
  collectionId: string
): thread is ResearchThreadSummary {
  return thread?.collection_id === collectionId;
}

const artifactFileSourceTypeByExtension: Record<string, StudySource["source_type"]> = {
  ".csv": "results_path",
  ".ipynb": "code_path",
  ".js": "code_path",
  ".json": "results_path",
  ".jsonl": "results_path",
  ".jsx": "code_path",
  ".md": "draft_path",
  ".markdown": "draft_path",
  ".py": "code_path",
  ".r": "code_path",
  ".rst": "draft_path",
  ".sh": "code_path",
  ".sql": "code_path",
  ".tex": "draft_path",
  ".ts": "code_path",
  ".tsx": "code_path",
  ".tsv": "results_path",
  ".txt": "draft_path",
  ".yaml": "results_path",
  ".yml": "results_path"
};

const artifactFileMaxBytes = 1024 * 1024;
const artifactFileMaxCount = 20;
const artifactFileMaxTotalBytes = 2 * 1024 * 1024;
const artifactSourceContentLimit = 20000;

function sourceTypeForFileName(fileName: string): StudySource["source_type"] | null {
  const lowerName = fileName.toLowerCase();
  const extensionStart = lowerName.lastIndexOf(".");
  if (extensionStart < 0) {
    return null;
  }
  return artifactFileSourceTypeByExtension[lowerName.slice(extensionStart)] ?? null;
}

function displayPathForBrowserFile(file: File): string {
  const relativePath = (file as File & { webkitRelativePath?: string }).webkitRelativePath;
  return relativePath && relativePath.trim() ? relativePath : file.name;
}

function emptyStudyBriefContent(): StudyBriefContent {
  return {
    aim: "",
    hypothesis: "",
    constraints: [],
    confirmed_decisions: [],
    open_risks: [],
    linked_source_ids: []
  };
}

function studyBriefItemsToText(items: StudyBriefItem[]): string {
  return items
    .map((item) => {
      const title = item.title.trim();
      const text = item.text.trim();
      return title && text ? `${title}: ${text}` : title || text;
    })
    .filter(Boolean)
    .join("\n");
}

function textToStudyBriefItems(value: string): StudyBriefItem[] {
  const items: StudyBriefItem[] = [];
  value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(0, 50)
    .forEach((line, index) => {
      const separatorIndex = line.indexOf(":");
      if (separatorIndex > 0) {
        const title = line.slice(0, separatorIndex).trim();
        const text = line.slice(separatorIndex + 1).trim();
        if (text) {
          items.push({
            title: title || `Item ${index + 1}`,
            text
          });
        }
        return;
      }
      items.push({ title: `Item ${index + 1}`, text: line });
    });
  return items;
}

function studyBriefTextFromContent(content: StudyBriefContent): StudyBriefDraftText {
  return {
    aim: content.aim,
    hypothesis: content.hypothesis,
    constraints: studyBriefItemsToText(content.constraints),
    confirmed_decisions: studyBriefItemsToText(content.confirmed_decisions),
    open_risks: studyBriefItemsToText(content.open_risks)
  };
}

function studyBriefContentFromTextDraft(
  draftText: StudyBriefDraftText,
  currentContent: StudyBriefContent = emptyStudyBriefContent()
): StudyBriefContent {
  return {
    ...currentContent,
    aim: draftText.aim,
    hypothesis: draftText.hypothesis,
    constraints: textToStudyBriefItems(draftText.constraints),
    confirmed_decisions: textToStudyBriefItems(draftText.confirmed_decisions),
    open_risks: textToStudyBriefItems(draftText.open_risks)
  };
}

function artifactHasStudyBriefProposal(artifact: ResearchArtifactSummary): boolean {
  if (artifact.status !== "completed" || !artifact.output_payload) {
    return false;
  }
  const payload = artifact.output_payload;
  const proposalPayload = payload.study_brief_update ?? payload.study_brief_updates;
  return (
    typeof proposalPayload === "object" &&
    proposalPayload !== null &&
    !Array.isArray(proposalPayload)
  );
}

function collectStudyBriefProposalCandidates(
  artifacts: ResearchArtifactSummary[]
): ResearchArtifactSummary[] {
  const seenArtifactIds = new Set<string>();
  return artifacts.filter((artifact) => {
    if (seenArtifactIds.has(artifact.id) || !artifactHasStudyBriefProposal(artifact)) {
      return false;
    }
    seenArtifactIds.add(artifact.id);
    return true;
  });
}

export function App() {
  const [activeView, setActiveView] = useState<PrimaryView>("study");
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [collections, setCollections] = useState<CollectionSummary[]>([]);
  const [selectedCollectionId, setSelectedCollectionId] = useState<string | null>(null);
  const [papers, setPapers] = useState<CollectionPaperMembership[]>([]);
  const [recoveryActions, setRecoveryActions] = useState<CollectionExtractionRecoveryAction[]>([]);
  const [studies, setStudies] = useState<WorkspaceSummary[]>([]);
  const [activeStudy, setActiveStudy] = useState<WorkspaceSummary | null>(null);
  const [studyBrief, setStudyBrief] = useState<StudyBrief | null>(null);
  const [studyBriefDraft, setStudyBriefDraft] = useState<StudyBriefContent>(
    emptyStudyBriefContent()
  );
  const [studyBriefDraftText, setStudyBriefDraftText] = useState<StudyBriefDraftText>(
    studyBriefTextFromContent(emptyStudyBriefContent())
  );
  const [studySources, setStudySources] = useState<StudySource[]>([]);
  const [activeStudySourceIds, setActiveStudySourceIds] = useState<string[]>([]);
  const [artifactFolderPath, setArtifactFolderPath] = useState("");
  const [artifactFolderListing, setArtifactFolderListing] =
    useState<ArtifactFolderListing | null>(null);
  const [selectedArtifactFilePaths, setSelectedArtifactFilePaths] = useState<string[]>([]);
  const [pendingArtifactFiles, setPendingArtifactFiles] = useState<PendingArtifactFile[]>([]);
  const [savedArtifacts, setSavedArtifacts] = useState<ResearchArtifactSummary[]>([]);
  const [threadArtifacts, setThreadArtifacts] = useState<ResearchArtifactSummary[]>([]);
  const [runByArtifactId, setRunByArtifactId] = useState<Record<string, ResearchAgentRunSummary>>(
    {}
  );
  const [studyBriefProposal, setStudyBriefProposal] = useState<StudyBriefProposal | null>(null);
  const [studyBriefProposalArtifactId, setStudyBriefProposalArtifactId] = useState("");
  const [studyBriefProposalError, setStudyBriefProposalError] = useState<string | null>(null);
  const [isStudyBriefProposalLoading, setIsStudyBriefProposalLoading] = useState(false);
  const [isStudyBriefProposalAccepting, setIsStudyBriefProposalAccepting] = useState(false);
  const [threads, setThreads] = useState<ResearchThreadSummary[]>([]);
  const [activeThread, setActiveThread] = useState<ResearchThreadSummary | null>(null);
  const [chatEntries, setChatEntries] = useState<ChatEntry[]>([]);
  const [chatDraft, setChatDraft] = useState("");
  const [pendingJobIds, setPendingJobIds] = useState<string[]>([]);
  const [pendingArtifactIds, setPendingArtifactIds] = useState<string[]>([]);
  const [statusMessage, setStatusMessage] = useState("Loading Paperbase libraries...");
  const [modelProviderStatus, setModelProviderStatus] = useState<ModelProviderStatus | null>(null);
  const [workerModelProviderStatus, setWorkerModelProviderStatus] =
    useState<WorkerModelProviderStatus | null>(null);
  const [workerHeartbeatStatus, setWorkerHeartbeatStatus] =
    useState<WorkerHeartbeatStatus | null>(null);
  const [projectDataPathStatus, setProjectDataPathStatus] =
    useState<ProjectDataPathStatus | null>(null);
  const [runtimeStatusError, setRuntimeStatusError] = useState<string | null>(null);
  const [isRuntimeStatusLoading, setIsRuntimeStatusLoading] = useState(false);
  const [providerSmokeStatus, setProviderSmokeStatus] =
    useState<RuntimeModelProviderSmokeResult | null>(null);
  const [providerSmokeError, setProviderSmokeError] = useState<string | null>(null);
  const [isProviderSmokeRunning, setIsProviderSmokeRunning] = useState(false);
  const selectedCollectionIdRef = useRef<string | null>(null);
  const previousSelectedCollectionIdRef = useRef<string | null>(null);
  const didAutoSelectInitialLibraryRef = useRef(false);
  const isStudyBriefDraftDirtyRef = useRef(false);
  const researchThreadLoadRequestRef = useRef(0);
  const pendingResearchThreadSelectionRef = useRef<string | null>(null);
  const requestedResearchThreadIdRef = pendingResearchThreadSelectionRef;

  useEffect(() => {
    void refreshCollections();
    void refreshStudies();
    void refreshRuntimeStatus();
  }, []);

  useEffect(() => {
    if (isSettingsOpen) {
      void refreshRuntimeStatus();
    }
  }, [isSettingsOpen]);

  useEffect(() => {
    selectedCollectionIdRef.current = selectedCollectionId;
  }, [selectedCollectionId]);

  useEffect(() => {
    if (activeThread?.id && pendingResearchThreadSelectionRef.current === null) {
      pendingResearchThreadSelectionRef.current = activeThread.id;
    }
  }, [activeThread?.id]);

  const selectedLibrary = useMemo(
    () => collections.find((collection) => collection.id === selectedCollectionId) ?? null,
    [collections, selectedCollectionId]
  );

  function selectCollection(collectionId: string | null) {
    selectedCollectionIdRef.current = collectionId;
    setRecoveryActions([]);
    setSelectedCollectionId(collectionId);
  }

  function resetStudyBriefDraft(content: StudyBriefContent) {
    isStudyBriefDraftDirtyRef.current = false;
    setStudyBriefDraft(content);
    setStudyBriefDraftText(studyBriefTextFromContent(content));
  }

  function resetStudyBriefProposal() {
    setStudyBriefProposal(null);
    setStudyBriefProposalArtifactId("");
    setStudyBriefProposalError(null);
    setIsStudyBriefProposalLoading(false);
    setIsStudyBriefProposalAccepting(false);
  }

  function applyLoadedStudyBrief(loadedBrief: StudyBrief) {
    if (!isStudyBriefDraftDirtyRef.current) {
      setStudyBrief(loadedBrief);
      resetStudyBriefDraft(loadedBrief.brief);
    }
  }

  useEffect(() => {
    if (
      !selectedCollectionId &&
      collections.length > 0 &&
      !didAutoSelectInitialLibraryRef.current
    ) {
      didAutoSelectInitialLibraryRef.current = true;
      selectCollection(collections[0].id);
    }
  }, [collections, selectedCollectionId]);

  useEffect(() => {
    const selectedLibraryId = selectedLibrary?.id ?? null;
    const libraryChanged = previousSelectedCollectionIdRef.current !== selectedLibraryId;
    previousSelectedCollectionIdRef.current = selectedLibraryId;

    if (!selectedLibrary) {
      setPapers([]);
      setRecoveryActions([]);
      setThreads([]);
      setActiveThread(null);
      requestedResearchThreadIdRef.current = null;
      setActiveStudy(null);
      setChatEntries([]);
      setChatDraft("");
      setStudyBrief(null);
      resetStudyBriefDraft(emptyStudyBriefContent());
      setStudySources([]);
      setActiveStudySourceIds([]);
      setArtifactFolderListing(null);
      setSelectedArtifactFilePaths([]);
      setPendingArtifactFiles([]);
      setSavedArtifacts([]);
      setThreadArtifacts([]);
      setRunByArtifactId({});
      resetStudyBriefProposal();
      return;
    }

    if (libraryChanged) {
      setPapers([]);
      setRecoveryActions([]);
      setThreads([]);
      setActiveThread(null);
      requestedResearchThreadIdRef.current = null;
      setActiveStudy(null);
      setChatEntries([]);
      setChatDraft("");
      setStudyBrief(null);
      resetStudyBriefDraft(emptyStudyBriefContent());
      setStudySources([]);
      setActiveStudySourceIds([]);
      setArtifactFolderListing(null);
      setSelectedArtifactFilePaths([]);
      setPendingArtifactFiles([]);
      setSavedArtifacts([]);
      setThreadArtifacts([]);
      setRunByArtifactId({});
      resetStudyBriefProposal();
    }

    const matchingStudy =
      studies.find((study) => study.collection_id === selectedLibrary.id) ?? null;
    setActiveStudy(matchingStudy);
    void refreshCollectionDetails(selectedLibrary, matchingStudy);
  }, [selectedLibrary, studies]);

  useEffect(() => {
    if (pendingJobIds.length === 0 && pendingArtifactIds.length === 0) {
      return;
    }

    const intervalId = window.setInterval(() => {
      void pollActiveWork();
    }, 2500);
    void pollActiveWork();

    return () => window.clearInterval(intervalId);
  }, [pendingJobIds, pendingArtifactIds, selectedLibrary?.id, activeThread?.id]);

  async function refreshCollections() {
    try {
      const loadedCollections = await listCollections();
      setCollections(loadedCollections);
      setStatusMessage(
        loadedCollections.length > 0
          ? "Loaded libraries from Paperbase."
          : "No research library is ready yet."
      );
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : "Unable to load libraries.");
    }
  }

  async function refreshStudies() {
    try {
      setStudies(await listStudies());
    } catch {
      setStudies([]);
    }
  }

  async function refreshRuntimeStatus() {
    setIsRuntimeStatusLoading(true);
    try {
      const runtimeStatus = await fetchRuntimeStatus();
      setModelProviderStatus(runtimeStatus.model_provider);
      setWorkerModelProviderStatus(runtimeStatus.worker_model_provider);
      setWorkerHeartbeatStatus(runtimeStatus.worker_heartbeat);
      setProjectDataPathStatus(runtimeStatus.project_data_paths);
      setRuntimeStatusError(null);
    } catch (error) {
      console.error(error);
      setRuntimeStatusError(error instanceof Error ? error.message : "Runtime status unavailable");
    } finally {
      setIsRuntimeStatusLoading(false);
    }
  }

  async function handleRunProviderSmokeTest() {
    setIsProviderSmokeRunning(true);
    setProviderSmokeError(null);
    try {
      const smokeResult = await runModelProviderSmokeTest();
      setProviderSmokeStatus(smokeResult);
      await refreshRuntimeStatus();
    } catch (error) {
      console.error(error);
      setProviderSmokeError(
        error instanceof Error ? error.message : "Provider smoke test unavailable"
      );
    } finally {
      setIsProviderSmokeRunning(false);
    }
  }

  async function refreshCollectionDetails(
    collection: CollectionSummary,
    study: WorkspaceSummary | null = activeStudy
  ) {
    try {
      const [loadedPapers, loadedSummary, loadedThreads, loadedArtifacts] = await Promise.all([
        listCollectionPapers(collection.id),
        fetchCollectionStructuredSummary(collection.id),
        listResearchThreads(collection.id),
        listSavedResearchArtifacts(collection.id)
      ]);
      if (selectedCollectionIdRef.current !== collection.id) {
        return;
      }
      setPapers(loadedPapers);
      setRecoveryActions(loadedSummary.extraction_recovery_actions ?? []);
      setThreads(loadedThreads);
      setActiveThread((currentThread) => {
        if (
          currentThread && threadBelongsToCollection(currentThread, collection.id) &&
          loadedThreads.some((thread) => thread.id === currentThread.id)
        ) {
          return currentThread;
        }
        return loadedThreads[0] ?? null;
      });
      setSavedArtifacts(loadedArtifacts);
    } catch (error) {
      if (selectedCollectionIdRef.current === collection.id) {
        setStatusMessage(error instanceof Error ? error.message : "Unable to load library detail.");
      }
    }

    if (selectedCollectionIdRef.current !== collection.id) {
      return;
    }

    if (studyBelongsToCollection(study, collection.id)) {
      try {
        const [loadedSources, loadedBrief] = await Promise.all([
          listStudySources(study.id),
          fetchStudyBrief(study.id)
        ]);
        if (selectedCollectionIdRef.current === collection.id) {
          setStudySources(loadedSources);
          applyLoadedStudyBrief(loadedBrief);
          setActiveStudySourceIds((currentIds) =>
            currentIds.filter((sourceId) =>
              loadedSources.some((source) => source.id === sourceId)
            )
          );
        }
      } catch {
        if (selectedCollectionIdRef.current === collection.id) {
          if (!isStudyBriefDraftDirtyRef.current) {
            setStudyBrief(null);
            resetStudyBriefDraft(emptyStudyBriefContent());
          }
          setStudySources([]);
          setActiveStudySourceIds([]);
        }
      }
    } else {
      if (!isStudyBriefDraftDirtyRef.current) {
        setStudyBrief(null);
        resetStudyBriefDraft(emptyStudyBriefContent());
      }
      setStudySources([]);
      setActiveStudySourceIds([]);
    }
  }

  async function pollActiveWork() {
    try {
      const [jobs] = await Promise.all([listBackgroundJobs(50), refreshCollections()]);
      const activeJobIds = new Set(
        jobs.filter((job) => isActiveStatus(job.status)).map((job) => job.id)
      );

      setPendingJobIds((currentIds) =>
        currentIds.filter((jobId) => activeJobIds.has(jobId))
      );

      const activeArtifactIds: string[] = [];
      const fetchedRuns: Record<string, ResearchAgentRunSummary> = {};
      for (const artifactId of pendingArtifactIds) {
        try {
          const artifact = await fetchResearchArtifact(artifactId);
          const run = await fetchResearchArtifactRun(artifactId).catch(() => null);
          if (run) {
            fetchedRuns[artifactId] = run;
          }
          if (isActiveStatus(artifact.status) || isActiveStatus(run?.status)) {
            activeArtifactIds.push(artifactId);
          }
        } catch {
          activeArtifactIds.push(artifactId);
        }
      }
      if (Object.keys(fetchedRuns).length > 0) {
        setRunByArtifactId((currentRuns) => ({
          ...currentRuns,
          ...fetchedRuns,
        }));
      }
      setPendingArtifactIds(activeArtifactIds);

      if (selectedLibrary) {
        await refreshCollectionDetails(selectedLibrary);
      }
      if (selectedLibrary && threadBelongsToCollection(activeThread, selectedLibrary.id)) {
        await loadResearchThread(activeThread.id, "refresh");
      }
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : "Unable to refresh active work.");
    }
  }

  async function ensureStudy(collection: CollectionSummary): Promise<WorkspaceSummary> {
    if (activeStudy && studyBelongsToCollection(activeStudy, collection.id)) {
      return activeStudy;
    }
    const existingStudy =
      studies.find((study) => studyBelongsToCollection(study, collection.id)) ?? null;
    if (existingStudy) {
      setActiveStudy(existingStudy);
      return existingStudy;
    }
    const createdStudy = await createStudy(collection);
    setStudies((currentStudies) => [...currentStudies, createdStudy]);
    if (selectedCollectionIdRef.current === collection.id) {
      setActiveStudy(createdStudy);
    }
    return createdStudy;
  }

  async function ensureThread(
    collection: CollectionSummary,
    message: string
  ): Promise<ResearchThreadSummary> {
    if (activeThread && threadBelongsToCollection(activeThread, collection.id)) {
      return activeThread;
    }
    const study =
      activeStudy && studyBelongsToCollection(activeStudy, collection.id)
        ? activeStudy
        : await ensureStudy(collection);
    const title = message.trim().slice(0, 72) || "Arxie chat";
    const createdThread = await createResearchThread(collection, title, study.id);
    if (selectedCollectionIdRef.current === collection.id) {
      setThreads((currentThreads) => [createdThread, ...currentThreads]);
      setActiveThread(createdThread);
      requestedResearchThreadIdRef.current = createdThread.id;
    }
    return createdThread;
  }

  async function loadResearchThread(threadId: string, mode: ThreadLoadMode = "select") {
    const isBackgroundLoad = mode === "refresh";
    if (isBackgroundLoad && requestedResearchThreadIdRef.current !== threadId) {
      return false;
    }
    const previousRequestedThreadId = requestedResearchThreadIdRef.current;
    const loadRequestId = isBackgroundLoad
      ? researchThreadLoadRequestRef.current
      : researchThreadLoadRequestRef.current + 1;
    let shouldRestoreRequestedThread = false;
    if (!isBackgroundLoad) {
      researchThreadLoadRequestRef.current = loadRequestId;
      requestedResearchThreadIdRef.current = threadId;
      shouldRestoreRequestedThread = true;
    }
    try {
      const detail = await fetchResearchThreadDetail(threadId);
      if (
        (!isBackgroundLoad && loadRequestId !== researchThreadLoadRequestRef.current) ||
        requestedResearchThreadIdRef.current !== threadId ||
        detail.collection_id !== selectedCollectionIdRef.current
      ) {
        return false;
      }
      const artifactRuns = await Promise.all(
        detail.artifacts.map(async (artifact) => {
          const run = await fetchResearchArtifactRun(artifact.id).catch(() => null);
          return [artifact.id, run] as const;
        })
      );
      if (
        (!isBackgroundLoad && loadRequestId !== researchThreadLoadRequestRef.current) ||
        requestedResearchThreadIdRef.current !== threadId ||
        detail.collection_id !== selectedCollectionIdRef.current
      ) {
        return false;
      }
      const isChangingThread = activeThread?.id !== detail.id;
      if (isChangingThread) {
        resetStudyBriefProposal();
      }
      setActiveThread(detail);
      setThreadArtifacts(detail.artifacts);
      setRunByArtifactId((currentRuns) => {
        const nextRuns = { ...currentRuns };
        for (const [artifactId, run] of artifactRuns) {
          if (run) {
            nextRuns[artifactId] = run;
          }
        }
        return nextRuns;
      });
      shouldRestoreRequestedThread = false;
      setChatEntries([
        ...detail.messages.map((message): ChatEntry => {
          if (message.role === "user") {
            return { id: message.id, role: "user", content: message.content };
          }
          return {
            id: message.id,
            role: "assistant",
            content: message.content,
            status: "Thread message",
            evidenceLines: [],
          };
        }),
        ...detail.artifacts.map(
          (artifact): ChatEntry => ({
            id: artifact.id,
            role: "assistant",
            content: renderArtifactAnswer(artifact),
            status: artifact.status,
            evidenceLines: renderEvidenceLines(artifact),
            artifact,
          })
        ),
      ]);
      return true;
    } finally {
      if (
        shouldRestoreRequestedThread &&
        loadRequestId === researchThreadLoadRequestRef.current &&
        requestedResearchThreadIdRef.current === threadId
      ) {
        requestedResearchThreadIdRef.current = previousRequestedThreadId;
      }
    }
  }

  function handleNewChat() {
    researchThreadLoadRequestRef.current += 1;
    requestedResearchThreadIdRef.current = null;
    setActiveThread(null);
    setChatEntries([]);
    setThreadArtifacts([]);
    setRunByArtifactId({});
    resetStudyBriefProposal();
    setChatDraft("");
    setStatusMessage("Started a new chat draft.");
  }

  async function handleSelectThread(threadId: string) {
    try {
      const loaded = await loadResearchThread(threadId);
      setStatusMessage(
        loaded ? "Loaded recent chat." : "Select that chat's library before opening it."
      );
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : "Unable to load chat.");
    }
  }

  function handleNewLibrary() {
    selectedCollectionIdRef.current = null;
    setSelectedCollectionId(null);
    setRecoveryActions([]);
    researchThreadLoadRequestRef.current += 1;
    requestedResearchThreadIdRef.current = null;
    setPapers([]);
    setThreads([]);
    setActiveThread(null);
    setActiveStudy(null);
    setChatEntries([]);
    setChatDraft("");
    setStudyBrief(null);
    resetStudyBriefDraft(emptyStudyBriefContent());
    setStudySources([]);
    setActiveStudySourceIds([]);
    setSavedArtifacts([]);
    setThreadArtifacts([]);
    setRunByArtifactId({});
    resetStudyBriefProposal();
    setActiveView("library");
    setStatusMessage("Add PDFs below to create a new library.");
  }

  function toggleStudySource(sourceId: string) {
    setActiveStudySourceIds((currentIds) =>
      currentIds.includes(sourceId)
        ? currentIds.filter((currentId) => currentId !== sourceId)
        : [...currentIds, sourceId]
    );
  }

  function handleStudyBriefFieldChange(field: "aim" | "hypothesis", value: string) {
    isStudyBriefDraftDirtyRef.current = true;
    setStudyBriefDraftText((currentText) => ({
      ...currentText,
      [field]: value
    }));
    setStudyBriefDraft((currentBrief) => ({
      ...currentBrief,
      [field]: value
    }));
  }

  function handleStudyBriefItemsChange(field: StudyBriefListField, value: string) {
    isStudyBriefDraftDirtyRef.current = true;
    setStudyBriefDraftText((currentText) => ({
      ...currentText,
      [field]: value
    }));
    setStudyBriefDraft((currentBrief) => ({
      ...currentBrief,
      [field]: textToStudyBriefItems(value)
    }));
  }

  function handleStudyBriefProposalArtifactChange(artifactId: string) {
    setStudyBriefProposalArtifactId(artifactId);
    setStudyBriefProposal(null);
    setStudyBriefProposalError(null);
  }

  async function handleStudyBriefSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedLibrary) {
      return;
    }
    const collectionId = selectedLibrary.id;

    try {
      const existingStudy =
        activeStudy && studyBelongsToCollection(activeStudy, collectionId)
          ? activeStudy
          : studies.find((study) => studyBelongsToCollection(study, collectionId)) ?? null;
      const study = await ensureStudy(selectedLibrary);
      const isExistingStudyBriefReady =
        !existingStudy || studyBrief?.workspace_id === study.id;
      if (!isExistingStudyBriefReady) {
        const loadedBrief = await fetchStudyBrief(study.id);
        if (selectedCollectionIdRef.current === collectionId) {
          setStudyBrief(loadedBrief);
          resetStudyBriefDraft(loadedBrief.brief);
          setStatusMessage("Loaded the current Study Brief. Review it before saving.");
        }
        return;
      }
      const expectedVersion =
        studyBrief?.workspace_id === study.id
          ? studyBrief.version
          : 0;
      const draftToSave = studyBriefContentFromTextDraft(
        studyBriefDraftText,
        studyBrief?.brief ?? studyBriefDraft
      );
      const savedBrief = await saveStudyBrief(study.id, draftToSave, expectedVersion);
      if (selectedCollectionIdRef.current === collectionId) {
        setStudyBrief(savedBrief);
        resetStudyBriefDraft(savedBrief.brief);
        setStatusMessage(`Saved Study Brief version ${savedBrief.version}.`);
      }
    } catch (error) {
      if (selectedCollectionIdRef.current === collectionId) {
        setStatusMessage(error instanceof Error ? error.message : "Unable to save Study Brief.");
      }
    }
  }

  async function handleLoadStudyBriefProposal(artifactId: string) {
    if (!selectedLibrary || !artifactId) {
      return;
    }
    const collectionId = selectedLibrary.id;
    setIsStudyBriefProposalLoading(true);
    setStudyBriefProposalError(null);

    try {
      const study = await ensureStudy(selectedLibrary);
      const proposal = await fetchStudyBriefProposal(study.id, artifactId);
      if (selectedCollectionIdRef.current === collectionId) {
        setStudyBriefProposal(proposal);
        setStudyBriefProposalArtifactId(proposal.artifact_id);
        isStudyBriefDraftDirtyRef.current = true;
        setStudyBriefDraft(proposal.proposed_brief);
        setStudyBriefDraftText(studyBriefTextFromContent(proposal.proposed_brief));
        setStatusMessage("Applied Study Brief proposal to the draft.");
      }
    } catch (error) {
      if (selectedCollectionIdRef.current === collectionId) {
        const message =
          error instanceof Error ? error.message : "Unable to load Study Brief proposal.";
        setStudyBriefProposalError(message);
        setStatusMessage(message);
      }
    } finally {
      if (selectedCollectionIdRef.current === collectionId) {
        setIsStudyBriefProposalLoading(false);
      }
    }
  }

  async function handleAcceptStudyBriefProposal() {
    if (!selectedLibrary || !studyBriefProposal) {
      return;
    }
    if (!proposalBelongsToCurrentCandidates()) {
      resetStudyBriefProposal();
      setStatusMessage("Reload the Study Brief proposal before accepting it.");
      return;
    }
    const collectionId = selectedLibrary.id;
    setIsStudyBriefProposalAccepting(true);
    setStudyBriefProposalError(null);

    try {
      const study = await ensureStudy(selectedLibrary);
      const currentContent = studyBriefDraft;
      const draftToAccept = studyBriefContentFromTextDraft(
        studyBriefDraftText,
        currentContent
      );
      const savedBrief = await acceptStudyBriefProposal(
        study.id,
        studyBriefProposal.artifact_id,
        draftToAccept,
        studyBriefProposal.current_version
      );
      if (selectedCollectionIdRef.current === collectionId) {
        setStudyBrief(savedBrief);
        resetStudyBriefDraft(savedBrief.brief);
        setStudyBriefProposal(null);
        setStudyBriefProposalArtifactId("");
        setStatusMessage(`Accepted Study Brief proposal as version ${savedBrief.version}.`);
      }
    } catch (error) {
      if (selectedCollectionIdRef.current === collectionId) {
        const message =
          error instanceof Error ? error.message : "Unable to accept Study Brief proposal.";
        setStudyBriefProposalError(message);
        setStatusMessage(message);
      }
    } finally {
      if (selectedCollectionIdRef.current === collectionId) {
        setIsStudyBriefProposalAccepting(false);
      }
    }
  }

  function handleSuggestionClick(suggestion: ChatSuggestion) {
    const selectedSources = studySources.filter((source) =>
      activeStudySourceIds.includes(source.id)
    );
    if (!suggestionHasRequiredContext(suggestion, selectedSources)) {
      setStatusMessage("Attach a draft or note first.");
      return;
    }
    if (!suggestionHasReadyModelProvider(suggestion, modelProviderStatus)) {
      setStatusMessage("Model setup required. Open Settings for provider setup.");
      return;
    }
    const libraryTitle = selectedLibrary ? `"${selectedLibrary.title}"` : "this library";
    const attachedSourceTitles = selectedSources.map((source) => source.title).join(", ");
    const attachedContext = attachedSourceTitles
      ? ` Use attached context: ${attachedSourceTitles}.`
      : "";
    setChatDraft(suggestion.buildPrompt(libraryTitle, attachedContext));
    setStatusMessage(`Drafted a ${suggestion.label.toLowerCase()} prompt.`);
  }

  async function handleChatSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const message = chatDraft.trim();
    if (!selectedLibrary || !message) {
      return;
    }
    const collectionId = selectedLibrary.id;

    const userEntry: ChatEntry = { id: `user-${Date.now()}`, role: "user", content: message };
    setChatEntries((currentEntries) => [...currentEntries, userEntry]);
    setChatDraft("");
    setStatusMessage("Sending message to Arxie...");

    try {
      const thread = await ensureThread(selectedLibrary, message);
      if (
        selectedCollectionIdRef.current !== collectionId ||
        !threadBelongsToCollection(thread, collectionId)
      ) {
        return;
      }
      const selectedStudySourceIds = activeStudySourceIds.filter((sourceId) =>
        studySources.some((source) => source.id === sourceId)
      );
      const response = await postResearchMessage(
        thread.id,
        message,
        selectedStudySourceIds
      );
      if (selectedCollectionIdRef.current !== collectionId) {
        return;
      }
      setChatEntries((currentEntries) => [
        ...currentEntries,
        {
          id: response.artifact.id,
          role: "assistant",
          content: response.artifact.title,
          status: `Pending assistant artifact: ${response.job.status}`,
          evidenceLines: [],
          artifact: response.artifact,
        }
      ]);
      setPendingJobIds((currentIds) => [...new Set([...currentIds, response.job.id])]);
      setPendingArtifactIds((currentIds) => [
        ...new Set([...currentIds, response.artifact.id]),
      ]);
      setStatusMessage("Queued research-agent run for this chat.");
      await refreshCollectionDetails(selectedLibrary);
      void pollActiveWork();
    } catch (error) {
      if (selectedCollectionIdRef.current === collectionId) {
        setStatusMessage(error instanceof Error ? error.message : "Unable to send message.");
      }
    }
  }

  async function handleUploadSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const fileInput = form.elements.namedItem("files");
    const files = fileInput instanceof HTMLInputElement ? fileInput.files : null;
    if (!files || files.length === 0) {
      setStatusMessage("Choose one or more PDF files before adding papers.");
      return;
    }
    const formData = new FormData(form);
    const title = String(formData.get("collection_title") ?? "");
    const description = String(formData.get("collection_description") ?? "");

    try {
      const job = await uploadLocalLibraryFiles(files, title, description);
      setStatusMessage(`Queued upload import job ${job.id}.`);
      setPendingJobIds((currentIds) => [...new Set([...currentIds, job.id])]);
      form.reset();
      await refreshCollections();
      void pollActiveWork();
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : "Unable to upload papers.");
    }
  }

  async function handlePrepareLibrary() {
    if (!selectedLibrary) {
      return;
    }
    const unparsedPaperIds = papers
      .filter((membership) => !membership.is_parsed)
      .map((membership) => membership.paper_id);
    const unextractedPaperIds = papers
      .filter((membership) => !membership.is_extracted)
      .map((membership) => membership.paper_id);

    try {
      const queuedJobs: BackgroundJobSummary[] = [];
      if (unparsedPaperIds.length > 0) {
        queuedJobs.push(await queueCollectionParse(selectedLibrary.id, unparsedPaperIds));
      }
      if (unextractedPaperIds.length > 0) {
        queuedJobs.push(await queueCollectionExtraction(selectedLibrary, unextractedPaperIds));
      }
      setStatusMessage(
        queuedJobs.length > 0
          ? `Queued ${queuedJobs.length} preparation job(s).`
          : "This library has no missing parse or extraction work."
      );
      setPendingJobIds((currentIds) => [
        ...new Set([...currentIds, ...queuedJobs.map((job) => job.id)]),
      ]);
      await refreshCollectionDetails(selectedLibrary);
      await refreshCollections();
      void pollActiveWork();
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : "Unable to prepare library.");
    }
  }

  async function handleQueueRecoveryAction(action: CollectionExtractionRecoveryAction) {
    if (!selectedLibrary || !action.can_queue_job) {
      return;
    }
    if (action.paper_ids.length === 0) {
      setStatusMessage("This recovery action has no queueable papers in the current batch.");
      return;
    }

    try {
      let job: BackgroundJobSummary;
      if (action.action_type === "parse") {
        job = await queueCollectionParse(selectedLibrary.id, action.paper_ids);
      } else if (action.action_type === "extract") {
        job = await queueCollectionExtraction(selectedLibrary, action.paper_ids);
      } else {
        setStatusMessage("Unsupported recovery action type.");
        return;
      }
      setStatusMessage(`Queued recovery action ${job.id}.`);
      setPendingJobIds((currentIds) => [...new Set([...currentIds, job.id])]);
      await refreshCollectionDetails(selectedLibrary);
      await refreshCollections();
      void pollActiveWork();
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : "Unable to queue recovery action.");
    }
  }

  async function handleRetryArtifact(artifactId: string) {
    if (!selectedLibrary) {
      return;
    }
    const collectionId = selectedLibrary.id;
    setStatusMessage("Retrying model-backed artifact...");

    try {
      const response = await retryResearchArtifact(artifactId);
      if (selectedCollectionIdRef.current !== collectionId) {
        return;
      }
      setPendingJobIds((currentIds) => [...new Set([...currentIds, response.job.id])]);
      setPendingArtifactIds((currentIds) => [
        ...new Set([...currentIds, response.artifact.id]),
      ]);
      setChatEntries((currentEntries) =>
        currentEntries.map((entry) =>
          entry.role === "assistant" && entry.id === response.artifact.id
            ? {
                ...entry,
                content: renderArtifactAnswer(response.artifact),
                status: response.artifact.status,
                evidenceLines: renderEvidenceLines(response.artifact),
                artifact: response.artifact,
              }
            : entry
        )
      );
      await refreshRuntimeStatus();
      await refreshCollectionDetails(selectedLibrary);
      if (activeThread && threadBelongsToCollection(activeThread, collectionId)) {
        await loadResearchThread(activeThread.id, "refresh");
      }
      setStatusMessage("Queued model-backed artifact retry.");
      void pollActiveWork();
    } catch (error) {
      if (selectedCollectionIdRef.current === collectionId) {
        setStatusMessage(error instanceof Error ? error.message : "Unable to retry artifact.");
      }
    }
  }

  async function handleSourceSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedLibrary) {
      return;
    }
    const form = event.currentTarget;
    const formData = new FormData(form);
    const sourceType = String(formData.get("source_type")) as StudySource["source_type"];
    const title = String(formData.get("title") ?? "").trim();
    const value = String(formData.get("value") ?? "").trim();
    if (!title || !value) {
      setStatusMessage("Add a title and note/path before registering project context.");
      return;
    }
    const collectionId = selectedLibrary.id;

    try {
      const study = await ensureStudy(selectedLibrary);
      await createStudySource(study.id, {
        source_type: sourceType,
        title,
        ...(sourceType === "text" ? { content: value } : { path: value })
      });
      const loadedSources = await listStudySources(study.id);
      if (selectedCollectionIdRef.current === collectionId) {
        setStudySources(loadedSources);
        setActiveStudySourceIds((currentIds) =>
          currentIds.filter((sourceId) =>
            loadedSources.some((source) => source.id === sourceId)
          )
        );
        form.reset();
        setStatusMessage("Registered project context for My Artifacts.");
      }
    } catch (error) {
      if (selectedCollectionIdRef.current === collectionId) {
        setStatusMessage(error instanceof Error ? error.message : "Unable to register source.");
      }
    }
  }

  async function handleBrowseArtifactFolder(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedLibrary) {
      return;
    }
    const rootPath = artifactFolderPath.trim();
    if (!rootPath) {
      setStatusMessage("Enter a local folder path before opening My Artifacts.");
      return;
    }
    const collectionId = selectedLibrary.id;
    try {
      const study = await ensureStudy(selectedLibrary);
      const listing = await browseArtifactFolder(study.id, rootPath);
      if (selectedCollectionIdRef.current === collectionId) {
        setArtifactFolderListing(listing);
        setSelectedArtifactFilePaths([]);
        setStatusMessage(`Opened My Artifacts folder with ${listing.entries.length} visible item(s).`);
      }
    } catch (error) {
      if (selectedCollectionIdRef.current === collectionId) {
        setStatusMessage(error instanceof Error ? error.message : "Unable to open folder.");
      }
    }
  }

  async function handleOpenArtifactSubfolder(relativePath: string) {
    if (!selectedLibrary || !artifactFolderPath.trim()) {
      return;
    }
    const collectionId = selectedLibrary.id;
    try {
      const study = await ensureStudy(selectedLibrary);
      const listing = await browseArtifactFolder(study.id, artifactFolderPath.trim(), relativePath);
      if (selectedCollectionIdRef.current === collectionId) {
        setArtifactFolderListing(listing);
        setSelectedArtifactFilePaths([]);
        setStatusMessage(`Opened ${relativePath || "My Artifacts"} folder.`);
      }
    } catch (error) {
      if (selectedCollectionIdRef.current === collectionId) {
        setStatusMessage(error instanceof Error ? error.message : "Unable to open folder.");
      }
    }
  }

  function toggleArtifactFolderFile(path: string) {
    setSelectedArtifactFilePaths((currentPaths) =>
      currentPaths.includes(path)
        ? currentPaths.filter((currentPath) => currentPath !== path)
        : [...currentPaths, path]
    );
  }

  async function handleRegisterSelectedArtifactFiles() {
    if (!selectedLibrary || !artifactFolderListing) {
      return;
    }
    const selectedEntries = artifactFolderListing.entries.filter(
      (entry) =>
        entry.entry_type === "file" &&
        entry.source_type &&
        selectedArtifactFilePaths.includes(entry.path)
    );
    if (selectedEntries.length === 0) {
      setStatusMessage("Select one or more files from My Artifacts first.");
      return;
    }
    const collectionId = selectedLibrary.id;
    try {
      const study = await ensureStudy(selectedLibrary);
      for (const entry of selectedEntries) {
        await createStudySource(study.id, {
          source_type: entry.source_type as StudySource["source_type"],
          title: entry.name,
          path: entry.path
        });
      }
      const loadedSources = await listStudySources(study.id);
      if (selectedCollectionIdRef.current === collectionId) {
        setStudySources(loadedSources);
        setActiveStudySourceIds((currentIds) => [
          ...new Set([
            ...currentIds.filter((sourceId) =>
              loadedSources.some((source) => source.id === sourceId)
            ),
            ...loadedSources
              .filter((source) =>
                selectedEntries.some((entry) => entry.path === source.path)
              )
              .map((source) => source.id)
          ])
        ]);
        setSelectedArtifactFilePaths([]);
        setStatusMessage(`Registered ${selectedEntries.length} file(s) from My Artifacts.`);
      }
    } catch (error) {
      if (selectedCollectionIdRef.current === collectionId) {
        setStatusMessage(error instanceof Error ? error.message : "Unable to register files.");
      }
    }
  }

  function handleArtifactFileSelection(files: FileList | null) {
    if (!files || files.length === 0) {
      return;
    }
    const nextFiles: PendingArtifactFile[] = [];
    let totalBytes = 0;
    let skippedCount = 0;
    for (const file of Array.from(files)) {
      const displayPath = displayPathForBrowserFile(file);
      const sourceType = sourceTypeForFileName(displayPath);
      if (!sourceType || file.size > artifactFileMaxBytes) {
        skippedCount += 1;
        continue;
      }
      if (nextFiles.length >= artifactFileMaxCount || totalBytes + file.size > artifactFileMaxTotalBytes) {
        skippedCount += 1;
        continue;
      }
      totalBytes += file.size;
      nextFiles.push({
        id: `${displayPath}-${file.size}-${file.lastModified}`,
        file,
        title: file.name,
        displayPath,
        sourceType
      });
    }
    setPendingArtifactFiles(nextFiles);
    setStatusMessage(
      nextFiles.length > 0
        ? `Selected ${nextFiles.length} file(s) for My Artifacts.${skippedCount ? ` Skipped ${skippedCount}.` : ""}`
        : "No supported text-like artifact files were selected."
    );
  }

  async function handleRegisterPendingArtifactFiles() {
    if (!selectedLibrary || pendingArtifactFiles.length === 0) {
      return;
    }
    const collectionId = selectedLibrary.id;
    try {
      const study = await ensureStudy(selectedLibrary);
      for (const pendingFile of pendingArtifactFiles) {
        const content = (await pendingFile.file.text()).slice(0, artifactSourceContentLimit);
        await createStudySource(study.id, {
          source_type: pendingFile.sourceType,
          title: pendingFile.title,
          path: pendingFile.displayPath,
          content
        });
      }
      const loadedSources = await listStudySources(study.id);
      if (selectedCollectionIdRef.current === collectionId) {
        setStudySources(loadedSources);
        setActiveStudySourceIds((currentIds) => [
          ...new Set([
            ...currentIds.filter((sourceId) =>
              loadedSources.some((source) => source.id === sourceId)
            ),
            ...loadedSources
              .filter((source) =>
                pendingArtifactFiles.some((pendingFile) => pendingFile.displayPath === source.path)
              )
              .map((source) => source.id)
          ])
        ]);
        setPendingArtifactFiles([]);
        setStatusMessage(`Registered ${pendingArtifactFiles.length} selected file(s).`);
      }
    } catch (error) {
      if (selectedCollectionIdRef.current === collectionId) {
        setStatusMessage(error instanceof Error ? error.message : "Unable to register selected files.");
      }
    }
  }

  const selectedLibraryStudy = selectedLibrary
    ? activeStudy && studyBelongsToCollection(activeStudy, selectedLibrary.id)
      ? activeStudy
      : studies.find((study) => studyBelongsToCollection(study, selectedLibrary.id)) ?? null
    : null;
  const isStudyBriefReady =
    !selectedLibraryStudy || studyBrief?.workspace_id === selectedLibraryStudy.id;
  const studyBriefProposalCandidates = useMemo(
    () => collectStudyBriefProposalCandidates([...threadArtifacts, ...savedArtifacts]),
    [threadArtifacts, savedArtifacts]
  );

  function proposalBelongsToCurrentCandidates() {
    return Boolean(
      studyBriefProposal &&
        studyBriefProposalCandidates.some(
          (candidate) => candidate.id === studyBriefProposal.artifact_id
        )
    );
  }

  useEffect(() => {
    if (studyBriefProposal && !proposalBelongsToCurrentCandidates()) {
      resetStudyBriefProposal();
    }
  }, [studyBriefProposal, studyBriefProposalCandidates]);

  return (
    <div className="react-app-shell" data-app-shell="chat-first">
      <header className="app-topbar">
        <div className="brand-lockup">
          <span className="brand-mark">A</span>
          <div>
            <p className="eyebrow">Arxie</p>
            <h1>Study</h1>
          </div>
        </div>
        <nav className="primary-nav" aria-label="Primary workspace">
          <button
            type="button"
            data-nav-tab="study"
            className={activeView === "study" ? "active" : ""}
            onClick={() => setActiveView("study")}
          >
            Study
          </button>
          <button
            type="button"
            data-nav-tab="library"
            className={activeView === "library" ? "active" : ""}
            onClick={() => setActiveView("library")}
          >
            Library
          </button>
        </nav>
        <button
          type="button"
          className="settings-trigger"
          aria-label="Open top-right settings"
          aria-haspopup="dialog"
          aria-controls="settings-modal"
          onClick={() => setIsSettingsOpen(true)}
        >
          Settings
        </button>
      </header>

      {activeView === "study" ? (
        <StudyView
          selectedLibrary={selectedLibrary}
          collections={collections}
          threads={threads}
          chatEntries={chatEntries}
          chatDraft={chatDraft}
          studyBrief={studyBrief}
          studyBriefDraftText={studyBriefDraftText}
          isStudyBriefReady={isStudyBriefReady}
          studySources={studySources}
          activeStudySourceIds={activeStudySourceIds}
          artifactFolderPath={artifactFolderPath}
          artifactFolderListing={artifactFolderListing}
          selectedArtifactFilePaths={selectedArtifactFilePaths}
          pendingArtifactFiles={pendingArtifactFiles}
          savedArtifacts={savedArtifacts}
          runByArtifactId={runByArtifactId}
          studyBriefProposalCandidates={studyBriefProposalCandidates}
          studyBriefProposal={studyBriefProposal}
          studyBriefProposalArtifactId={studyBriefProposalArtifactId}
          studyBriefProposalError={studyBriefProposalError}
          isStudyBriefProposalLoading={isStudyBriefProposalLoading}
          isStudyBriefProposalAccepting={isStudyBriefProposalAccepting}
          recoveryActions={recoveryActions}
          modelProviderStatus={modelProviderStatus}
          onSelectCollection={selectCollection}
          onChatDraftChange={setChatDraft}
          onSubmitChat={handleChatSubmit}
          onSuggestionClick={handleSuggestionClick}
          onQueueRecoveryAction={handleQueueRecoveryAction}
          onRetryArtifact={handleRetryArtifact}
          onGoToLibrary={() => setActiveView("library")}
          onStudyBriefFieldChange={handleStudyBriefFieldChange}
          onStudyBriefItemsChange={handleStudyBriefItemsChange}
          onSubmitStudyBrief={handleStudyBriefSubmit}
          onStudyBriefProposalArtifactChange={handleStudyBriefProposalArtifactChange}
          onLoadStudyBriefProposal={handleLoadStudyBriefProposal}
          onAcceptStudyBriefProposal={handleAcceptStudyBriefProposal}
          onSubmitSource={handleSourceSubmit}
          onArtifactFolderPathChange={setArtifactFolderPath}
          onBrowseArtifactFolder={handleBrowseArtifactFolder}
          onOpenArtifactSubfolder={handleOpenArtifactSubfolder}
          onToggleArtifactFolderFile={toggleArtifactFolderFile}
          onRegisterSelectedArtifactFiles={handleRegisterSelectedArtifactFiles}
          onArtifactFileSelection={handleArtifactFileSelection}
          onRegisterPendingArtifactFiles={handleRegisterPendingArtifactFiles}
          onToggleStudySource={toggleStudySource}
          onNewChat={handleNewChat}
          onSelectThread={handleSelectThread}
        />
      ) : (
        <LibraryView
          collections={collections}
          selectedLibrary={selectedLibrary}
          papers={papers}
          recoveryActions={recoveryActions}
          onSelectCollection={selectCollection}
          onNewLibrary={handleNewLibrary}
          onUploadSubmit={handleUploadSubmit}
          onPrepareLibrary={handlePrepareLibrary}
          onQueueRecoveryAction={handleQueueRecoveryAction}
        />
      )}

      <p className="status-line" role="status">
        {statusMessage}
      </p>

      {isSettingsOpen ? (
        <SettingsModal
          apiStatus={statusMessage}
          currentProjectId={currentProjectId()}
          selectedLibraryName={selectedLibrary?.title ?? "No library selected"}
          activeStudyTitle={activeStudy?.title ?? "No active Study"}
          collectionCount={collections.length}
          activeJobCount={pendingJobIds.length}
          activeArtifactCount={pendingArtifactIds.length}
          modelProviderStatus={modelProviderStatus}
          workerModelProviderStatus={workerModelProviderStatus}
          workerHeartbeatStatus={workerHeartbeatStatus}
          projectDataPathStatus={projectDataPathStatus}
          providerSmokeStatus={providerSmokeStatus}
          providerSmokeError={providerSmokeError}
          isProviderSmokeRunning={isProviderSmokeRunning}
          runtimeStatusError={runtimeStatusError}
          isRuntimeStatusLoading={isRuntimeStatusLoading}
          onRunProviderSmokeTest={handleRunProviderSmokeTest}
          onClose={() => setIsSettingsOpen(false)}
        />
      ) : null}
    </div>
  );
}
