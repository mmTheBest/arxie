export const apiEndpoints = {
  collections: "/api/v1/collections",
  jobs: "/api/v1/jobs",
  localLibraryUpload: "/api/v1/ingest/local-library-upload",
  researchThreads: "/api/v1/research/threads",
  studies: "/api/v1/studies",
  researchArtifacts: "/api/v1/research/artifacts",
  runtimeStatus: "/api/v1/runtime/status",
  runtimeModelProviderSmoke: "/api/v1/runtime/model-provider/smoke"
} as const;

const PROJECT_STORAGE_KEY = "arxie.currentProjectId";

interface ApiEnvelope<T> {
  data: T;
}

interface HostedAppBootstrap {
  csrf_token?: string | null;
}

export type JsonRecord = Record<string, unknown>;

export interface CollectionSummary {
  id: string;
  title: string;
  description?: string | null;
  extraction_profile_id?: string | null;
  paper_count?: number;
  parsed_paper_count?: number;
  extracted_paper_count?: number;
  latest_job_status?: string | null;
}

export interface PaperSummary {
  id: string;
  title: string;
  publication_year?: number | null;
  venue?: string | null;
}

export interface CollectionPaperMembership {
  id: string;
  paper_id: string;
  paper: PaperSummary;
  is_parsed: boolean;
  is_extracted: boolean;
  latest_parse_job_status?: string | null;
  latest_extraction_job_status?: string | null;
  latest_job_error?: string | null;
}

export interface CollectionExtractionRecoveryAction {
  action_id: string;
  action_type: "parse" | "extract" | "review_evidence";
  can_queue_job: boolean;
  priority: number;
  label: string;
  description: string;
  paper_count: number;
  paper_ids: string[];
  truncated: boolean;
  stale_reasons: string[];
  missing_structured_evidence: string[];
  unresolved_evidence_span_count: number;
  unresolved_evidence_span_samples: Array<{
    paper_id: string;
    paper_title: string;
    mode: string | null;
    reason: string | null;
    target_type: string | null;
    target_id: string | null;
    page_number: number | null;
    quote_preview: string | null;
  }>;
}

export interface CollectionStructuredSummary {
  collection_id: string;
  paper_count: number;
  parsed_paper_count: number;
  extracted_paper_count: number;
  extraction_recovery_actions: CollectionExtractionRecoveryAction[];
}

export interface WorkspaceSummary {
  id: string;
  title: string;
  collection_id?: string | null;
}

export interface StudyBriefItem {
  title: string;
  text: string;
}

export interface StudyBriefContent {
  aim: string;
  hypothesis: string;
  constraints: StudyBriefItem[];
  confirmed_decisions: StudyBriefItem[];
  open_risks: StudyBriefItem[];
  linked_source_ids: string[];
}

export interface StudyBrief {
  id?: string | null;
  workspace_id: string;
  brief: StudyBriefContent;
  version: number;
  updated_by?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface StudyBriefProposalChange {
  field:
    | "aim"
    | "hypothesis"
    | "constraints"
    | "confirmed_decisions"
    | "open_risks"
    | "linked_source_ids";
  before: unknown;
  after: unknown;
}

export interface StudyBriefProposal {
  workspace_id: string;
  artifact_id: string;
  artifact_type: string;
  artifact_title: string;
  current_version: number;
  proposed_brief: StudyBriefContent;
  changes: StudyBriefProposalChange[];
}

export interface StudySource {
  id: string;
  source_type: "text" | "code_path" | "draft_path" | "results_path";
  title: string;
  path?: string | null;
  content?: string | null;
  read_status?: string | null;
  source_size_bytes?: number | null;
  source_mtime_ns?: number | null;
  is_stale?: boolean;
}

export interface ArtifactFileEntry {
  name: string;
  relative_path: string;
  path: string;
  entry_type: "directory" | "file";
  source_type?: StudySource["source_type"] | null;
  size_bytes?: number | null;
  source_mtime_ns?: number | null;
  selectable: boolean;
}

export interface ArtifactFolderListing {
  root_path: string;
  relative_path: string;
  entries: ArtifactFileEntry[];
  truncated: boolean;
  ignored_count: number;
}

export interface ResearchThreadSummary {
  id: string;
  title: string;
  collection_id: string;
  workspace_id?: string | null;
  status: string;
}

export interface ResearchArtifactSummary {
  id: string;
  collection_id?: string;
  thread_id?: string | null;
  title: string;
  status: string;
  artifact_type: string;
  output_payload?: JsonRecord;
  evidence_payload?: JsonRecord;
  error_message?: string | null;
  saved_title?: string | null;
}

export interface BackgroundJobSummary {
  id: string;
  job_type: string;
  status: string;
  payload?: JsonRecord;
  result?: JsonRecord | null;
  error_message?: string | null;
}

export interface ResearchMessageJob {
  artifact: ResearchArtifactSummary;
  job: BackgroundJobSummary;
  run_id?: string | null;
}

export interface ResearchMessageSummary {
  id: string;
  thread_id: string;
  role: "user" | "assistant" | string;
  content: string;
  artifact_id?: string | null;
  metadata?: JsonRecord;
}

export interface ResearchThreadDetail extends ResearchThreadSummary {
  messages: ResearchMessageSummary[];
  artifacts: ResearchArtifactSummary[];
}

export interface ResearchAgentStepSummary {
  id: string;
  run_id: string;
  attempt_number: number;
  ordinal: number;
  step_type: string;
  label: string;
  status: string;
  input_json?: JsonRecord;
  output_json?: JsonRecord;
  error_message?: string | null;
}

export interface ResearchAgentRunSummary {
  id: string;
  thread_id?: string | null;
  artifact_id: string;
  collection_id?: string;
  workspace_id?: string | null;
  skill_id?: string;
  artifact_type?: string;
  model_policy?: string;
  status: string;
  model_name?: string | null;
  error_message?: string | null;
  steps?: ResearchAgentStepSummary[];
  context_pack?: {
    attempt_number?: number;
    context_summary?: JsonRecord;
    retrieval_summary?: JsonRecord;
    context_materialization_summary?: JsonRecord;
    selection_diagnostics?: JsonRecord;
    selected_item_counts?: JsonRecord;
    readiness_warnings?: string[];
  } | null;
  validation_report?: {
    attempt_number?: number;
    harness_status: string;
    missing_evidence?: string[];
    unsupported_claims?: string[];
    readiness_blockers?: string[];
    validation_issues?: JsonRecord[];
    validation_issue_counts?: JsonRecord;
    support_label_counts?: JsonRecord;
    recommendation_health?: JsonRecord;
    reference_integrity?: JsonRecord;
    support_screening?: JsonRecord;
    task_evidence_roles?: JsonRecord;
    task_quality?: JsonRecord;
  } | null;
}

export interface ModelProviderStatus {
  provider: string;
  model_name?: string | null;
  configured: boolean;
  usable: boolean;
  missing_setup: string[];
  setup_hints: string[];
  warnings: string[];
  command?: string | null;
  command_available?: boolean | null;
  allow_agentic_cli: boolean;
  login_status: string;
  login_status_checked: boolean;
  login_status_command?: string | null;
}

export interface WorkerModelProviderStatus {
  provider?: string | null;
  matches_api_provider?: boolean | null;
  source: string;
  setup_hints: string[];
  warnings: string[];
}

export interface WorkerHeartbeatStatus {
  heartbeat_status: "online" | "stale" | "unavailable" | "unknown_project";
  expected_runtime_scope: "default" | "project" | "unknown_project";
  active_worker_count: number;
  stale_worker_count: number;
  latest_seen_seconds_ago?: number | null;
  heartbeat_stale_after_seconds: number;
  setup_hints: string[];
  warnings: string[];
}

export interface ProjectDataPathStatus {
  runtime_data_scope: "default" | "project" | "unknown_project";
  project_id_present: boolean;
  project_found?: boolean | null;
  registry_available: boolean;
  registry_file_exists: boolean;
  registered_project_count: number;
  database_backend: string;
  hosted_mode: boolean;
  host_path_import_policy: "local_unrestricted" | "hosted_allowlisted" | "hosted_disabled";
  allowed_root_count: number;
  setup_hints: string[];
  warnings: string[];
}

export interface RuntimeStatus {
  service: string;
  version: string;
  model_provider: ModelProviderStatus;
  worker_model_provider: WorkerModelProviderStatus;
  worker_heartbeat: WorkerHeartbeatStatus;
  project_data_paths: ProjectDataPathStatus;
}

export interface RuntimeModelProviderSmokeResult {
  status: "success" | "not_configured" | "failed";
  grade?:
    | "passed"
    | "missing_setup"
    | "provider_disabled"
    | "client_factory_missing"
    | "client_factory_failed"
    | "provider_call_failed";
  grade_label?: string | null;
  provider: string;
  model_name?: string | null;
  message: string;
  next_actions?: string[];
  missing_setup: string[];
  setup_hints: string[];
  warnings: string[];
}

function urlWithoutQuery(path: string): string {
  return path.split("?", 1)[0];
}

function hostedAppBootstrap(): HostedAppBootstrap | null {
  const bootstrapElement = document.getElementById("arxie-bootstrap");
  if (!bootstrapElement?.textContent) {
    return null;
  }

  try {
    return JSON.parse(bootstrapElement.textContent) as HostedAppBootstrap;
  } catch {
    return null;
  }
}

export function currentProjectId(): string | null {
  try {
    return sessionStorage.getItem(PROJECT_STORAGE_KEY);
  } catch {
    return null;
  }
}

export function storeCurrentProjectId(projectId: string | null): void {
  try {
    if (projectId) {
      sessionStorage.setItem(PROJECT_STORAGE_KEY, projectId);
    } else {
      sessionStorage.removeItem(PROJECT_STORAGE_KEY);
    }
  } catch {
    // Session storage may be unavailable in hardened browser contexts.
  }
}

export function withProjectHeaders(path: string, options: RequestInit = {}): RequestInit {
  const headers = new Headers(options.headers ?? {});
  const projectId = currentProjectId();
  const bootstrap = hostedAppBootstrap();
  if (projectId && !urlWithoutQuery(path).startsWith("/api/v1/projects")) {
    headers.set("X-Arxie-Project-Id", projectId);
  }
  const method = String(options.method ?? "GET").toUpperCase();
  if (bootstrap?.csrf_token && !["GET", "HEAD", "OPTIONS"].includes(method)) {
    headers.set("X-Arxie-CSRF-Token", bootstrap.csrf_token);
  }
  return { ...options, headers };
}

export async function fetchJson<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, withProjectHeaders(path, options));
  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    const message = typeof payload.message === "string" ? payload.message : `HTTP ${response.status}`;
    throw new Error(`Request failed for ${path}: ${message}`);
  }

  return payload as T;
}

async function fetchData<T>(path: string, options: RequestInit = {}): Promise<T> {
  const envelope = await fetchJson<ApiEnvelope<T>>(path, options);
  return envelope.data;
}

function jsonOptions(method: "POST" | "PATCH" | "PUT", body: object): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  };
}

export function listCollections(): Promise<CollectionSummary[]> {
  return fetchData<CollectionSummary[]>(apiEndpoints.collections);
}

export function fetchRuntimeStatus(): Promise<RuntimeStatus> {
  return fetchData<RuntimeStatus>(apiEndpoints.runtimeStatus);
}

export function normalizeRuntimeModelProviderSmokeResult(
  result: RuntimeModelProviderSmokeResult
): RuntimeModelProviderSmokeResult {
  return {
    ...result,
    grade_label: result.grade_label ?? null,
    next_actions: result.next_actions ?? []
  };
}

export function runModelProviderSmokeTest(): Promise<RuntimeModelProviderSmokeResult> {
  return fetchData<RuntimeModelProviderSmokeResult>(
    apiEndpoints.runtimeModelProviderSmoke,
    {
      method: "POST"
    }
  ).then(normalizeRuntimeModelProviderSmokeResult);
}

export function listCollectionPapers(collectionId: string): Promise<CollectionPaperMembership[]> {
  return fetchData<CollectionPaperMembership[]>(
    `${apiEndpoints.collections}/${collectionId}/papers`
  );
}

export function fetchCollectionStructuredSummary(
  collectionId: string
): Promise<CollectionStructuredSummary> {
  return fetchData<CollectionStructuredSummary>(
    `${apiEndpoints.collections}/${collectionId}/structured-summary`
  );
}

export function listStudies(): Promise<WorkspaceSummary[]> {
  return fetchData<WorkspaceSummary[]>(apiEndpoints.studies);
}

export function createStudy(collection: CollectionSummary): Promise<WorkspaceSummary> {
  return fetchData<WorkspaceSummary>(
    apiEndpoints.studies,
    jsonOptions("POST", {
      title: `${collection.title} Study`,
      collection_id: collection.id,
      owner_id: "local-user"
    })
  );
}

export function fetchStudyBrief(studyId: string): Promise<StudyBrief> {
  return fetchData<StudyBrief>(`${apiEndpoints.studies}/${studyId}/brief`);
}

export function saveStudyBrief(
  studyId: string,
  brief: StudyBriefContent,
  expectedVersion: number
): Promise<StudyBrief> {
  return fetchData<StudyBrief>(
    `${apiEndpoints.studies}/${studyId}/brief`,
    jsonOptions("PUT", {
      expected_version: expectedVersion,
      brief
    })
  );
}

export function fetchStudyBriefProposal(
  studyId: string,
  artifactId: string
): Promise<StudyBriefProposal> {
  const params = new URLSearchParams({ artifact_id: artifactId });
  return fetchData<StudyBriefProposal>(
    `${apiEndpoints.studies}/${studyId}/brief/proposal?${params.toString()}`
  );
}

export function acceptStudyBriefProposal(
  studyId: string,
  artifactId: string,
  brief: StudyBriefContent,
  expectedVersion: number
): Promise<StudyBrief> {
  return fetchData<StudyBrief>(
    `${apiEndpoints.studies}/${studyId}/brief/proposal/accept`,
    jsonOptions("POST", {
      artifact_id: artifactId,
      expected_version: expectedVersion,
      brief
    })
  );
}

export function listStudySources(studyId: string): Promise<StudySource[]> {
  return fetchData<StudySource[]>(`${apiEndpoints.studies}/${studyId}/sources`);
}

export function createStudySource(
  studyId: string,
  payload: {
    source_type: StudySource["source_type"];
    title: string;
    path?: string;
    content?: string;
  }
): Promise<StudySource> {
  return fetchData<StudySource>(
    `${apiEndpoints.studies}/${studyId}/sources`,
    jsonOptions("POST", payload)
  );
}

export function browseArtifactFolder(
  studyId: string,
  rootPath: string,
  relativePath = ""
): Promise<ArtifactFolderListing> {
  const params = new URLSearchParams({ root_path: rootPath });
  if (relativePath) {
    params.set("relative_path", relativePath);
  }
  return fetchData<ArtifactFolderListing>(
    `${apiEndpoints.studies}/${studyId}/artifact-files?${params.toString()}`
  );
}

export function listResearchThreads(collectionId: string): Promise<ResearchThreadSummary[]> {
  return fetchData<ResearchThreadSummary[]>(
    `${apiEndpoints.researchThreads}?collection_id=${encodeURIComponent(collectionId)}`
  );
}

export function fetchResearchThreadDetail(threadId: string): Promise<ResearchThreadDetail> {
  return fetchData<ResearchThreadDetail>(`${apiEndpoints.researchThreads}/${threadId}`);
}

export function createResearchThread(
  collection: CollectionSummary,
  title: string,
  workspaceId?: string | null
): Promise<ResearchThreadSummary> {
  return fetchData<ResearchThreadSummary>(
    apiEndpoints.researchThreads,
    jsonOptions("POST", {
      title,
      collection_id: collection.id,
      owner_id: "local-user",
      workspace_id: workspaceId || undefined,
      selected_paper_ids: []
    })
  );
}

export async function postResearchMessage(
  threadId: string,
  message: string,
  sourceIds: string[]
): Promise<ResearchMessageJob> {
  const data = await fetchData<{
    artifact: ResearchArtifactSummary;
    job: BackgroundJobSummary;
    run_id?: string | null;
  }>(
    `${apiEndpoints.researchThreads}/${threadId}/messages`,
    jsonOptions("POST", {
      message,
      source_ids: sourceIds
    })
  );
  return data;
}

export function listSavedResearchArtifacts(collectionId?: string): Promise<ResearchArtifactSummary[]> {
  const params = new URLSearchParams({ saved_only: "true" });
  if (collectionId) {
    params.set("collection_id", collectionId);
  }
  return fetchData<ResearchArtifactSummary[]>(
    `${apiEndpoints.researchArtifacts}?${params.toString()}`
  );
}

export function fetchResearchArtifact(artifactId: string): Promise<ResearchArtifactSummary> {
  return fetchData<ResearchArtifactSummary>(`${apiEndpoints.researchArtifacts}/${artifactId}`);
}

export function fetchResearchArtifactRun(artifactId: string): Promise<ResearchAgentRunSummary> {
  return fetchData<ResearchAgentRunSummary>(
    `${apiEndpoints.researchArtifacts}/${artifactId}/run`
  );
}

export function retryResearchArtifact(artifactId: string): Promise<ResearchMessageJob> {
  return fetchData<ResearchMessageJob>(
    `${apiEndpoints.researchArtifacts}/${artifactId}/retry`,
    jsonOptions("POST", {})
  );
}

export function listBackgroundJobs(limit = 50): Promise<BackgroundJobSummary[]> {
  return fetchData<BackgroundJobSummary[]>(
    `${apiEndpoints.jobs}?limit=${encodeURIComponent(String(limit))}`
  );
}

export function uploadLocalLibraryFiles(
  files: FileList,
  collectionTitle: string,
  collectionDescription: string
): Promise<BackgroundJobSummary> {
  const formData = new FormData();
  Array.from(files).forEach((file) => {
    const fileWithPath = file as File & { webkitRelativePath?: string };
    formData.append("files", file, fileWithPath.webkitRelativePath || file.name);
  });
  if (collectionTitle.trim()) {
    formData.append("collection_title", collectionTitle.trim());
  }
  if (collectionDescription.trim()) {
    formData.append("collection_description", collectionDescription.trim());
  }
  return fetchData<BackgroundJobSummary>(apiEndpoints.localLibraryUpload, {
    method: "POST",
    body: formData
  });
}

export function queueCollectionParse(
  collectionId: string,
  paperIds?: string[]
): Promise<BackgroundJobSummary> {
  const body = Array.isArray(paperIds) ? { paper_ids: paperIds } : {};
  return fetchData<BackgroundJobSummary>(
    `${apiEndpoints.collections}/${collectionId}/parse`,
    jsonOptions("POST", body)
  );
}

export function queueCollectionExtraction(
  collection: CollectionSummary,
  paperIds?: string[]
): Promise<BackgroundJobSummary> {
  const body: {
    prompt_version: string;
    schema_version: string;
    schema_payload?: Record<string, boolean>;
    extraction_profile_id?: string;
    paper_ids?: string[];
  } = {
    prompt_version: "paperbase-v1",
    schema_version: "paperbase-v1"
  };
  if (collection.extraction_profile_id) {
    body.extraction_profile_id = collection.extraction_profile_id;
  } else {
    body.schema_payload = {
      datasets: true,
      methods: true,
      metrics: true,
      results: true,
      engineering_tricks: true,
      limitations: true,
      research_design_elements: true
    };
  }
  if (Array.isArray(paperIds)) {
    body.paper_ids = paperIds;
  }
  return fetchData<BackgroundJobSummary>(
    `${apiEndpoints.collections}/${collection.id}/extract`,
    jsonOptions("POST", body)
  );
}
