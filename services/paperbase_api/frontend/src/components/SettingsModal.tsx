import {
  apiEndpoints,
  type ModelProviderStatus,
  type ProjectDataPathStatus,
  type RuntimeModelProviderSmokeResult,
  type WorkerHeartbeatStatus,
  type WorkerModelProviderStatus
} from "../api/client";

export interface SettingsModalProps {
  apiStatus: string;
  currentProjectId: string | null;
  selectedLibraryName: string;
  activeStudyTitle: string;
  collectionCount: number;
  activeJobCount: number;
  activeArtifactCount: number;
  modelProviderStatus: ModelProviderStatus | null;
  workerModelProviderStatus: WorkerModelProviderStatus | null;
  workerHeartbeatStatus: WorkerHeartbeatStatus | null;
  projectDataPathStatus: ProjectDataPathStatus | null;
  providerSmokeStatus: RuntimeModelProviderSmokeResult | null;
  providerSmokeError: string | null;
  isProviderSmokeRunning: boolean;
  runtimeStatusError: string | null;
  isRuntimeStatusLoading: boolean;
  onRunProviderSmokeTest: () => void;
  onClose: () => void;
}

export function SettingsModal({
  apiStatus,
  currentProjectId,
  selectedLibraryName,
  activeStudyTitle,
  collectionCount,
  activeJobCount,
  activeArtifactCount,
  modelProviderStatus,
  workerModelProviderStatus,
  workerHeartbeatStatus,
  projectDataPathStatus,
  providerSmokeStatus,
  providerSmokeError,
  isProviderSmokeRunning,
  runtimeStatusError,
  isRuntimeStatusLoading,
  onRunProviderSmokeTest,
  onClose
}: SettingsModalProps) {
  const projectSummary = currentProjectId
    ? `Project id ${currentProjectId}; project-scoped .arxie runtime data.`
    : "Default local Paperbase data; no project id is stored in this browser session.";
  const providerSummary = modelProviderStatus
    ? `${modelProviderStatus.provider}${
        modelProviderStatus.model_name ? ` (${modelProviderStatus.model_name})` : ""
      }: ${modelProviderStatus.usable ? "ready" : "needs setup"}`
    : "Runtime status has not been loaded yet.";
  const setupDetails = modelProviderStatus
    ? [
        ...modelProviderStatus.missing_setup.map((item) => `Missing: ${item}`),
        ...modelProviderStatus.setup_hints,
        ...modelProviderStatus.warnings.map((item) => `Warning: ${item}`)
      ]
    : [];
  const providerLoginSummary = modelProviderStatus
    ? `Login state: ${providerLoginStatusText(modelProviderStatus)}${
        modelProviderStatus.login_status_checked ? " (checked)" : ""
      }.`
    : "Provider login status has not been loaded yet.";
  const providerSmokeSummary = providerSmokeStatus
    ? `Smoke test ${providerSmokeStatusText(providerSmokeStatus)} for ${
        providerSmokeStatus.provider
      }${providerSmokeStatus.model_name ? ` (${providerSmokeStatus.model_name})` : ""}.`
    : "Smoke test has not run in this session.";
  const providerSmokeGrade = providerSmokeStatus?.grade_label?.trim()
    ? `Smoke grade: ${providerSmokeStatus.grade_label}.`
    : null;
  const providerSmokeActionItems = providerSmokeStatus
    ? providerSmokeStatus.next_actions ?? []
    : [];
  const providerSmokeDetails = providerSmokeStatus
    ? [
        ...providerSmokeStatus.missing_setup.map((item) => `Missing: ${item}`),
        ...providerSmokeStatus.setup_hints,
        ...providerSmokeActionItems.map((item) => `Next: ${item}`),
        ...providerSmokeStatus.warnings.map((item) => `Warning: ${item}`)
      ]
    : [];
  const workerProviderSummary = workerModelProviderStatus
    ? `Worker model provider ${
        workerModelProviderStatus.provider ?? "unknown"
      }; Provider agreement ${workerProviderAgreementText(workerModelProviderStatus)} (${
        workerModelProviderStatus.source
      }).`
    : "Worker model provider status has not been loaded yet.";
  const workerProviderDetails = workerModelProviderStatus
    ? [
        ...workerModelProviderStatus.setup_hints,
        ...workerModelProviderStatus.warnings.map((item) => `Warning: ${item}`)
      ]
    : [];
  const workerHeartbeatSummary = workerHeartbeatStatus
    ? `Worker heartbeat: ${workerHeartbeatStatusText(workerHeartbeatStatus)}; ${
        workerHeartbeatStatus.active_worker_count
      } active worker(s), ${workerHeartbeatStatus.stale_worker_count} stale worker(s).`
    : "Worker heartbeat status has not been loaded yet.";
  const workerHeartbeatDetails = workerHeartbeatStatus
    ? [
        ...workerHeartbeatStatus.setup_hints,
        ...workerHeartbeatStatus.warnings.map((item) => `Warning: ${item}`)
      ]
    : [];
  const projectDataPathSummary = projectDataPathStatus
    ? `Project data paths: ${dataScopeText(projectDataPathStatus)}; registry ${
        projectDataPathStatus.registry_available ? "available" : "unavailable"
      }, ${projectDataPathStatus.registered_project_count} registered project(s); database backend ${
        projectDataPathStatus.database_backend
      }.`
    : "Project data path status has not been loaded yet.";
  const hostPathImportSummary = projectDataPathStatus
    ? `Host path import: ${hostPathImportPolicyText(
        projectDataPathStatus
      )}; ${projectDataPathStatus.allowed_root_count} allowed root(s) configured.`
    : "Host path import status has not been loaded yet.";
  const projectDataPathDetails = projectDataPathStatus
    ? [
        ...projectDataPathStatus.setup_hints,
        ...projectDataPathStatus.warnings.map((item) => `Warning: ${item}`)
      ]
    : [];
  const providerSetupChecklist = buildProviderSetupChecklist(
    modelProviderStatus,
    workerModelProviderStatus
  );
  const providerSetupRequiredSteps = buildProviderSetupRequiredSteps(
    modelProviderStatus,
    workerModelProviderStatus
  );

  return (
    <div className="modal-backdrop" role="presentation">
      <section id="settings-modal" className="settings-modal" role="dialog" aria-modal="true">
        <div className="module-heading">
          <div>
            <p className="section-label">Runtime</p>
            <h2>Settings</h2>
          </div>
          <button type="button" onClick={onClose} aria-label="Close settings">
            Close
          </button>
        </div>
        <p className="settings-note">frontend-safe status from current browser state</p>
        <dl>
          <div>
            <dt>Model provider status</dt>
            <dd>
              {isRuntimeStatusLoading ? <span>Checking runtime status...</span> : null}
              <p className="settings-runtime-status">{providerSummary}</p>
              {modelProviderStatus?.command ? (
                <p>
                  Command: {modelProviderStatus.command}{" "}
                  {modelProviderStatus.command_available === false ? "(not found)" : "(found)"}
                </p>
              ) : null}
              {runtimeStatusError ? (
                <p className="settings-warning">Runtime status error: {runtimeStatusError}</p>
              ) : null}
              {setupDetails.length > 0 ? (
                <ul className="settings-inline-list">
                  {setupDetails.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              ) : null}
            </dd>
          </div>
          <div className="settings-provider-list">
            <dt>Provider login</dt>
            <dd>
              <p>{providerLoginSummary}</p>
              {modelProviderStatus?.login_status_command ? (
                <p>Status command: {modelProviderStatus.login_status_command}</p>
              ) : null}
              <p>No provider tokens, account ids, or CLI output are shown here.</p>
            </dd>
          </div>
          <div className="settings-provider-list">
            <dt>Provider smoke test</dt>
            <dd>
              <p>{providerSmokeSummary}</p>
              {providerSmokeGrade ? <p>{providerSmokeGrade}</p> : null}
              <p>No research content is sent and no model output is shown here.</p>
              <button
                type="button"
                onClick={onRunProviderSmokeTest}
                disabled={isProviderSmokeRunning}
              >
                {isProviderSmokeRunning ? "Running provider smoke test..." : "Run provider smoke test"}
              </button>
              {providerSmokeError ? (
                <p className="settings-warning">Provider smoke test error: {providerSmokeError}</p>
              ) : null}
              {providerSmokeDetails.length > 0 ? (
                <ul className="settings-inline-list">
                  {providerSmokeDetails.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              ) : null}
            </dd>
          </div>
          <div className="settings-provider-list">
            <dt>Provider setup</dt>
            <dd>
              <p>Current provider path: {currentProviderSetupPath(modelProviderStatus)}.</p>
              <p>Required now</p>
              <ul className="settings-inline-list">
                {providerSetupRequiredSteps.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
              <p>Available provider paths</p>
              <ul>
                {providerSetupChecklist.map((item) => (
                  <li key={item.id}>
                    <strong>{item.label}</strong>
                    <span>
                      {setupStepStatusText(item.status)}: {item.detail}
                    </span>
                  </li>
                ))}
              </ul>
            </dd>
          </div>
          <div>
            <dt>Search/backend readiness</dt>
            <dd>
              Backend status: {apiStatus}. Search readiness is not live-checked by this
              React modal.
            </dd>
          </div>
          <div>
            <dt>Worker/API readiness</dt>
            <dd>
              <p>{workerProviderSummary}</p>
              {workerProviderDetails.length > 0 ? (
                <ul className="settings-inline-list">
                  {workerProviderDetails.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              ) : null}
              <p>{workerHeartbeatSummary}</p>
              <p>No worker process ids or queue names are shown here.</p>
              {workerHeartbeatDetails.length > 0 ? (
                <ul className="settings-inline-list">
                  {workerHeartbeatDetails.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              ) : null}
              <p>
                API contracts: {apiEndpoints.collections}, {apiEndpoints.researchThreads},{" "}
                {apiEndpoints.studies}, {apiEndpoints.researchArtifacts}. Tracked active
                work: {activeJobCount} job(s), {activeArtifactCount} artifact(s).
              </p>
            </dd>
          </div>
          <div>
            <dt>Local project/data</dt>
            <dd>
              <p>
                {projectSummary} Library: {selectedLibraryName}. Study: {activeStudyTitle}.
                Libraries loaded: {collectionCount}.
              </p>
              <p>{projectDataPathSummary}</p>
              <p>{hostPathImportSummary}</p>
              <p>No raw local paths are shown here.</p>
              {projectDataPathDetails.length > 0 ? (
                <ul className="settings-inline-list">
                  {projectDataPathDetails.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              ) : null}
            </dd>
          </div>
        </dl>
      </section>
    </div>
  );
}

interface ProviderSetupChecklistItem {
  id: string;
  label: string;
  status: "ready" | "needs_setup" | "available" | "disabled";
  detail: string;
}

function buildProviderSetupChecklist(
  status: ModelProviderStatus | null,
  workerStatus: WorkerModelProviderStatus | null
): ProviderSetupChecklistItem[] {
  const selectedProvider = status?.provider ?? null;
  return [
    {
      id: "openai",
      label: "OpenAI API",
      status: providerPathStatus(status, "openai"),
      detail: "Set OPENAI_API_KEY in the repo-local .env, then restart API and worker."
    },
    {
      id: "claude_cli",
      label: "Claude Code CLI",
      status: providerPathStatus(status, "claude_cli"),
      detail: "Run claude setup-token before starting Arxie."
    },
    {
      id: "codex_cli",
      label: "Codex CLI",
      status: providerPathStatus(status, "codex_cli"),
      detail:
        "Set CODEX_HOME for Arxie, run codex login, and confirm with codex login status."
    },
    {
      id: "worker_agreement",
      label: "Worker agreement",
      status: workerAgreementSetupStatus(status, workerStatus),
      detail:
        "Set PAPERBASE_EXPECTED_WORKER_MODEL_PROVIDER when API and worker use separate startup environments."
    }
  ];
}

function providerPathStatus(
  status: ModelProviderStatus | null,
  provider: string
): ProviderSetupChecklistItem["status"] {
  if (!status) {
    return "available";
  }
  if (status.provider === "none") {
    return "available";
  }
  if (status.provider !== provider) {
    return "available";
  }
  return status.usable ? "ready" : "needs_setup";
}

function workerAgreementSetupStatus(
  status: ModelProviderStatus | null,
  workerStatus: WorkerModelProviderStatus | null
): ProviderSetupChecklistItem["status"] {
  if (status?.provider === "none") {
    return "disabled";
  }
  if (workerStatus?.matches_api_provider === false) {
    return "needs_setup";
  }
  if (workerStatus?.matches_api_provider === true) {
    return "ready";
  }
  return "available";
}

function buildProviderSetupRequiredSteps(
  status: ModelProviderStatus | null,
  workerStatus: WorkerModelProviderStatus | null
): string[] {
  const actions: string[] = [];
  const missingSetup = new Set(status?.missing_setup ?? []);

  function add(action: string) {
    if (!actions.includes(action)) {
      actions.push(action);
    }
  }

  if (!status) {
    add("Load runtime status before choosing a model provider path.");
    return actions;
  }

  if (status.provider === "none") {
    add("Choose OpenAI API, Claude Code CLI, or Codex CLI for model-backed Study skills.");
  }
  if (missingSetup.has("OPENAI_API_KEY")) {
    add("Set OPENAI_API_KEY in the repo-local .env, then restart API and worker.");
  }
  if (missingSetup.has("PAPERBASE_CLAUDE_COMMAND")) {
    add("Install or configure the claude command.");
  }
  if (missingSetup.has("CLAUDE_CODE_OAUTH_TOKEN")) {
    add("Run claude setup-token before starting Arxie, then restart API and worker.");
  }
  if (missingSetup.has("PAPERBASE_CODEX_COMMAND")) {
    add("Install or configure the codex command.");
  }
  if (missingSetup.has("PAPERBASE_ALLOW_AGENTIC_CLI")) {
    add("Set PAPERBASE_ALLOW_AGENTIC_CLI=true only for trusted local corpora.");
  }
  if (missingSetup.has("CODEX_HOME")) {
    add("Set CODEX_HOME for Arxie, run codex login, and confirm with codex login status.");
  }
  if (missingSetup.has("codex login")) {
    add("Run codex login and confirm with codex login status.");
  }
  if (status.provider !== "none" && workerStatus?.matches_api_provider === false) {
    add("Restart API and worker with matching model-provider environment variables.");
  }
  if (actions.length === 0 && status.usable) {
    add("Run provider smoke test to verify the current provider path.");
  }
  if (actions.length === 0) {
    add("Review provider status and setup hints, then rerun runtime status.");
  }
  return actions;
}

function currentProviderSetupPath(status: ModelProviderStatus | null): string {
  if (!status) {
    return "runtime status not loaded";
  }
  if (status.provider === "none") {
    return "model-backed features disabled";
  }
  if (status.provider === "openai") {
    return "OpenAI API";
  }
  if (status.provider === "claude_cli") {
    return "Claude Code CLI";
  }
  if (status.provider === "codex_cli") {
    return "Codex CLI";
  }
  return status.provider;
}

function setupStepStatusText(status: ProviderSetupChecklistItem["status"]): string {
  switch (status) {
    case "ready":
      return "Ready";
    case "needs_setup":
      return "Needs setup";
    case "disabled":
      return "Disabled";
    case "available":
    default:
      return "Available";
  }
}

function workerProviderAgreementText(status: WorkerModelProviderStatus): string {
  if (status.matches_api_provider === true) {
    return "matches API provider";
  }
  if (status.matches_api_provider === false) {
    return "does not match API provider";
  }
  return "not reported by runtime";
}

function providerLoginStatusText(status: { login_status?: string | null }): string {
  switch (status.login_status) {
    case "logged_in":
      return "logged in";
    case "logged_out":
      return "not logged in";
    case "token_configured":
      return "token configured";
    case "not_checked":
      return "not checked";
    case "unknown":
      return "unknown";
    case "not_applicable":
    case null:
    case undefined:
      return "not required for this provider";
    default:
      return status.login_status;
  }
}

function providerSmokeStatusText(status: RuntimeModelProviderSmokeResult): string {
  if (status.status === "success") {
    return "passed";
  }
  if (status.status === "not_configured") {
    return "needs setup";
  }
  return "failed";
}

function workerHeartbeatStatusText(status: WorkerHeartbeatStatus): string {
  const scope =
    status.expected_runtime_scope === "project"
      ? "project worker"
      : status.expected_runtime_scope === "unknown_project"
        ? "unknown project"
        : "default worker";
  const freshness =
    status.latest_seen_seconds_ago === null || status.latest_seen_seconds_ago === undefined
      ? "no heartbeat seen"
      : `last seen ${status.latest_seen_seconds_ago}s ago`;
  return `${scope} ${status.heartbeat_status}, ${freshness}`;
}

function dataScopeText(status: ProjectDataPathStatus): string {
  if (status.runtime_data_scope === "project") {
    return status.project_found === true
      ? "active project-scoped .arxie data"
      : "project-scoped data";
  }
  if (status.runtime_data_scope === "unknown_project") {
    return "unknown browser project id";
  }
  return "default local Paperbase data";
}

function hostPathImportPolicyText(status: ProjectDataPathStatus): string {
  if (status.host_path_import_policy === "hosted_allowlisted") {
    return "hosted allowlist";
  }
  if (status.host_path_import_policy === "hosted_disabled") {
    return "disabled in hosted mode";
  }
  return "local unrestricted";
}
