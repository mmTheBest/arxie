import type { FormEvent } from "react";

import { ArtifactIntake, type PendingArtifactFileItem } from "./ArtifactIntake";
import { ArtifactList } from "./ArtifactList";
import { RecoveryActionSampleList } from "./RecoveryActionSamples";
import type {
  ArtifactFolderListing,
  CollectionExtractionRecoveryAction,
  CollectionSummary,
  JsonRecord,
  ModelProviderStatus,
  ResearchAgentRunSummary,
  ResearchArtifactSummary,
  ResearchThreadSummary,
  StudyBrief,
  StudyBriefProposal,
  StudyBriefProposalChange,
  StudySource
} from "../api/client";

export type ChatEntry =
  | { id: string; role: "user"; content: string }
  | {
      id: string;
      role: "assistant";
      content: string;
      status: string;
      evidenceLines: string[];
      artifact?: ResearchArtifactSummary;
    };

export type ChatSuggestion = {
  label: string;
  requiresArtifact?: boolean;
  requiresModelProvider?: boolean;
  buildPrompt: (libraryTitle: string, attachedContext: string) => string;
};

export type StudyBriefListField = "constraints" | "confirmed_decisions" | "open_risks";

export interface StudyBriefDraftText {
  aim: string;
  hypothesis: string;
  constraints: string;
  confirmed_decisions: string;
  open_risks: string;
}

type SchemaValidationDiagnostics = {
  status: string;
  schemaName: string;
  attempts: number | null;
};

type RetrievalSummaryDiagnostics = {
  backendStatus: string;
  retrievalMatchCount: number;
  selectedChunkCount: number;
  sqlChunkCount: number;
  selectedFigureCount: number;
  selectedTableCount: number;
  selectedStructuredEntityCount: number;
  selectedResultEvidenceCount: number;
};

type ContextMaterializationDiagnostics = {
  methodCount: number;
  datasetCount: number;
  metricCount: number;
  baselineCount: number;
  limitationCount: number;
  benchmarkTableCount: number;
  limitationCategoryCount: number;
  resultEvidenceCount: number;
  claimCount: number;
  claimEvidenceSpanCount: number;
  claimEvidenceMapCount: number;
};

type ContextCacheDiagnostics = {
  status: string;
  sourceContextPackId: string | null;
  sourceAttemptNumber: number | null;
};

type SelectionDiagnosticItem = {
  itemId: string;
  itemType: string;
  label: string;
  contextRole: string;
  contextReason: string;
  selectionScore: number | null;
  featureSummary: string[];
};

type SelectionDiagnostics = {
  items: SelectionDiagnosticItem[];
};

type ReadToolDiagnosticItem = {
  name: string;
  state: "available" | "blocked";
  contextGroups: string[];
  missingRequirements: string[];
  selectedContextCounts: string[];
  sideEffects: boolean;
};

type ReadToolReadinessDiagnostics = {
  availableCount: number;
  blockedCount: number;
  items: ReadToolDiagnosticItem[];
};

type ReadToolObservationItem = {
  name: string;
  state: string;
  observedCounts: string[];
  observedFacets: string[];
};

type ReadToolObservationDiagnostics = {
  executedCount: number;
  blockedCount: number;
  items: ReadToolObservationItem[];
};

type ValidationIssueItem = {
  category: string;
  severity: string;
  code: string;
  source: string;
  message: string;
  remediation: string;
};

type ValidationIssueDiagnostics = {
  issueCount: number;
  severitySummary: string[];
  items: ValidationIssueItem[];
};

type RecommendationHealthDiagnostics = {
  recommendationCount: number;
  missingSupportStatus: number;
  invalidSupportStatus: number;
  missingSupportingLayers: number;
  missingEvidenceReferences: number;
  speculativeAsSupported: number;
  unavailableLayers: number;
  invalidLayers: number;
  sourceFactMarkedSupported: number;
  availableSupportingLayers: string[];
};

type TaskEvidenceRoleDiagnostics = {
  availableGroups: string[];
  satisfiedGroups: string[];
  missingGroups: string[];
  referencedRoles: string[];
  availableRoles: string[];
};

type SupportCoverageDiagnostics = {
  supportStatuses: string[];
  supportingLayers: string[];
};

type ReferenceIntegrityDiagnostics = {
  artifactReferences: number;
  validReferences: number;
  invalidReferences: number;
  unverifiableReferences: number;
  incompatibleReferenceTypes: number;
};

type SupportScreeningItem = {
  label: string;
  checkedRecommendations: number;
  weakRecommendations: number;
  unavailableReferences: number;
};

type SupportScreeningDiagnostics = {
  items: SupportScreeningItem[];
};

type TaskQualityDiagnostics = {
  artifactType: string | null;
  checkedArtifactCount: number;
  requiredSections: string[];
  missingRequiredSections: string[];
};

export const CHAT_SUGGESTIONS: ChatSuggestion[] = [
  {
    label: "Literature review of this library",
    buildPrompt: (libraryTitle) =>
      `Write a literature review of ${libraryTitle}. Organize major themes, evidence strength, disagreements, and future directions.`
  },
  {
    label: "Summarize the field consensus",
    buildPrompt: (libraryTitle) =>
      `Synthesize and summarize the field consensus in ${libraryTitle}. Separate well-supported findings, contested claims, and open questions.`
  },
  {
    label: "Find gaps and limitations",
    buildPrompt: (libraryTitle) =>
      `Synthesize research gaps and limitations across ${libraryTitle}. Prioritize gaps with clear evidence from the papers and explain why they matter.`
  },
  {
    label: "Review my draft",
    requiresArtifact: true,
    buildPrompt: (libraryTitle, attachedContext) =>
      `Review my attached draft or note against ${libraryTitle}. Identify unsupported claims, missing citations, unclear logic, and concrete revisions.${attachedContext}`
  },
  {
    label: "Design an experiment",
    requiresModelProvider: true,
    buildPrompt: (libraryTitle) =>
      `Design an experiment grounded in ${libraryTitle}. Include hypothesis, controls, measurements, expected outcomes, and risks.`
  },
  {
    label: "Compare papers",
    buildPrompt: (libraryTitle) =>
      `Compare the key papers in ${libraryTitle}. Group them by methods, datasets, findings, limitations, and citation-backed disagreements.`
  },
  {
    label: "Plan a benchmark",
    buildPrompt: (libraryTitle) =>
      `Plan a benchmark based on ${libraryTitle}. Define tasks, datasets, metrics, baselines, ablations, and reporting standards.`
  },
  {
    label: "Refine a proposal",
    requiresArtifact: true,
    requiresModelProvider: true,
    buildPrompt: (libraryTitle, attachedContext) =>
      `Refine my attached proposal or note using ${libraryTitle}. Strengthen motivation, novelty, aims, methods, and evidence gaps.${attachedContext}`
  },
  {
    label: "Critique my idea",
    requiresArtifact: true,
    buildPrompt: (libraryTitle, attachedContext) =>
      `Critique my attached idea or note against ${libraryTitle}. Identify weak assumptions, missing evidence, risks, and sharper alternatives.${attachedContext}`
  }
];

const SAFE_READ_TOOL_COUNT_KEYS = new Set([
  "papers",
  "chunks",
  "evidence_spans",
  "structured_evidence",
  "structured_entities",
  "result_evidence",
  "figures",
  "tables",
  "evidence_memory",
  "pattern_memory",
  "source_fact_memory",
  "graph_nodes",
  "graph_edges",
  "study_brief",
  "sources"
]);
const SAFE_TRACE_LABEL_PATTERN = /^[A-Za-z0-9_.-]+$/;
const PRIVATE_TRACE_LABEL_PATTERN = /(prompt|secret|token|key|chain_of_thought|raw_response|private)/i;
const VALIDATION_ISSUE_SCAN_LIMIT = 24;

const studyBriefProposalFieldLabels: Record<StudyBriefProposalChange["field"], string> = {
  aim: "Current aim",
  hypothesis: "Working hypothesis",
  constraints: "Constraints",
  confirmed_decisions: "Confirmed decisions",
  open_risks: "Open risks",
  linked_source_ids: "Linked sources"
};

function isJsonRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function selectedDraftOrNoteSources(sources: StudySource[]): StudySource[] {
  return sources.filter(
    (source) => source.source_type === "text" || source.source_type === "draft_path"
  );
}

export function suggestionHasRequiredContext(
  suggestion: ChatSuggestion,
  selectedSources: StudySource[]
): boolean {
  if (!suggestion.requiresArtifact) {
    return true;
  }
  return selectedDraftOrNoteSources(selectedSources).length > 0;
}

export function suggestionRequiresReadyModelProvider(suggestion: ChatSuggestion): boolean {
  return Boolean(suggestion.requiresModelProvider);
}

export function suggestionHasReadyModelProvider(
  suggestion: ChatSuggestion,
  modelProviderStatus: ModelProviderStatus | null
): boolean {
  if (!suggestionRequiresReadyModelProvider(suggestion)) {
    return true;
  }
  return modelProviderStatus?.usable !== false;
}

export function isRetryableModelArtifact(
  artifact: ResearchArtifactSummary | null | undefined
): artifact is ResearchArtifactSummary {
  return artifact?.status === "blocked" && artifact.output_payload?.model_required === true;
}

export function providerSetupGuidance(
  modelProviderStatus: ModelProviderStatus | null
): string {
  if (!modelProviderStatus) {
    return "Open Settings for provider setup, then retry this artifact.";
  }

  const missingSetup = new Set(modelProviderStatus.missing_setup);
  const runSmokeThenRetry =
    "Run provider smoke test in Settings, then retry this artifact.";

  if (modelProviderStatus.usable) {
    return runSmokeThenRetry;
  }
  if (modelProviderStatus.provider === "none") {
    return "Enable a model provider in Settings, run provider smoke test in Settings, then retry this artifact.";
  }

  const setupSteps: string[] = [];
  const addSetupStep = (step: string) => {
    if (!setupSteps.includes(step)) {
      setupSteps.push(step);
    }
  };

  if (
    missingSetup.has("PAPERBASE_CODEX_COMMAND") ||
    (modelProviderStatus.provider === "codex_cli" &&
      modelProviderStatus.command_available === false)
  ) {
    addSetupStep("Install or configure the codex command");
  }
  if (
    missingSetup.has("PAPERBASE_CLAUDE_COMMAND") ||
    (modelProviderStatus.provider === "claude_cli" &&
      modelProviderStatus.command_available === false)
  ) {
    addSetupStep("Install or configure the claude command");
  }
  if (missingSetup.has("OPENAI_API_KEY")) {
    addSetupStep("Set OPENAI_API_KEY in the repo-local .env, then restart API and worker");
  }
  if (missingSetup.has("PAPERBASE_ALLOW_AGENTIC_CLI")) {
    addSetupStep(
      "Set PAPERBASE_ALLOW_AGENTIC_CLI=true only for trusted local corpora, then restart API and worker"
    );
  }
  if (missingSetup.has("CODEX_HOME")) {
    addSetupStep("Set CODEX_HOME for Arxie, run codex login, confirm with codex login status");
  }
  if (
    missingSetup.has("codex login") ||
    (modelProviderStatus.login_status === "logged_out" &&
      modelProviderStatus.login_status_command === "codex login status")
  ) {
    addSetupStep("Run codex login, confirm with codex login status");
  }
  if (missingSetup.has("CLAUDE_CODE_OAUTH_TOKEN")) {
    addSetupStep("Run claude setup-token before starting Arxie, then restart API and worker");
  }

  if (setupSteps.length > 0) {
    return `${setupSteps.join("; ")}. ${runSmokeThenRetry}`;
  }

  return "Open Settings for provider setup, run provider smoke test in Settings, then retry this artifact.";
}

export function BlockedArtifactRecovery({
  artifact,
  modelProviderStatus,
  onRetryArtifact
}: {
  artifact: ResearchArtifactSummary | null | undefined;
  modelProviderStatus: ModelProviderStatus | null;
  onRetryArtifact: (artifactId: string) => void;
}) {
  if (!isRetryableModelArtifact(artifact)) {
    return null;
  }

  return (
    <div
      className="blocked-artifact-recovery"
      data-contract="Blocked model artifact retry"
    >
      <p>{providerSetupGuidance(modelProviderStatus)}</p>
      <button type="button" onClick={() => onRetryArtifact(artifact.id)}>
        Retry model run
      </button>
    </div>
  );
}

function schemaValidationFromRun(
  run: ResearchAgentRunSummary | null | undefined
): SchemaValidationDiagnostics | null {
  const synthesisStep = [...(run?.steps ?? [])]
    .reverse()
    .find((step) => step.step_type === "synthesis");
  const schemaValidation = synthesisStep?.output_json?.schema_validation;
  if (!isJsonRecord(schemaValidation)) {
    return null;
  }

  const status = typeof schemaValidation.status === "string" ? schemaValidation.status : "";
  if (!status) {
    return null;
  }
  const schemaName =
    typeof schemaValidation.schema === "string"
      ? schemaValidation.schema
      : typeof schemaValidation.schema_name === "string"
        ? schemaValidation.schema_name
        : "model output";
  const attemptsValue = schemaValidation.attempts;
  const attempts =
    typeof attemptsValue === "number" && Number.isFinite(attemptsValue)
      ? attemptsValue
      : null;

  return {
    status,
    schemaName,
    attempts
  };
}

function numberFromRecord(record: JsonRecord, key: string): number {
  const value = record[key];
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function countFromRecord(record: JsonRecord, key: string): number {
  const value = record[key];
  return typeof value === "number" && Number.isInteger(value) && value > 0
    ? value
    : 0;
}

function retrievalSummaryFromRun(
  run: ResearchAgentRunSummary | null | undefined
): RetrievalSummaryDiagnostics | null {
  const retrievalSummary = run?.context_pack?.retrieval_summary;
  if (!isJsonRecord(retrievalSummary)) {
    return null;
  }
  if (Object.keys(retrievalSummary).length === 0) {
    return null;
  }
  const backendStatus =
    typeof retrievalSummary.backend_status === "string"
      ? retrievalSummary.backend_status
      : "unknown";
  return {
    backendStatus,
    retrievalMatchCount: numberFromRecord(retrievalSummary, "retrieval_match_count"),
    selectedChunkCount: numberFromRecord(retrievalSummary, "selected_chunk_count"),
    sqlChunkCount: numberFromRecord(retrievalSummary, "sql_chunk_count"),
    selectedFigureCount: numberFromRecord(retrievalSummary, "selected_figure_count"),
    selectedTableCount: numberFromRecord(retrievalSummary, "selected_table_count"),
    selectedStructuredEntityCount: numberFromRecord(
      retrievalSummary,
      "selected_structured_entity_count"
    ),
    selectedResultEvidenceCount: numberFromRecord(
      retrievalSummary,
      "selected_result_evidence_count"
    )
  };
}

function contextMaterializationFromRun(
  run: ResearchAgentRunSummary | null | undefined
): ContextMaterializationDiagnostics | null {
  const materialization = run?.context_pack?.context_materialization_summary;
  if (!isJsonRecord(materialization)) {
    return null;
  }
  const summary = materialization.summary;
  if (!isJsonRecord(summary)) {
    return null;
  }
  const diagnostics = {
    methodCount: numberFromRecord(summary, "method_count"),
    datasetCount: numberFromRecord(summary, "dataset_count"),
    metricCount: numberFromRecord(summary, "metric_count"),
    baselineCount: numberFromRecord(summary, "baseline_count"),
    limitationCount: numberFromRecord(summary, "limitation_count"),
    benchmarkTableCount: numberFromRecord(summary, "benchmark_table_count"),
    limitationCategoryCount: numberFromRecord(summary, "limitation_category_count"),
    resultEvidenceCount: numberFromRecord(summary, "result_evidence_count"),
    claimCount: numberFromRecord(summary, "claim_count"),
    claimEvidenceSpanCount: numberFromRecord(summary, "claim_evidence_span_count"),
    claimEvidenceMapCount: numberFromRecord(summary, "claim_evidence_map_count")
  };
  const total =
    diagnostics.methodCount +
    diagnostics.datasetCount +
    diagnostics.metricCount +
    diagnostics.baselineCount +
    diagnostics.limitationCount +
    diagnostics.benchmarkTableCount +
    diagnostics.limitationCategoryCount +
    diagnostics.resultEvidenceCount +
    diagnostics.claimCount +
    diagnostics.claimEvidenceSpanCount +
    diagnostics.claimEvidenceMapCount;
  return total > 0 ? diagnostics : null;
}

function contextCacheFromRun(
  run: ResearchAgentRunSummary | null | undefined
): ContextCacheDiagnostics | null {
  const contextStep = [...(run?.steps ?? [])]
    .reverse()
    .find((step) => step.step_type === "context");
  const cacheTrace = contextStep?.output_json?.context_cache;
  if (isJsonRecord(cacheTrace)) {
    const status = typeof cacheTrace.status === "string" ? cacheTrace.status : "";
    if (!status) {
      return null;
    }
    const sourceContextPackId =
      typeof cacheTrace.source_context_pack_id === "string"
        ? cacheTrace.source_context_pack_id
        : null;
    const sourceAttemptValue = cacheTrace.source_attempt_number;
    const sourceAttemptNumber =
      typeof sourceAttemptValue === "number" && Number.isFinite(sourceAttemptValue)
        ? sourceAttemptValue
        : null;
    return {
      status,
      sourceContextPackId,
      sourceAttemptNumber
    };
  }

  if (typeof contextStep?.output_json?.cache_hit === "boolean") {
    return {
      status: contextStep.output_json.cache_hit ? "hit" : "miss",
      sourceContextPackId: null,
      sourceAttemptNumber: null
    };
  }

  return null;
}

function stringFromRecord(record: JsonRecord, key: string): string | null {
  const value = record[key];
  return typeof value === "string" && value.trim() ? value : null;
}

function scoreFromRecord(record: JsonRecord, key: string): number | null {
  const value = record[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function safeTraceLabel(value: string): string | null {
  const trimmed = value.trim();
  if (
    !trimmed ||
    trimmed.length > 80 ||
    !SAFE_TRACE_LABEL_PATTERN.test(trimmed) ||
    PRIVATE_TRACE_LABEL_PATTERN.test(trimmed)
  ) {
    return null;
  }
  return trimmed;
}

function safeTraceMessage(value: string): string | null {
  const compact = value.trim().replace(/\s+/g, " ");
  if (!compact || PRIVATE_TRACE_LABEL_PATTERN.test(compact)) {
    return null;
  }
  return compact.length > 240 ? `${compact.slice(0, 237)}...` : compact;
}

function stringListFromRecord(record: JsonRecord, key: string): string[] {
  const value = record[key];
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter((item): item is string => typeof item === "string")
    .map((item) => safeTraceLabel(item))
    .filter((item): item is string => Boolean(item))
    .slice(0, 4);
}

function countMapSummaryFromRecord(record: JsonRecord, key: string): string[] {
  const value = record[key];
  if (!isJsonRecord(value)) {
    return [];
  }
  const summary: string[] = [];
  for (const [rawLabel, rawCount] of Object.entries(value)) {
    const label = safeTraceLabel(rawLabel);
    if (
      !label ||
      typeof rawCount !== "number" ||
      !Number.isInteger(rawCount) ||
      rawCount <= 0
    ) {
      continue;
    }
    summary.push(`${label}: ${rawCount}`);
    if (summary.length >= 4) {
      break;
    }
  }
  return summary;
}

function selectedContextCountSummaryFromRecord(record: JsonRecord): string[] {
  const counts = record.selected_context_counts;
  if (!isJsonRecord(counts)) {
    return [];
  }
  const summary: string[] = [];
  for (const [key, value] of Object.entries(counts)) {
    if (
      !SAFE_READ_TOOL_COUNT_KEYS.has(key) ||
      typeof value !== "number" ||
      !Number.isFinite(value) ||
      value <= 0
    ) {
      continue;
    }
    summary.push(`${key}: ${value}`);
    if (summary.length >= 4) {
      break;
    }
  }
  return summary;
}

function observedCountSummaryFromRecord(record: JsonRecord): string[] {
  const counts = record.observed_counts;
  if (!isJsonRecord(counts)) {
    return [];
  }
  const summary: string[] = [];
  for (const [key, value] of Object.entries(counts)) {
    if (
      !SAFE_READ_TOOL_COUNT_KEYS.has(key) ||
      typeof value !== "number" ||
      !Number.isFinite(value) ||
      value <= 0
    ) {
      continue;
    }
    summary.push(`${key}: ${value}`);
    if (summary.length >= 4) {
      break;
    }
  }
  return summary;
}

function observedFacetSummaryFromRecord(record: JsonRecord): string[] {
  const facets = record.observed_facets;
  if (!isJsonRecord(facets)) {
    return [];
  }
  const summary: string[] = [];
  for (const [groupName, groupValue] of Object.entries(facets)) {
    const safeGroupName = safeTraceLabel(groupName);
    if (!safeGroupName || !isJsonRecord(groupValue)) {
      continue;
    }
    for (const [facetName, facetValue] of Object.entries(groupValue)) {
      const safeFacetName = safeTraceLabel(facetName);
      if (!safeFacetName || !isJsonRecord(facetValue)) {
        continue;
      }
      for (const [facetValueName, countValue] of Object.entries(facetValue)) {
        const safeFacetValueName = safeTraceLabel(facetValueName);
        if (
          !safeFacetValueName ||
          typeof countValue !== "number" ||
          !Number.isFinite(countValue) ||
          countValue <= 0
        ) {
          continue;
        }
        summary.push(
          `${safeGroupName}.${safeFacetName} ${safeFacetValueName}: ${countValue}`
        );
        if (summary.length >= 6) {
          return summary;
        }
      }
    }
  }
  return summary;
}

function scalarFeatureText(key: string, value: unknown): string | null {
  if (typeof value === "boolean" || typeof value === "number" || typeof value === "string") {
    return `${key}: ${String(value)}`;
  }
  return null;
}

function featureSummaryFromRecord(record: JsonRecord): string[] {
  const features = record.selection_features;
  if (!isJsonRecord(features)) {
    return [];
  }
  return Object.entries(features)
    .map(([key, value]) => scalarFeatureText(key, value))
    .filter((value): value is string => Boolean(value))
    .slice(0, 4);
}

function selectionItemFromRecord(record: JsonRecord): SelectionDiagnosticItem | null {
  const itemId = stringFromRecord(record, "item_id");
  if (!itemId) {
    return null;
  }
  const itemType = stringFromRecord(record, "item_type") ?? "context";
  return {
    itemId,
    itemType,
    label: stringFromRecord(record, "label") ?? itemId,
    contextRole: stringFromRecord(record, "context_role") ?? itemType,
    contextReason: stringFromRecord(record, "context_reason") ?? "Selected for this run.",
    selectionScore: scoreFromRecord(record, "selection_score"),
    featureSummary: featureSummaryFromRecord(record)
  };
}

function selectionDiagnosticsFromRun(
  run: ResearchAgentRunSummary | null | undefined
): SelectionDiagnostics | null {
  const diagnostics = run?.context_pack?.selection_diagnostics;
  if (!isJsonRecord(diagnostics)) {
    return null;
  }
  const groups = ["papers", "chunks", "memory_records", "graph_items"];
  const items: SelectionDiagnosticItem[] = [];
  for (const group of groups) {
    const groupItems = diagnostics[group];
    if (!Array.isArray(groupItems)) {
      continue;
    }
    for (const item of groupItems) {
      if (!isJsonRecord(item)) {
        continue;
      }
      const diagnosticItem = selectionItemFromRecord(item);
      if (diagnosticItem) {
        items.push(diagnosticItem);
      }
      if (items.length >= 8) {
        break;
      }
    }
    if (items.length >= 8) {
      break;
    }
  }
  return items.length > 0 ? { items } : null;
}

function readToolItemsFromValue(
  value: unknown,
  state: ReadToolDiagnosticItem["state"]
): ReadToolDiagnosticItem[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const items: ReadToolDiagnosticItem[] = [];
  for (const item of value) {
    if (!isJsonRecord(item)) {
      continue;
    }
    const name = stringFromRecord(item, "name");
    if (!name) {
      continue;
    }
    items.push({
      name,
      state,
      contextGroups: stringListFromRecord(item, "context_groups"),
      missingRequirements:
        state === "blocked" ? stringListFromRecord(item, "missing_requirements") : [],
      selectedContextCounts: selectedContextCountSummaryFromRecord(item),
      sideEffects: item.side_effects === true
    });
    if (items.length >= 6) {
      break;
    }
  }
  return items;
}

function readToolReadinessFromRun(
  run: ResearchAgentRunSummary | null | undefined
): ReadToolReadinessDiagnostics | null {
  const traceStep = [...(run?.steps ?? [])]
    .reverse()
    .find((step) => step.step_type === "tool_call" || step.step_type === "plan");
  const readTools = traceStep?.output_json?.read_tools;
  if (!isJsonRecord(readTools)) {
    return null;
  }

  const availableItems = readToolItemsFromValue(readTools.available, "available");
  const blockedItems = readToolItemsFromValue(readTools.blocked, "blocked");
  if (availableItems.length === 0 && blockedItems.length === 0) {
    return null;
  }
  return {
    availableCount: availableItems.length,
    blockedCount: blockedItems.length,
    items: [...availableItems, ...blockedItems].slice(0, 8)
  };
}

function readToolObservationItemsFromValue(
  value: unknown,
  fallbackState: string
): ReadToolObservationItem[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const items: ReadToolObservationItem[] = [];
  for (const item of value) {
    if (!isJsonRecord(item)) {
      continue;
    }
    const name = stringFromRecord(item, "name");
    if (!name) {
      continue;
    }
    items.push({
      name,
      state: stringFromRecord(item, "status") ?? fallbackState,
      observedCounts: observedCountSummaryFromRecord(item),
      observedFacets: observedFacetSummaryFromRecord(item)
    });
    if (items.length >= 6) {
      break;
    }
  }
  return items;
}

function readToolObservationsFromRun(
  run: ResearchAgentRunSummary | null | undefined
): ReadToolObservationDiagnostics | null {
  const toolStep = [...(run?.steps ?? [])]
    .reverse()
    .find((step) => step.step_type === "tool_call");
  const toolObservations = toolStep?.output_json?.tool_observations;
  if (!isJsonRecord(toolObservations)) {
    return null;
  }

  const executedItems = readToolObservationItemsFromValue(
    toolObservations.executed,
    "completed"
  );
  const blockedItems = readToolObservationItemsFromValue(
    toolObservations.blocked,
    "blocked"
  );
  if (executedItems.length === 0 && blockedItems.length === 0) {
    return null;
  }
  return {
    executedCount: executedItems.length,
    blockedCount: blockedItems.length,
    items: [...executedItems, ...blockedItems].slice(0, 8)
  };
}

function supportCoverageFromRun(
  run: ResearchAgentRunSummary | null | undefined
): SupportCoverageDiagnostics | null {
  const supportCounts = run?.validation_report?.support_label_counts;
  if (!isJsonRecord(supportCounts)) {
    return null;
  }
  const diagnostics = {
    supportStatuses: countMapSummaryFromRecord(supportCounts, "support_statuses"),
    supportingLayers: countMapSummaryFromRecord(supportCounts, "supporting_layers")
  };
  return diagnostics.supportStatuses.length > 0 ||
    diagnostics.supportingLayers.length > 0
    ? diagnostics
    : null;
}

function referenceIntegrityFromRun(
  run: ResearchAgentRunSummary | null | undefined
): ReferenceIntegrityDiagnostics | null {
  const integrity = run?.validation_report?.reference_integrity;
  if (!isJsonRecord(integrity)) {
    return null;
  }
  const diagnostics = {
    artifactReferences: countFromRecord(integrity, "artifact_references"),
    validReferences: countFromRecord(integrity, "valid_references"),
    invalidReferences: countFromRecord(integrity, "invalid_references"),
    unverifiableReferences: countFromRecord(integrity, "unverifiable_references"),
    incompatibleReferenceTypes: countFromRecord(
      integrity,
      "incompatible_reference_types"
    )
  };
  const total =
    diagnostics.artifactReferences +
    diagnostics.validReferences +
    diagnostics.invalidReferences +
    diagnostics.unverifiableReferences +
    diagnostics.incompatibleReferenceTypes;
  return total > 0 ? diagnostics : null;
}

function supportScreeningItemFromRecord(
  record: JsonRecord,
  key: string,
  label: string
): SupportScreeningItem | null {
  const value = record[key];
  if (!isJsonRecord(value)) {
    return null;
  }
  const item = {
    label,
    checkedRecommendations: countFromRecord(value, "checked_recommendations"),
    weakRecommendations: countFromRecord(value, "weak_recommendations"),
    unavailableReferences: countFromRecord(value, "unavailable_references")
  };
  const total =
    item.checkedRecommendations + item.weakRecommendations + item.unavailableReferences;
  return total > 0 ? item : null;
}

function supportScreeningFromRun(
  run: ResearchAgentRunSummary | null | undefined
): SupportScreeningDiagnostics | null {
  const screening = run?.validation_report?.support_screening;
  if (!isJsonRecord(screening)) {
    return null;
  }
  const items = [
    supportScreeningItemFromRecord(screening, "chunk_span", "chunk/span"),
    supportScreeningItemFromRecord(
      screening,
      "structured_evidence",
      "structured evidence"
    ),
    supportScreeningItemFromRecord(
      screening,
      "source_fact_memory",
      "source facts"
    ),
    supportScreeningItemFromRecord(screening, "entailment", "entailment")
  ].filter((item): item is SupportScreeningItem => Boolean(item));
  return items.length > 0 ? { items } : null;
}

function recommendationHealthFromRun(
  run: ResearchAgentRunSummary | null | undefined
): RecommendationHealthDiagnostics | null {
  const health = run?.validation_report?.recommendation_health;
  if (!isJsonRecord(health)) {
    return null;
  }
  const diagnostics = {
    recommendationCount: countFromRecord(health, "recommendation_count"),
    missingSupportStatus: countFromRecord(health, "missing_support_status"),
    invalidSupportStatus: countFromRecord(health, "invalid_support_status"),
    missingSupportingLayers: countFromRecord(health, "missing_supporting_layers"),
    missingEvidenceReferences: countFromRecord(
      health,
      "missing_evidence_references"
    ),
    speculativeAsSupported: countFromRecord(health, "speculative_as_supported"),
    unavailableLayers: countFromRecord(health, "unavailable_layers"),
    invalidLayers: countFromRecord(health, "invalid_layers"),
    sourceFactMarkedSupported: countFromRecord(
      health,
      "source_fact_marked_supported"
    ),
    availableSupportingLayers: stringListFromRecord(
      health,
      "available_supporting_layers"
    )
  };
  const total =
    diagnostics.recommendationCount +
    diagnostics.missingSupportStatus +
    diagnostics.invalidSupportStatus +
    diagnostics.missingSupportingLayers +
    diagnostics.missingEvidenceReferences +
    diagnostics.speculativeAsSupported +
    diagnostics.unavailableLayers +
    diagnostics.invalidLayers +
    diagnostics.sourceFactMarkedSupported +
    diagnostics.availableSupportingLayers.length;
  return total > 0 ? diagnostics : null;
}

function validationIssueItemsFromValue(value: unknown): ValidationIssueItem[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const items: ValidationIssueItem[] = [];
  for (const item of value.slice(0, VALIDATION_ISSUE_SCAN_LIMIT)) {
    if (!isJsonRecord(item)) {
      continue;
    }
    const category = stringFromRecord(item, "category");
    const severity = stringFromRecord(item, "severity");
    const code = stringFromRecord(item, "code");
    const source = stringFromRecord(item, "source");
    const message = stringFromRecord(item, "message");
    const remediation = stringFromRecord(item, "remediation");
    const safeMessage = message ? safeTraceMessage(message) : null;
    const safeRemediation = remediation ? safeTraceMessage(remediation) : null;
    const safeCode = code ? safeTraceLabel(code) : null;
    if (!safeCode || !safeMessage) {
      continue;
    }
    items.push({
      category: category ? safeTraceLabel(category) ?? "validation" : "validation",
      severity: severity ? safeTraceLabel(severity) ?? "issue" : "issue",
      code: safeCode,
      source: source ? safeTraceLabel(source) ?? "report" : "report",
      message: safeMessage,
      remediation: safeRemediation ?? ""
    });
    if (items.length >= 6) {
      break;
    }
  }
  return items;
}

function validationSeveritySummary(items: ValidationIssueItem[]): string[] {
  const counts = new Map<string, number>();
  for (const item of items) {
    counts.set(item.severity, (counts.get(item.severity) ?? 0) + 1);
  }
  return Array.from(counts.entries()).map(([severity, count]) => `${severity}: ${count}`);
}

function validationIssuesFromRun(
  run: ResearchAgentRunSummary | null | undefined
): ValidationIssueDiagnostics | null {
  const items = validationIssueItemsFromValue(
    run?.validation_report?.validation_issues
  );
  if (items.length === 0) {
    return null;
  }
  const issueCounts = run?.validation_report?.validation_issue_counts;
  const totalIssueCount =
    isJsonRecord(issueCounts) &&
    typeof issueCounts.total === "number" &&
    Number.isFinite(issueCounts.total) &&
    issueCounts.total > items.length
      ? issueCounts.total
      : items.length;
  return {
    issueCount: totalIssueCount,
    severitySummary: validationSeveritySummary(items),
    items
  };
}

function taskEvidenceRolesFromRun(
  run: ResearchAgentRunSummary | null | undefined
): TaskEvidenceRoleDiagnostics | null {
  const roles = run?.validation_report?.task_evidence_roles;
  if (!isJsonRecord(roles)) {
    return null;
  }
  const diagnostics = {
    availableGroups: stringListFromRecord(roles, "available_groups"),
    satisfiedGroups: stringListFromRecord(roles, "satisfied_groups"),
    missingGroups: stringListFromRecord(roles, "missing_groups"),
    referencedRoles: stringListFromRecord(roles, "referenced_roles"),
    availableRoles: stringListFromRecord(roles, "available_roles")
  };
  return [
    diagnostics.availableGroups,
    diagnostics.satisfiedGroups,
    diagnostics.missingGroups,
    diagnostics.referencedRoles,
    diagnostics.availableRoles
  ].some((items) => items.length > 0)
    ? diagnostics
    : null;
}

function taskQualityFromRun(
  run: ResearchAgentRunSummary | null | undefined
): TaskQualityDiagnostics | null {
  const quality = run?.validation_report?.task_quality;
  if (!isJsonRecord(quality)) {
    return null;
  }
  const artifactType = stringFromRecord(quality, "artifact_type");
  const diagnostics = {
    artifactType: artifactType ? safeTraceLabel(artifactType) : null,
    checkedArtifactCount: numberFromRecord(quality, "checked_artifact_count"),
    requiredSections: stringListFromRecord(quality, "required_sections"),
    missingRequiredSections: stringListFromRecord(
      quality,
      "missing_required_sections"
    )
  };
  return diagnostics.artifactType ||
    diagnostics.checkedArtifactCount > 0 ||
    diagnostics.requiredSections.length > 0 ||
    diagnostics.missingRequiredSections.length > 0
    ? diagnostics
    : null;
}

function describeSchemaValidation(diagnostics: SchemaValidationDiagnostics): string {
  const label = diagnostics.status === "repaired" ? "Schema repair" : "Schema validation";
  const attempts = diagnostics.attempts;
  const attemptText =
    attempts && attempts > 0
      ? `${attempts} attempt${attempts === 1 ? "" : "s"}`
      : "attempt count unavailable";
  return `${label}: ${diagnostics.status} · ${diagnostics.schemaName} · ${attemptText}`;
}

function describeRetrievalSummary(diagnostics: RetrievalSummaryDiagnostics): string {
  return [
    `Retrieval: backend ${diagnostics.backendStatus}`,
    `${diagnostics.retrievalMatchCount} matched papers`,
    `${diagnostics.selectedChunkCount} chunks`,
    `${diagnostics.sqlChunkCount} SQL chunks`,
    `${diagnostics.selectedFigureCount} figures`,
    `${diagnostics.selectedTableCount} tables`,
    `${diagnostics.selectedStructuredEntityCount} entities`,
    `${diagnostics.selectedResultEvidenceCount} result evidence`
  ].join(" · ");
}

function describeContextMaterialization(
  diagnostics: ContextMaterializationDiagnostics
): string {
  return [
    "Context map:",
    `${diagnostics.methodCount} methods`,
    `${diagnostics.datasetCount} datasets`,
    `${diagnostics.metricCount} metrics`,
    `${diagnostics.baselineCount} baselines`,
    `${diagnostics.benchmarkTableCount} benchmark rows`,
    `${diagnostics.limitationCount} limitations`,
    `${diagnostics.limitationCategoryCount} limitation categories`,
    `${diagnostics.resultEvidenceCount} result evidence`,
    `${diagnostics.claimCount} claims`,
    `${diagnostics.claimEvidenceSpanCount} claim spans`,
    `${diagnostics.claimEvidenceMapCount} claim maps`
  ].join(" · ");
}

function describeContextCache(diagnostics: ContextCacheDiagnostics): string {
  if (diagnostics.status === "hit") {
    const sourceDetail =
      diagnostics.sourceContextPackId && diagnostics.sourceAttemptNumber !== null
        ? ` from attempt ${diagnostics.sourceAttemptNumber}`
        : "";
    return `Context cache: hit${sourceDetail}`;
  }
  if (diagnostics.status === "miss") {
    return "Context cache: fresh build";
  }
  return `Context cache: ${diagnostics.status}`;
}

function describeSelectionDiagnostics(diagnostics: SelectionDiagnostics): string {
  const count = diagnostics.items.length;
  return `Selection reasons: ${count} item${count === 1 ? "" : "s"}`;
}

function describeReadToolReadiness(diagnostics: ReadToolReadinessDiagnostics): string {
  return `Tool readiness: ${diagnostics.availableCount} available tools · ${diagnostics.blockedCount} blocked tools`;
}

function describeReadToolObservations(
  diagnostics: ReadToolObservationDiagnostics
): string {
  return `Tool observations: ${diagnostics.executedCount} executed tools · ${diagnostics.blockedCount} blocked tools`;
}

function describeValidationIssues(diagnostics: ValidationIssueDiagnostics): string {
  const severityText =
    diagnostics.severitySummary.length > 0
      ? ` · ${diagnostics.severitySummary.join(", ")}`
      : "";
  return `Validation issues: ${diagnostics.issueCount}${severityText}`;
}

function describeSupportCoverage(diagnostics: SupportCoverageDiagnostics): string {
  const parts = [
    diagnostics.supportStatuses.length > 0
      ? `support statuses ${diagnostics.supportStatuses.join(", ")}`
      : "",
    diagnostics.supportingLayers.length > 0
      ? `supporting layers ${diagnostics.supportingLayers.join(", ")}`
      : ""
  ].filter(Boolean);
  return `Support coverage: ${parts.join(" · ")}`;
}

function describeReferenceIntegrity(diagnostics: ReferenceIntegrityDiagnostics): string {
  const parts = [
    diagnostics.artifactReferences > 0
      ? `${diagnostics.artifactReferences} artifact refs`
      : "",
    diagnostics.validReferences > 0 ? `${diagnostics.validReferences} valid refs` : "",
    diagnostics.invalidReferences > 0
      ? `${diagnostics.invalidReferences} invalid refs`
      : "",
    diagnostics.unverifiableReferences > 0
      ? `${diagnostics.unverifiableReferences} unverifiable refs`
      : "",
    diagnostics.incompatibleReferenceTypes > 0
      ? `${diagnostics.incompatibleReferenceTypes} incompatible types`
      : ""
  ].filter(Boolean);
  return `Reference integrity: ${parts.join(" · ")}`;
}

function describeSupportScreening(diagnostics: SupportScreeningDiagnostics): string {
  const parts = diagnostics.items.map((item) => {
    const itemParts = [
      `${item.label} ${item.checkedRecommendations} checked`,
      item.weakRecommendations > 0 ? `${item.weakRecommendations} weak` : "",
      item.unavailableReferences > 0
        ? `${item.unavailableReferences} unavailable refs`
        : ""
    ].filter(Boolean);
    return itemParts.join(", ");
  });
  return `Support screening: ${parts.join(" · ")}`;
}

function describeRecommendationHealth(
  diagnostics: RecommendationHealthDiagnostics
): string {
  const parts = [
    diagnostics.recommendationCount > 0
      ? `${diagnostics.recommendationCount} recommendations`
      : "",
    diagnostics.missingSupportStatus > 0
      ? `${diagnostics.missingSupportStatus} missing status`
      : "",
    diagnostics.invalidSupportStatus > 0
      ? `${diagnostics.invalidSupportStatus} invalid status`
      : "",
    diagnostics.missingSupportingLayers > 0
      ? `${diagnostics.missingSupportingLayers} missing layers`
      : "",
    diagnostics.missingEvidenceReferences > 0
      ? `${diagnostics.missingEvidenceReferences} missing refs`
      : "",
    diagnostics.speculativeAsSupported > 0
      ? `${diagnostics.speculativeAsSupported} speculative supported`
      : "",
    diagnostics.unavailableLayers > 0
      ? `${diagnostics.unavailableLayers} unavailable layers`
      : "",
    diagnostics.invalidLayers > 0
      ? `${diagnostics.invalidLayers} invalid layers`
      : "",
    diagnostics.sourceFactMarkedSupported > 0
      ? `${diagnostics.sourceFactMarkedSupported} source-fact supported`
      : "",
    diagnostics.availableSupportingLayers.length > 0
      ? `available layers ${diagnostics.availableSupportingLayers.join(", ")}`
      : ""
  ].filter(Boolean);
  return `Recommendation health: ${parts.join(" · ")}`;
}

function describeTaskEvidenceRoles(diagnostics: TaskEvidenceRoleDiagnostics): string {
  const parts = [
    diagnostics.missingGroups.length > 0
      ? `missing groups ${diagnostics.missingGroups.join(", ")}`
      : "",
    diagnostics.satisfiedGroups.length > 0
      ? `satisfied groups ${diagnostics.satisfiedGroups.join(", ")}`
      : "",
    diagnostics.availableGroups.length > 0
      ? `available groups ${diagnostics.availableGroups.join(", ")}`
      : "",
    diagnostics.referencedRoles.length > 0
      ? `referenced roles ${diagnostics.referencedRoles.join(", ")}`
      : "",
    diagnostics.availableRoles.length > 0
      ? `available roles ${diagnostics.availableRoles.join(", ")}`
      : ""
  ].filter(Boolean);
  return `Task evidence roles: ${parts.join(" · ")}`;
}

function describeTaskQuality(diagnostics: TaskQualityDiagnostics): string {
  const parts = [
    diagnostics.artifactType ? `artifact ${diagnostics.artifactType}` : "",
    diagnostics.checkedArtifactCount > 0
      ? `${diagnostics.checkedArtifactCount} checked artifact${
          diagnostics.checkedArtifactCount === 1 ? "" : "s"
        }`
      : "",
    diagnostics.missingRequiredSections.length > 0
      ? `missing sections ${diagnostics.missingRequiredSections.join(", ")}`
      : "",
    diagnostics.requiredSections.length > 0
      ? `required sections ${diagnostics.requiredSections.join(", ")}`
      : ""
  ].filter(Boolean);
  return `Task quality: ${parts.join(" · ")}`;
}

function hasStudyBriefProposalValue(value: unknown): boolean {
  if (typeof value === "string") {
    return value.trim().length > 0;
  }
  if (Array.isArray(value)) {
    return value.length > 0;
  }
  return value !== null && value !== undefined;
}

function describeStudyBriefProposalChange(change: StudyBriefProposalChange): string {
  const hadValue = hasStudyBriefProposalValue(change.before);
  const hasValue = hasStudyBriefProposalValue(change.after);
  if (!hadValue && hasValue) {
    return "adds content";
  }
  if (hadValue && !hasValue) {
    return "clears content";
  }
  return "updates content";
}

function studyBriefProposalStringList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter((item): item is string => typeof item === "string" && item.trim().length > 0)
    .slice(0, 8);
}

function formatStudyBriefProposalLinkedSources(
  value: unknown,
  studySources: StudySource[]
): string {
  const sourceIds = studyBriefProposalStringList(value);
  if (sourceIds.length === 0) {
    return "Linked source changes: none";
  }
  const sourceTitlesById = new Map(studySources.map((source) => [source.id, source.title]));
  const labels = sourceIds.map((sourceId) => sourceTitlesById.get(sourceId) ?? sourceId);
  return `Linked source changes: ${labels.join(", ")}`;
}

export interface StudyViewProps {
  selectedLibrary: CollectionSummary | null;
  collections: CollectionSummary[];
  threads: ResearchThreadSummary[];
  chatEntries: ChatEntry[];
  chatDraft: string;
  studyBrief: StudyBrief | null;
  studyBriefDraftText: StudyBriefDraftText;
  isStudyBriefReady: boolean;
  studySources: StudySource[];
  activeStudySourceIds: string[];
  artifactFolderPath: string;
  artifactFolderListing: ArtifactFolderListing | null;
  selectedArtifactFilePaths: string[];
  pendingArtifactFiles: PendingArtifactFileItem[];
  savedArtifacts: ResearchArtifactSummary[];
  runByArtifactId: Record<string, ResearchAgentRunSummary>;
  studyBriefProposalCandidates: ResearchArtifactSummary[];
  studyBriefProposal: StudyBriefProposal | null;
  studyBriefProposalArtifactId: string;
  studyBriefProposalError: string | null;
  isStudyBriefProposalLoading: boolean;
  isStudyBriefProposalAccepting: boolean;
  recoveryActions: CollectionExtractionRecoveryAction[];
  modelProviderStatus: ModelProviderStatus | null;
  onSelectCollection: (collectionId: string) => void;
  onChatDraftChange: (draft: string) => void;
  onSubmitChat: (event: FormEvent<HTMLFormElement>) => void;
  onSuggestionClick: (suggestion: ChatSuggestion) => void;
  onQueueRecoveryAction: (action: CollectionExtractionRecoveryAction) => void;
  onRetryArtifact: (artifactId: string) => void;
  onGoToLibrary: () => void;
  onStudyBriefFieldChange: (field: "aim" | "hypothesis", value: string) => void;
  onStudyBriefItemsChange: (field: StudyBriefListField, value: string) => void;
  onSubmitStudyBrief: (event: FormEvent<HTMLFormElement>) => void;
  onStudyBriefProposalArtifactChange: (artifactId: string) => void;
  onLoadStudyBriefProposal: (artifactId: string) => void;
  onAcceptStudyBriefProposal: () => void;
  onSubmitSource: (event: FormEvent<HTMLFormElement>) => void;
  onArtifactFolderPathChange: (path: string) => void;
  onBrowseArtifactFolder: (event: FormEvent<HTMLFormElement>) => void;
  onOpenArtifactSubfolder: (relativePath: string) => void;
  onToggleArtifactFolderFile: (path: string) => void;
  onRegisterSelectedArtifactFiles: () => void;
  onArtifactFileSelection: (files: FileList | null) => void;
  onRegisterPendingArtifactFiles: () => void;
  onToggleStudySource: (sourceId: string) => void;
  onNewChat: () => void;
  onSelectThread: (threadId: string) => void;
}

export function StudyView({
  selectedLibrary,
  collections,
  threads,
  chatEntries,
  chatDraft,
  studyBrief,
  studyBriefDraftText,
  isStudyBriefReady,
  studySources,
  activeStudySourceIds,
  artifactFolderPath,
  artifactFolderListing,
  selectedArtifactFilePaths,
  pendingArtifactFiles,
  savedArtifacts,
  runByArtifactId,
  studyBriefProposalCandidates,
  studyBriefProposal,
  studyBriefProposalArtifactId,
  studyBriefProposalError,
  isStudyBriefProposalLoading,
  isStudyBriefProposalAccepting,
  recoveryActions,
  modelProviderStatus,
  onSelectCollection,
  onChatDraftChange,
  onSubmitChat,
  onSuggestionClick,
  onQueueRecoveryAction,
  onRetryArtifact,
  onGoToLibrary,
  onStudyBriefFieldChange,
  onStudyBriefItemsChange,
  onSubmitStudyBrief,
  onStudyBriefProposalArtifactChange,
  onLoadStudyBriefProposal,
  onAcceptStudyBriefProposal,
  onSubmitSource,
  onArtifactFolderPathChange,
  onBrowseArtifactFolder,
  onOpenArtifactSubfolder,
  onToggleArtifactFolderFile,
  onRegisterSelectedArtifactFiles,
  onArtifactFileSelection,
  onRegisterPendingArtifactFiles,
  onToggleStudySource,
  onNewChat,
  onSelectThread
}: StudyViewProps) {
  const isTextReady = Boolean(selectedLibrary && (selectedLibrary.parsed_paper_count ?? 0) > 0);
  const canChat = Boolean(selectedLibrary && isTextReady);
  const selectedSources = studySources.filter((source) =>
    activeStudySourceIds.includes(source.id)
  );
  const evidenceWarning =
    selectedLibrary && isTextReady && (selectedLibrary.extracted_paper_count ?? 0) < (selectedLibrary.paper_count ?? 0)
      ? "Evidence extraction is still incomplete; lightweight chat is available, but citation coverage may be limited."
      : null;

  return (
    <main className="workspace-grid study-grid">
      <aside className="sidebar study-sidebar" aria-label="Study chats">
        <section>
          <p className="section-label">Library</p>
          {collections.length > 0 ? (
            <select
              className="library-picker"
              value={selectedLibrary?.id ?? ""}
              onChange={(event) => onSelectCollection(event.target.value)}
            >
              {collections.map((collection) => (
                <option key={collection.id} value={collection.id}>
                  {collection.title}
                </option>
              ))}
            </select>
          ) : (
            <p className="muted">No library selected</p>
          )}
        </section>
        <button type="button" className="action-button" onClick={onNewChat} disabled={!selectedLibrary}>
          + New chat
        </button>
        <section>
          <p className="section-label">Recent chats</p>
          {threads.length > 0 ? (
            <ul className="plain-list">
              {threads.map((thread) => (
                <li key={thread.id}>
                  <button type="button" onClick={() => onSelectThread(thread.id)}>
                    {thread.title}
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted">No chats for this library yet.</p>
          )}
        </section>
      </aside>

      <section className="chat-panel" aria-label="Chat with Arxie">
        {!canChat ? (
          <div className="onboarding-card">
            <h2>No research library is ready yet</h2>
            <p>
              Arxie needs a prepared paper library before it can answer field-aware
              research questions. Go to Library to add papers and prepare the corpus.
            </p>
            <button type="button" onClick={onGoToLibrary}>
              Go to Library
            </button>
          </div>
        ) : null}

        {evidenceWarning ? <p className="warning-banner">{evidenceWarning}</p> : null}

        {selectedLibrary ? (
          <StudyRecoverySuggestions
            actions={recoveryActions}
            canQueue={Boolean(selectedLibrary)}
            onQueueRecoveryAction={onQueueRecoveryAction}
            onGoToLibrary={onGoToLibrary}
          />
        ) : null}

        {canChat && chatEntries.length === 0 ? (
          <div className="empty-chat-state text-reveal">
            <h2>What should Arxie work on first?</h2>
            <SuggestionStrip
              disabled={!canChat}
              selectedSources={selectedSources}
              modelProviderStatus={modelProviderStatus}
              onSuggestionClick={onSuggestionClick}
            />
          </div>
        ) : null}

        {chatEntries.map((entry) => (
          <article
            key={entry.id}
            className={`message chat-message-enter ${entry.role === "user" ? "user-message" : "assistant-message"}`}
          >
            <p>{entry.content}</p>
            {entry.role === "assistant" ? (
              <div className="evidence-block text-reveal">
                <h3>Evidence</h3>
                <p>{entry.status}</p>
                {entry.evidenceLines.length > 0 ? (
                  <ul>
                    {entry.evidenceLines.map((line) => (
                      <li key={line}>{line}</li>
                    ))}
                  </ul>
                ) : null}
                <BlockedArtifactRecovery
                  artifact={entry.artifact}
                  modelProviderStatus={modelProviderStatus}
                  onRetryArtifact={onRetryArtifact}
                />
                <ResearchRunTrace run={runByArtifactId[entry.id]} />
              </div>
            ) : null}
          </article>
        ))}

        {selectedLibrary && chatEntries.length > 0 ? (
          <SuggestionStrip
            disabled={!canChat}
            selectedSources={selectedSources}
            modelProviderStatus={modelProviderStatus}
            onSuggestionClick={onSuggestionClick}
          />
        ) : null}

        <form className="chat-input" onSubmit={onSubmitChat}>
          <input
            name="message"
            aria-label="Ask Arxie"
            value={chatDraft}
            onChange={(event) => onChatDraftChange(event.target.value)}
            placeholder={
              canChat ? "Ask Arxie..." : "Prepare or select a library before chatting"
            }
            disabled={!canChat}
          />
          <button type="submit" disabled={!canChat}>
            Send
          </button>
        </form>
      </section>

      <aside className="artifacts-sidebar" aria-label="My Artifacts">
        <StudyBriefEditor
          studyBrief={studyBrief}
          draftText={studyBriefDraftText}
          disabled={!selectedLibrary || !isStudyBriefReady}
          proposalCandidates={studyBriefProposalCandidates}
          proposal={studyBriefProposal}
          proposalArtifactId={studyBriefProposalArtifactId}
          proposalError={studyBriefProposalError}
          isProposalLoading={isStudyBriefProposalLoading}
          isProposalAccepting={isStudyBriefProposalAccepting}
          studySources={studySources}
          onFieldChange={onStudyBriefFieldChange}
          onItemsChange={onStudyBriefItemsChange}
          onSubmit={onSubmitStudyBrief}
          onProposalArtifactChange={onStudyBriefProposalArtifactChange}
          onLoadProposal={onLoadStudyBriefProposal}
          onAcceptProposal={onAcceptStudyBriefProposal}
        />
        <div className="sidebar-title-row">
          <h2>My Artifacts</h2>
          <span>{activeStudySourceIds.length} selected context</span>
        </div>
        <ArtifactIntake
          disabled={!selectedLibrary}
          artifactFolderPath={artifactFolderPath}
          artifactFolderListing={artifactFolderListing}
          selectedArtifactFilePaths={selectedArtifactFilePaths}
          pendingArtifactFiles={pendingArtifactFiles}
          onArtifactFolderPathChange={onArtifactFolderPathChange}
          onBrowseArtifactFolder={onBrowseArtifactFolder}
          onOpenArtifactSubfolder={onOpenArtifactSubfolder}
          onToggleArtifactFolderFile={onToggleArtifactFolderFile}
          onRegisterSelectedArtifactFiles={onRegisterSelectedArtifactFiles}
          onArtifactFileSelection={onArtifactFileSelection}
          onRegisterPendingArtifactFiles={onRegisterPendingArtifactFiles}
        />
        <ArtifactList
          studySources={studySources}
          activeStudySourceIds={activeStudySourceIds}
          savedArtifacts={savedArtifacts}
          onToggleStudySource={onToggleStudySource}
        />
        <form className="source-form" onSubmit={onSubmitSource}>
          <label>
            Type
            <select name="source_type" disabled={!selectedLibrary}>
              <option value="text">Note</option>
              <option value="code_path">Code path</option>
              <option value="draft_path">Draft path</option>
              <option value="results_path">Results path</option>
            </select>
          </label>
          <label>
            Title
            <input name="title" disabled={!selectedLibrary} />
          </label>
          <label>
            Note or path
            <textarea name="value" disabled={!selectedLibrary} />
          </label>
          <button type="submit" disabled={!selectedLibrary}>
            Add manual source
          </button>
        </form>
      </aside>
    </main>
  );
}

export function ResearchRunTrace({ run }: { run: ResearchAgentRunSummary | undefined }) {
  if (!run) {
    return null;
  }
  const schemaDiagnostics = schemaValidationFromRun(run);
  const retrievalDiagnostics = retrievalSummaryFromRun(run);
  const contextMaterializationDiagnostics = contextMaterializationFromRun(run);
  const contextCacheDiagnostics = contextCacheFromRun(run);
  const selectionDiagnostics = selectionDiagnosticsFromRun(run);
  const readToolDiagnostics = readToolReadinessFromRun(run);
  const readToolObservationDiagnostics = readToolObservationsFromRun(run);
  const supportCoverageDiagnostics = supportCoverageFromRun(run);
  const referenceIntegrityDiagnostics = referenceIntegrityFromRun(run);
  const supportScreeningDiagnostics = supportScreeningFromRun(run);
  const recommendationHealthDiagnostics = recommendationHealthFromRun(run);
  const validationIssueDiagnostics = validationIssuesFromRun(run);
  const taskEvidenceRoleDiagnostics = taskEvidenceRolesFromRun(run);
  const taskQualityDiagnostics = taskQualityFromRun(run);
  const contextSummary = run.context_pack?.context_summary;
  const paperCount =
    contextSummary && typeof contextSummary.paper_count === "number"
      ? contextSummary.paper_count
      : null;
  const validationStatus = run.validation_report?.harness_status ?? "waiting";
  const recentSteps = (run.steps ?? []).slice(-4);

  return (
    <section
      className="run-trace-panel"
      aria-label="Agent run trace"
      data-contract="React research run trace"
    >
      <div className="run-trace-summary">
        <span>Run {run.status}</span>
        {run.skill_id ? <span>{run.skill_id}</span> : null}
        {paperCount !== null ? <span>{paperCount} papers</span> : null}
      </div>
      <p>Validation: {validationStatus}</p>
      {referenceIntegrityDiagnostics ? (
        <p className="reference-integrity-diagnostics">
          {describeReferenceIntegrity(referenceIntegrityDiagnostics)}
        </p>
      ) : null}
      {supportScreeningDiagnostics ? (
        <p className="support-screening-diagnostics">
          {describeSupportScreening(supportScreeningDiagnostics)}
        </p>
      ) : null}
      {supportCoverageDiagnostics ? (
        <p className="support-coverage-diagnostics">
          {describeSupportCoverage(supportCoverageDiagnostics)}
        </p>
      ) : null}
      {recommendationHealthDiagnostics ? (
        <p className="recommendation-health-diagnostics">
          {describeRecommendationHealth(recommendationHealthDiagnostics)}
        </p>
      ) : null}
      {taskEvidenceRoleDiagnostics ? (
        <p className="task-evidence-role-diagnostics">
          {describeTaskEvidenceRoles(taskEvidenceRoleDiagnostics)}
        </p>
      ) : null}
      {taskQualityDiagnostics ? (
        <p className="task-quality-diagnostics">
          {describeTaskQuality(taskQualityDiagnostics)}
        </p>
      ) : null}
      {validationIssueDiagnostics ? (
        <details className="validation-issues">
          <summary>{describeValidationIssues(validationIssueDiagnostics)}</summary>
          <ul className="validation-issues-list">
            {validationIssueDiagnostics.items.map((item, index) => (
              <li key={`${item.code}-${index}`}>
                <span>
                  {item.code} <strong>{item.severity}</strong>
                </span>
                <small>
                  {[
                    item.category,
                    item.source,
                    item.message,
                    item.remediation ? `Next action: ${item.remediation}` : null
                  ]
                    .filter(Boolean)
                    .join(" · ")}
                </small>
              </li>
            ))}
          </ul>
        </details>
      ) : null}
      {schemaDiagnostics ? (
        <p className="schema-repair-diagnostics">
          {describeSchemaValidation(schemaDiagnostics)}
        </p>
      ) : null}
      {retrievalDiagnostics ? (
        <p className="retrieval-diagnostics">
          {describeRetrievalSummary(retrievalDiagnostics)}
        </p>
      ) : null}
      {contextMaterializationDiagnostics ? (
        <p className="context-materialization-diagnostics">
          {describeContextMaterialization(contextMaterializationDiagnostics)}
        </p>
      ) : null}
      {contextCacheDiagnostics ? (
        <p className="context-cache-diagnostics">
          {describeContextCache(contextCacheDiagnostics)}
        </p>
      ) : null}
      {readToolDiagnostics ? (
        <details className="read-tool-diagnostics">
          <summary>{describeReadToolReadiness(readToolDiagnostics)}</summary>
          <ul className="read-tool-diagnostics-list">
            {readToolDiagnostics.items.map((item) => (
              <li key={`${item.state}-${item.name}`}>
                <span>
                  {item.name} <strong>{item.state}</strong>
                </span>
                <small>
                  {[
                    item.sideEffects ? "side effects" : "read only",
                    item.contextGroups.length > 0
                      ? `groups ${item.contextGroups.join(", ")}`
                      : "",
                    item.selectedContextCounts.length > 0
                      ? `counts ${item.selectedContextCounts.join(", ")}`
                      : "",
                    item.missingRequirements.length > 0
                      ? `missing ${item.missingRequirements.join(", ")}`
                      : ""
                  ]
                    .filter(Boolean)
                    .join(" · ")}
                </small>
              </li>
            ))}
          </ul>
        </details>
      ) : null}
      {readToolObservationDiagnostics ? (
        <details className="read-tool-observations">
          <summary>{describeReadToolObservations(readToolObservationDiagnostics)}</summary>
          <ul className="read-tool-observations-list">
            {readToolObservationDiagnostics.items.map((item) => (
              <li key={`${item.state}-${item.name}`}>
                <span>
                  {item.name} <strong>{item.state}</strong>
                </span>
                <small>
                  {[...item.observedCounts, ...item.observedFacets]
                    .filter(Boolean)
                    .join(" · ")}
                </small>
              </li>
            ))}
          </ul>
        </details>
      ) : null}
      {selectionDiagnostics ? (
        <details className="selection-diagnostics">
          <summary>{describeSelectionDiagnostics(selectionDiagnostics)}</summary>
          <ul>
            {selectionDiagnostics.items.map((item) => (
              <li key={`${item.itemType}-${item.itemId}`}>
                <span>
                  {item.itemType}: {item.label}
                </span>
                <small>
                  {[
                    item.contextRole,
                    item.contextReason,
                    item.selectionScore !== null ? `score ${item.selectionScore}` : "",
                    ...item.featureSummary
                  ]
                    .filter(Boolean)
                    .join(" · ")}
                </small>
              </li>
            ))}
          </ul>
        </details>
      ) : null}
      {run.error_message ? <p className="form-error">{run.error_message}</p> : null}
      {recentSteps.length > 0 ? (
        <ul className="run-trace-steps">
          {recentSteps.map((step) => (
            <li key={step.id}>
              <span>{step.label || step.step_type}</span>
              <strong>{step.status}</strong>
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}

export function StudyBriefEditor({
  studyBrief,
  draftText,
  disabled,
  proposalCandidates,
  proposal,
  proposalArtifactId,
  proposalError,
  isProposalLoading,
  isProposalAccepting,
  studySources,
  onFieldChange,
  onItemsChange,
  onSubmit,
  onProposalArtifactChange,
  onLoadProposal,
  onAcceptProposal
}: {
  studyBrief: StudyBrief | null;
  draftText: StudyBriefDraftText;
  disabled: boolean;
  proposalCandidates: ResearchArtifactSummary[];
  proposal: StudyBriefProposal | null;
  proposalArtifactId: string;
  proposalError: string | null;
  isProposalLoading: boolean;
  isProposalAccepting: boolean;
  studySources: StudySource[];
  onFieldChange: (field: "aim" | "hypothesis", value: string) => void;
  onItemsChange: (field: StudyBriefListField, value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onProposalArtifactChange: (artifactId: string) => void;
  onLoadProposal: (artifactId: string) => void;
  onAcceptProposal: () => void;
}) {
  const studyBriefVersion = studyBrief?.version ?? 0;
  const selectedProposalArtifactId = proposalCandidates.some(
    (candidate) => candidate.id === proposalArtifactId
  )
    ? proposalArtifactId
    : proposalCandidates[0]?.id ?? "";

  return (
    <form
      className="study-brief-editor"
      data-contract="Versioned Study Brief editor"
      onSubmit={onSubmit}
    >
      <div className="sidebar-title-row">
        <h2>Study Brief</h2>
        <span>Version {studyBriefVersion}</span>
      </div>
      <section
        className="study-brief-proposal-panel"
        data-contract="Study Brief proposal review"
        aria-label="Study Brief proposal review"
      >
        <div className="study-brief-proposal-heading">
          <h3>Review agent proposal</h3>
          {proposal ? <small>Target version {proposal.current_version}</small> : null}
        </div>
        {proposalCandidates.length > 0 ? (
          <div className="study-brief-proposal-controls">
            <select
              aria-label="Study Brief proposal artifact"
              value={selectedProposalArtifactId}
              disabled={disabled || isProposalLoading || isProposalAccepting}
              onChange={(event) => onProposalArtifactChange(event.target.value)}
            >
              {proposalCandidates.map((candidate) => (
                <option key={candidate.id} value={candidate.id}>
                  {candidate.saved_title || candidate.title}
                </option>
              ))}
            </select>
            <button
              type="button"
              disabled={disabled || !selectedProposalArtifactId || isProposalLoading}
              onClick={() => onLoadProposal(selectedProposalArtifactId)}
            >
              {isProposalLoading ? "Applying..." : "Apply to draft"}
            </button>
          </div>
        ) : (
          <p className="muted">No completed brief proposals yet.</p>
        )}
        {proposal ? (
          <div className="study-brief-proposal-detail">
            <small>{proposal.artifact_title}</small>
            {proposal.changes.length > 0 ? (
              <ul>
                {proposal.changes.map((change) => (
                  <li key={change.field} className="study-brief-proposal-change">
                    <strong>{studyBriefProposalFieldLabels[change.field]}</strong>
                    {change.field === "linked_source_ids" ? (
                      <span className="study-brief-proposal-linked-source">
                        {formatStudyBriefProposalLinkedSources(change.after, studySources)}
                      </span>
                    ) : (
                      <span>{describeStudyBriefProposalChange(change)}</span>
                    )}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="muted">No field changes.</p>
            )}
            <button
              type="button"
              disabled={disabled || isProposalAccepting}
              onClick={onAcceptProposal}
            >
              {isProposalAccepting ? "Accepting..." : "Accept proposal"}
            </button>
          </div>
        ) : null}
        {proposalError ? (
          <p className="form-error" role="alert">
            {proposalError}
          </p>
        ) : null}
      </section>
      <label>
        Current aim
        <textarea
          value={draftText.aim}
          disabled={disabled}
          onChange={(event) => onFieldChange("aim", event.target.value)}
        />
      </label>
      <label>
        Working hypothesis
        <textarea
          value={draftText.hypothesis}
          disabled={disabled}
          onChange={(event) => onFieldChange("hypothesis", event.target.value)}
        />
      </label>
      <label>
        Constraints
        <textarea
          value={draftText.constraints}
          disabled={disabled}
          onChange={(event) => onItemsChange("constraints", event.target.value)}
        />
      </label>
      <label>
        Confirmed decisions
        <textarea
          value={draftText.confirmed_decisions}
          disabled={disabled}
          onChange={(event) => onItemsChange("confirmed_decisions", event.target.value)}
        />
      </label>
      <label>
        Open risks
        <textarea
          value={draftText.open_risks}
          disabled={disabled}
          onChange={(event) => onItemsChange("open_risks", event.target.value)}
        />
      </label>
      <button type="submit" disabled={disabled}>
        Save brief
      </button>
    </form>
  );
}

export function StudyRecoverySuggestions({
  actions,
  canQueue,
  onQueueRecoveryAction,
  onGoToLibrary
}: {
  actions: CollectionExtractionRecoveryAction[];
  canQueue: boolean;
  onQueueRecoveryAction: (action: CollectionExtractionRecoveryAction) => void;
  onGoToLibrary: () => void;
}) {
  const visibleActions = actions.filter((action) => {
    const hasQueueablePaperIds = action.can_queue_job && action.paper_ids.length > 0;
    return hasQueueablePaperIds || action.action_type === "review_evidence";
  });

  if (visibleActions.length === 0) {
    return null;
  }

  return (
    <section
      className="study-recovery-suggestions"
      aria-label="Study readiness suggestions"
      data-contract="Study suggestions reuse API-bounded recovery actions"
    >
      <p className="section-label">Readiness suggestions</p>
      <div className="study-recovery-suggestion-list">
        {visibleActions.map((action) => {
          const canQueueAction =
            canQueue && action.can_queue_job && action.paper_ids.length > 0;
          return (
            <article key={action.action_id} className="study-recovery-suggestion">
              <div>
                <strong>{action.label}</strong>
                <p>{action.description}</p>
                <small>
                  {action.paper_count} paper{action.paper_count === 1 ? "" : "s"}
                  {action.truncated ? " · first batch shown" : ""}
                  {action.action_type === "review_evidence"
                    ? ` · ${action.unresolved_evidence_span_count} unresolved span${
                        action.unresolved_evidence_span_count === 1 ? "" : "s"
                      }`
                    : ""}
                </small>
                <RecoveryActionSampleList action={action} />
              </div>
              {action.can_queue_job ? (
                <button
                  type="button"
                  aria-label={`Queue ${action.label} from Study`}
                  disabled={!canQueueAction}
                  onClick={() => onQueueRecoveryAction(action)}
                >
                  Queue recovery
                </button>
              ) : (
                <button
                  type="button"
                  aria-label={`Open Library for ${action.label}`}
                  onClick={onGoToLibrary}
                >
                  Open Library
                </button>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}

export function SuggestionStrip({
  disabled,
  selectedSources,
  modelProviderStatus,
  onSuggestionClick
}: {
  disabled: boolean;
  selectedSources: StudySource[];
  modelProviderStatus: ModelProviderStatus | null;
  onSuggestionClick: (suggestion: ChatSuggestion) => void;
}) {
  return (
    <div className="shortcut-strip" aria-label="Chat suggestions">
      {CHAT_SUGGESTIONS.map((suggestion) => {
        const requiresArtifact = Boolean(suggestion.requiresArtifact);
        const requiresModelProvider = suggestionRequiresReadyModelProvider(suggestion);
        const hasRequiredContext = suggestionHasRequiredContext(suggestion, selectedSources);
        const hasReadyModelProvider = suggestionHasReadyModelProvider(
          suggestion,
          modelProviderStatus
        );
        const suggestionId = `chat-suggestion-${suggestion.label
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, "-")}`;
        return (
          <SuggestionButton
            key={suggestion.label}
            disabled={disabled}
            hasReadyModelProvider={hasReadyModelProvider}
            hasRequiredContext={hasRequiredContext}
            requiresArtifact={requiresArtifact}
            requiresModelProvider={requiresModelProvider}
            suggestion={suggestion}
            suggestionId={suggestionId}
            onSuggestionClick={onSuggestionClick}
          />
        );
      })}
    </div>
  );
}

export function SuggestionButton({
  disabled,
  hasReadyModelProvider,
  hasRequiredContext,
  requiresArtifact,
  requiresModelProvider,
  suggestion,
  suggestionId,
  onSuggestionClick
}: {
  disabled: boolean;
  hasReadyModelProvider: boolean;
  hasRequiredContext: boolean;
  requiresArtifact: boolean;
  requiresModelProvider: boolean;
  suggestion: ChatSuggestion;
  suggestionId: string;
  onSuggestionClick: (suggestion: ChatSuggestion) => void;
}) {
  const blockerText = !hasRequiredContext
    ? "Attach a draft or note first"
    : !hasReadyModelProvider
      ? "Model setup required"
      : null;
  return (
    <button
      type="button"
      className="suggestion-button"
      data-chat-suggestion={suggestion.label}
      data-requires-artifact={requiresArtifact ? "true" : "false"}
      data-requires-model-provider={requiresModelProvider ? "true" : "false"}
      disabled={disabled || !hasRequiredContext || !hasReadyModelProvider}
      aria-describedby={blockerText ? suggestionId : undefined}
      onClick={() => onSuggestionClick(suggestion)}
    >
      <span>{suggestion.label}</span>
      {blockerText ? (
        <small id={suggestionId}>
          {blockerText}
          {!hasReadyModelProvider ? ". Open Settings for provider setup." : null}
        </small>
      ) : null}
    </button>
  );
}
