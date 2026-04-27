import type {
  ProposalBranchContract,
  ProposalSessionContract,
  ProposalStageId,
} from './StageNavigator';

export type ProposalSessionResponseContract = ProposalSessionContract;

export interface ProposalConversationMessageContract {
  message_id: string;
  session_id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  metadata: Record<string, string>;
  created_at: string;
}

export interface ProposalConversationThreadContract {
  session_id: string;
  count: number;
  messages: ProposalConversationMessageContract[];
}

export interface ProposalEvidenceInspectorItemContract {
  paper_id: string;
  title: string;
  bucket: string;
  relevance_score: number;
  provenance_link: string | null;
}

export interface ProposalEvidenceInspectorContract {
  session_id: string;
  count: number;
  items: ProposalEvidenceInspectorItemContract[];
}

export interface ProposalBranchListContract {
  session_id: string;
  count: number;
  branches: ProposalBranchContract[];
}

export interface ProposalShellStateContract {
  session: ProposalSessionResponseContract;
  conversation: ProposalConversationThreadContract;
  inspector: ProposalEvidenceInspectorContract;
  branches: ProposalBranchContract[];
}

export interface SaveStagePayloadInput {
  sessionId: string;
  stage: ProposalStageId;
  expectedVersion: number;
  payload: Record<string, unknown>;
}

export interface CreateConversationMessageInput {
  sessionId: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  metadata?: Record<string, string>;
}

export interface PromoteBranchInput {
  sessionId: string;
  branchId: string;
}

export interface ProposalShellM2Client {
  loadShellState: (sessionId: string) => Promise<ProposalShellStateContract>;
  saveStagePayload: (
    input: SaveStagePayloadInput,
  ) => Promise<ProposalSessionResponseContract>;
  createConversationMessage: (
    input: CreateConversationMessageInput,
  ) => Promise<ProposalConversationMessageContract>;
  promoteBranch: (input: PromoteBranchInput) => Promise<ProposalBranchContract>;
}

export interface ProposalShellM2ClientOptions {
  apiBase: string;
  fetchImpl?: typeof fetch;
}

class ProposalShellApiError extends Error {
  status: number;
  details: unknown;

  constructor(message: string, status: number, details: unknown) {
    super(message);
    this.name = 'ProposalShellApiError';
    this.status = status;
    this.details = details;
  }
}

function normalizeApiBase(apiBase: string): string {
  return apiBase.trim().replace(/\/+$/, '');
}

function normalizeSessionId(sessionId: string): string {
  const normalized = sessionId.trim();
  if (!normalized) {
    throw new Error('sessionId must not be empty');
  }
  return normalized;
}

function normalizeBranchId(branchId: string): string {
  const normalized = branchId.trim();
  if (!normalized) {
    throw new Error('branchId must not be empty');
  }
  return normalized;
}

function toRequestUrl(apiBase: string, path: string): string {
  return `${normalizeApiBase(apiBase)}${path}`;
}

async function readResponseBody(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text.trim()) {
    return null;
  }

  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function pickApiErrorMessage(body: unknown, status: number): string {
  if (body && typeof body === 'object' && 'message' in body && typeof body.message === 'string') {
    return body.message;
  }
  if (body && typeof body === 'object' && 'error' in body && typeof body.error === 'string') {
    return body.error;
  }
  return `Request failed with status ${status}`;
}

async function requestJson<T>(
  fetchImpl: typeof fetch,
  url: string,
  init: RequestInit,
): Promise<T> {
  const response = await fetchImpl(url, init);
  const body = await readResponseBody(response);

  if (!response.ok) {
    throw new ProposalShellApiError(
      pickApiErrorMessage(body, response.status),
      response.status,
      body,
    );
  }

  return body as T;
}

export function createProposalShellM2Client(
  options: ProposalShellM2ClientOptions,
): ProposalShellM2Client {
  const fetchImpl = options.fetchImpl ?? fetch;
  const apiBase = normalizeApiBase(options.apiBase);

  const getSession = async (
    sessionId: string,
  ): Promise<ProposalSessionResponseContract> => {
    const normalizedSessionId = normalizeSessionId(sessionId);
    return requestJson(
      fetchImpl,
      toRequestUrl(apiBase, `/api/proposal/sessions/${encodeURIComponent(normalizedSessionId)}`),
      {method: 'GET'},
    );
  };

  const createSession = async (
    sessionId: string,
  ): Promise<ProposalSessionResponseContract> => {
    const normalizedSessionId = normalizeSessionId(sessionId);
    return requestJson(
      fetchImpl,
      toRequestUrl(apiBase, '/api/proposal/sessions'),
      {
        method: 'POST',
        headers: {'content-type': 'application/json'},
        body: JSON.stringify({session_id: normalizedSessionId}),
      },
    );
  };

  const loadOrCreateSession = async (
    sessionId: string,
  ): Promise<ProposalSessionResponseContract> => {
    try {
      return await getSession(sessionId);
    } catch (error) {
      if (error instanceof ProposalShellApiError && error.status === 404) {
        return createSession(sessionId);
      }
      throw error;
    }
  };

  const getConversation = async (
    sessionId: string,
  ): Promise<ProposalConversationThreadContract> => {
    const normalizedSessionId = normalizeSessionId(sessionId);
    return requestJson(
      fetchImpl,
      toRequestUrl(
        apiBase,
        `/api/proposal/conversations/${encodeURIComponent(normalizedSessionId)}/messages`,
      ),
      {method: 'GET'},
    );
  };

  const getInspector = async (
    sessionId: string,
  ): Promise<ProposalEvidenceInspectorContract> => {
    const normalizedSessionId = normalizeSessionId(sessionId);
    return requestJson(
      fetchImpl,
      toRequestUrl(
        apiBase,
        `/api/proposal/evidence/${encodeURIComponent(normalizedSessionId)}/inspector`,
      ),
      {method: 'GET'},
    );
  };

  const listBranches = async (
    sessionId: string,
  ): Promise<ProposalBranchListContract> => {
    const normalizedSessionId = normalizeSessionId(sessionId);
    return requestJson(
      fetchImpl,
      toRequestUrl(
        apiBase,
        `/api/proposal/branches/${encodeURIComponent(normalizedSessionId)}`,
      ),
      {method: 'GET'},
    );
  };

  const loadShellState = async (sessionId: string): Promise<ProposalShellStateContract> => {
    const session = await loadOrCreateSession(sessionId);
    const normalizedSessionId = normalizeSessionId(session.session_id);

    const [conversation, inspector, branchList] = await Promise.all([
      getConversation(normalizedSessionId),
      getInspector(normalizedSessionId),
      listBranches(normalizedSessionId),
    ]);

    return {
      session,
      conversation,
      inspector,
      branches: branchList.branches,
    };
  };

  const saveStagePayload = async (
    input: SaveStagePayloadInput,
  ): Promise<ProposalSessionResponseContract> => {
    const normalizedSessionId = normalizeSessionId(input.sessionId);
    return requestJson(
      fetchImpl,
      toRequestUrl(
        apiBase,
        `/api/proposal/sessions/${encodeURIComponent(normalizedSessionId)}/stages/${input.stage}`,
      ),
      {
        method: 'PATCH',
        headers: {'content-type': 'application/json'},
        body: JSON.stringify({
          expected_version: input.expectedVersion,
          payload: input.payload,
        }),
      },
    );
  };

  const createConversationMessage = async (
    input: CreateConversationMessageInput,
  ): Promise<ProposalConversationMessageContract> => {
    const normalizedSessionId = normalizeSessionId(input.sessionId);
    return requestJson(
      fetchImpl,
      toRequestUrl(
        apiBase,
        `/api/proposal/conversations/${encodeURIComponent(normalizedSessionId)}/messages`,
      ),
      {
        method: 'POST',
        headers: {'content-type': 'application/json'},
        body: JSON.stringify({
          role: input.role,
          content: input.content,
          metadata: input.metadata ?? {},
        }),
      },
    );
  };

  const promoteBranch = async (input: PromoteBranchInput): Promise<ProposalBranchContract> => {
    const normalizedSessionId = normalizeSessionId(input.sessionId);
    const normalizedBranchId = normalizeBranchId(input.branchId);
    return requestJson(
      fetchImpl,
      toRequestUrl(
        apiBase,
        `/api/proposal/branches/${encodeURIComponent(normalizedSessionId)}/${encodeURIComponent(normalizedBranchId)}/promote`,
      ),
      {
        method: 'POST',
      },
    );
  };

  return {
    loadShellState,
    saveStagePayload,
    createConversationMessage,
    promoteBranch,
  };
}
