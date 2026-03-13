import {describe, expect, it, vi} from 'vitest';

import {
  createProposalShellM2Client,
  type ProposalBranchListContract,
  type ProposalConversationThreadContract,
  type ProposalEvidenceInspectorContract,
  type ProposalSessionResponseContract,
} from '../src/proposal-shell/proposalShellM2Client';

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      'content-type': 'application/json',
    },
  });
}

function makeSession(version: number): ProposalSessionResponseContract {
  return {
    session_id: 'session-1',
    version,
    state: {
      current_stage: 'idea_intake',
      stage_states: {
        idea_intake: {
          payload: {},
          confirmed: false,
        },
        logic_refinement: {
          payload: {},
          confirmed: false,
        },
        evidence_mapping: {
          payload: {},
          confirmed: false,
        },
        hypothesis_reshaping: {
          payload: {},
          confirmed: false,
        },
        data_feasibility_planning: {
          payload: {},
          confirmed: false,
        },
        experiment_analysis_design: {
          payload: {},
          confirmed: false,
        },
        proposal_assembly: {
          payload: {},
          confirmed: false,
        },
      },
    },
  };
}

describe('proposal shell M2 integration smoke', () => {
  it('loads shell state and bootstraps session on first load', async () => {
    const conversation: ProposalConversationThreadContract = {
      session_id: 'session-1',
      count: 1,
      messages: [
        {
          message_id: 'm-1',
          session_id: 'session-1',
          role: 'user',
          content: 'hello',
          metadata: {
            ui_source: 'dashboard',
          },
          created_at: '2026-03-13T00:00:00Z',
        },
      ],
    };
    const inspector: ProposalEvidenceInspectorContract = {
      session_id: 'session-1',
      count: 0,
      items: [],
    };
    const branches: ProposalBranchListContract = {
      session_id: 'session-1',
      count: 1,
      branches: [
        {
          session_id: 'session-1',
          branch_id: 'branch-a',
          name: 'Branch A',
          hypothesis: 'Hypothesis A',
          scorecard: {
            evidence_support: 0.7,
            feasibility: 0.8,
            risk: 0.3,
            impact: 0.9,
          },
          confidence_label: 'high',
          metadata: {},
          parent_branch_id: null,
          lineage: [],
          is_primary: true,
        },
      ],
    };

    const fetchMock = vi
      .fn<(input: URL | RequestInfo, init?: RequestInit) => Promise<Response>>()
      .mockResolvedValueOnce(jsonResponse({error: 'session_not_found'}, 404))
      .mockResolvedValueOnce(jsonResponse(makeSession(0), 201))
      .mockResolvedValueOnce(jsonResponse(conversation))
      .mockResolvedValueOnce(jsonResponse(inspector))
      .mockResolvedValueOnce(jsonResponse(branches));

    const client = createProposalShellM2Client({
      apiBase: 'http://localhost:8000',
      fetchImpl: fetchMock,
    });

    const shellState = await client.loadShellState('session-1');

    expect(shellState.session.version).toBe(0);
    expect(shellState.session.session_id).toBe('session-1');
    expect(shellState.conversation.count).toBe(1);
    expect(shellState.inspector.count).toBe(0);
    expect(shellState.branches).toHaveLength(1);
    expect(shellState.branches[0]?.branch_id).toBe('branch-a');

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      'http://localhost:8000/api/proposal/sessions/session-1',
      {method: 'GET'},
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      'http://localhost:8000/api/proposal/sessions',
      {
        method: 'POST',
        headers: {'content-type': 'application/json'},
        body: JSON.stringify({session_id: 'session-1'}),
      },
    );
  });

  it('saves stage payload and appends conversation message', async () => {
    const fetchMock = vi
      .fn<(input: URL | RequestInfo, init?: RequestInit) => Promise<Response>>()
      .mockResolvedValueOnce(jsonResponse(makeSession(1)))
      .mockResolvedValueOnce(
        jsonResponse({
          message_id: 'm-2',
          session_id: 'session-1',
          role: 'user',
          content: 'Saved update',
          metadata: {ui_source: 'dashboard'},
          created_at: '2026-03-13T01:00:00Z',
        }),
      );

    const client = createProposalShellM2Client({
      apiBase: 'http://localhost:8000',
      fetchImpl: fetchMock,
    });

    const saved = await client.saveStagePayload({
      sessionId: 'session-1',
      stage: 'idea_intake',
      expectedVersion: 0,
      payload: {
        problem: 'Insufficient literature support',
      },
    });
    const message = await client.createConversationMessage({
      sessionId: 'session-1',
      content: 'Saved update',
      role: 'user',
      metadata: {ui_source: 'dashboard'},
    });

    expect(saved.version).toBe(1);
    expect(saved.session_id).toBe('session-1');
    expect(message.message_id).toBe('m-2');
    expect(message.content).toBe('Saved update');

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      'http://localhost:8000/api/proposal/sessions/session-1/stages/idea_intake',
      {
        method: 'PATCH',
        headers: {'content-type': 'application/json'},
        body: JSON.stringify({
          expected_version: 0,
          payload: {
            problem: 'Insufficient literature support',
          },
        }),
      },
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      'http://localhost:8000/api/proposal/conversations/session-1/messages',
      {
        method: 'POST',
        headers: {'content-type': 'application/json'},
        body: JSON.stringify({
          role: 'user',
          content: 'Saved update',
          metadata: {ui_source: 'dashboard'},
        }),
      },
    );
  });

  it('promotes a branch through the primary branch endpoint', async () => {
    const fetchMock = vi
      .fn<(input: URL | RequestInfo, init?: RequestInit) => Promise<Response>>()
      .mockResolvedValueOnce(
        jsonResponse({
          session_id: 'session-1',
          branch_id: 'branch-a',
          name: 'Branch A',
          hypothesis: 'Hypothesis A',
          scorecard: {
            evidence_support: 0.7,
            feasibility: 0.8,
            risk: 0.3,
            impact: 0.9,
          },
          confidence_label: 'high',
          metadata: {},
          parent_branch_id: null,
          lineage: [],
          is_primary: true,
        }),
      );

    const client = createProposalShellM2Client({
      apiBase: 'http://localhost:8000',
      fetchImpl: fetchMock,
    });

    const promoted = await client.promoteBranch({
      sessionId: 'session-1',
      branchId: 'branch-a',
    });

    expect(promoted.branch_id).toBe('branch-a');
    expect(promoted.is_primary).toBe(true);
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      'http://localhost:8000/api/proposal/branches/session-1/branch-a/promote',
      {
        method: 'POST',
      },
    );
  });
});
