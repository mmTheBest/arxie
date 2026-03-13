import React from 'react';

export type ProposalStageId =
  | 'idea_intake'
  | 'logic_refinement'
  | 'evidence_mapping'
  | 'hypothesis_reshaping'
  | 'data_feasibility_planning'
  | 'experiment_analysis_design'
  | 'proposal_assembly';

export type StageBadgeStatus = 'draft' | 'needs_input' | 'ready' | 'locked';

export interface ProposalStageStateContract {
  payload: Record<string, unknown>;
  confirmed: boolean;
}

export interface ProposalSessionContract {
  session_id: string;
  version: number;
  state: {
    current_stage: string;
    stage_states: Record<string, ProposalStageStateContract>;
  };
}

export interface ProposalBranchContract {
  session_id: string;
  branch_id: string;
  name: string;
  hypothesis: string;
  scorecard: {
    evidence_support: number;
    feasibility: number;
    risk: number;
    impact: number;
  };
  confidence_label: 'low' | 'medium' | 'high';
  metadata: Record<string, string>;
  parent_branch_id: string | null;
  lineage: string[];
  is_primary: boolean;
}

export interface StageNavigatorItem {
  stageId: ProposalStageId;
  label: string;
  status: StageBadgeStatus;
  confirmed: boolean;
  isCurrent: boolean;
}

export const PROPOSAL_STAGE_SEQUENCE: ProposalStageId[] = [
  'idea_intake',
  'logic_refinement',
  'evidence_mapping',
  'hypothesis_reshaping',
  'data_feasibility_planning',
  'experiment_analysis_design',
  'proposal_assembly',
];

const STAGE_STATUS_LABEL: Record<StageBadgeStatus, string> = {
  draft: 'draft',
  needs_input: 'needs input',
  ready: 'ready',
  locked: 'locked',
};

export function formatStageLabel(stageId: string): string {
  return stageId
    .split('_')
    .filter(Boolean)
    .map((segment) => segment[0]?.toUpperCase() + segment.slice(1))
    .join(' ');
}

export function deriveStageNavigatorItems(
  session: ProposalSessionContract,
): StageNavigatorItem[] {
  const currentStage =
    PROPOSAL_STAGE_SEQUENCE.find((stage) => stage === session.state.current_stage) ??
    PROPOSAL_STAGE_SEQUENCE[0];

  const currentIndex = PROPOSAL_STAGE_SEQUENCE.indexOf(currentStage);

  return PROPOSAL_STAGE_SEQUENCE.map((stageId, index) => {
    const stageState = session.state.stage_states[stageId];
    const confirmed = Boolean(stageState?.confirmed);

    let status: StageBadgeStatus = 'draft';

    if (index < currentIndex) {
      status = 'locked';
    } else if (index === currentIndex) {
      status = confirmed ? 'ready' : 'needs_input';
    }

    return {
      stageId,
      label: formatStageLabel(stageId),
      status,
      confirmed,
      isCurrent: index === currentIndex,
    };
  });
}

export function StageNavigatorScaffold({
  session,
  onStageSelect,
}: {
  session: ProposalSessionContract;
  onStageSelect?: (stageId: ProposalStageId) => void;
}): JSX.Element {
  const stages = deriveStageNavigatorItems(session);

  return (
    <nav aria-label="Proposal workflow stages">
      <ol style={{display: 'grid', gap: 8, listStyle: 'none', margin: 0, padding: 0}}>
        {stages.map((stage) => (
          <li key={stage.stageId}>
            <button
              type="button"
              aria-current={stage.isCurrent ? 'step' : undefined}
              data-stage-id={stage.stageId}
              data-stage-status={stage.status}
              disabled={stage.status === 'locked'}
              onClick={() => onStageSelect?.(stage.stageId)}
              style={{
                width: '100%',
                display: 'flex',
                justifyContent: 'space-between',
                padding: '8px 10px',
                borderRadius: 8,
                border: '1px solid #d5d5d5',
                background: stage.isCurrent ? '#f4fbf7' : '#ffffff',
                fontSize: 13,
              }}
            >
              <span>{stage.label}</span>
              <span>{STAGE_STATUS_LABEL[stage.status]}</span>
            </button>
          </li>
        ))}
      </ol>
    </nav>
  );
}

export function BranchSelectorScaffold({
  branches,
  selectedBranchId,
  disabled = false,
  onSelect,
}: {
  branches: ProposalBranchContract[];
  selectedBranchId: string | null;
  disabled?: boolean;
  onSelect?: (branchId: string) => void;
}): JSX.Element {
  const hasBranches = branches.length > 0;

  return (
    <div style={{display: 'grid', gap: 6}}>
      <label htmlFor="branch-selector" style={{fontSize: 12, fontWeight: 600}}>
        Hypothesis branch
      </label>
      <select
        id="branch-selector"
        value={selectedBranchId ?? ''}
        disabled={disabled || !hasBranches}
        onChange={(event) => onSelect?.(event.target.value)}
        style={{
          borderRadius: 8,
          border: '1px solid #d5d5d5',
          padding: '8px 10px',
          fontSize: 13,
          background: '#ffffff',
        }}
      >
        {!hasBranches ? (
          <option value="">No branches yet</option>
        ) : (
          branches.map((branch) => (
            <option key={branch.branch_id} value={branch.branch_id}>
              {branch.name}
              {branch.is_primary ? ' (primary)' : ''}
            </option>
          ))
        )}
      </select>
    </div>
  );
}
