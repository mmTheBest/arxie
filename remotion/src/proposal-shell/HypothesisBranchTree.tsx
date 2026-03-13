import React from 'react';

import type {ProposalBranchContract} from './StageNavigator';

const CONFIDENCE_WEIGHT: Record<ProposalBranchContract['confidence_label'], number> = {
  low: 1,
  medium: 2,
  high: 3,
};

function computeBranchPriority(branch: ProposalBranchContract): number {
  const score =
    branch.scorecard.evidence_support +
    branch.scorecard.feasibility +
    branch.scorecard.impact -
    branch.scorecard.risk;
  return CONFIDENCE_WEIGHT[branch.confidence_label] * 10 + score;
}

function compareBranchesForTree(a: ProposalBranchContract, b: ProposalBranchContract): number {
  const byPriority = computeBranchPriority(b) - computeBranchPriority(a);
  if (byPriority !== 0) {
    return byPriority;
  }
  return a.name.localeCompare(b.name);
}

export interface HypothesisBranchTreeLanes {
  h1: ProposalBranchContract | null;
  h2: ProposalBranchContract | null;
  alternatives: ProposalBranchContract[];
}

export function deriveHypothesisBranchTreeLanes(
  branches: ProposalBranchContract[],
): HypothesisBranchTreeLanes {
  if (branches.length === 0) {
    return {
      h1: null,
      h2: null,
      alternatives: [],
    };
  }

  const sorted = [...branches].sort(compareBranchesForTree);
  const primary = branches.find((branch) => branch.is_primary) ?? sorted[0];

  if (!primary) {
    return {
      h1: null,
      h2: null,
      alternatives: [],
    };
  }

  const remaining = sorted.filter((branch) => branch.branch_id !== primary.branch_id);

  return {
    h1: primary,
    h2: remaining[0] ?? null,
    alternatives: remaining.slice(1),
  };
}

export function promotePrimaryBranchInMemory(
  branches: ProposalBranchContract[],
  targetBranchId: string,
): ProposalBranchContract[] {
  const normalizedTarget = targetBranchId.trim();
  if (!normalizedTarget) {
    return branches.map((branch) => ({...branch}));
  }

  const targetExists = branches.some((branch) => branch.branch_id === normalizedTarget);
  if (!targetExists) {
    return branches.map((branch) => ({...branch}));
  }

  return branches.map((branch) => ({
    ...branch,
    is_primary: branch.branch_id === normalizedTarget,
  }));
}

function HypothesisBranchCard({
  branch,
  disabled,
  onPrimaryBranchSelect,
}: {
  branch: ProposalBranchContract;
  disabled: boolean;
  onPrimaryBranchSelect?: (branchId: string) => void;
}): JSX.Element {
  const lineageText =
    branch.lineage.length > 0 ? `lineage: ${branch.lineage.join(' -> ')}` : 'lineage: root';

  return (
    <article
      data-branch-id={branch.branch_id}
      style={{
        border: '1px solid #d5d5d5',
        borderRadius: 10,
        padding: 10,
        background: '#ffffff',
        display: 'grid',
        gap: 8,
      }}
    >
      <header style={{display: 'grid', gap: 2}}>
        <strong style={{fontSize: 13}}>{branch.name}</strong>
        <span style={{fontSize: 12, color: '#5a6370'}}>{branch.hypothesis}</span>
      </header>
      <div style={{display: 'grid', gap: 2, fontSize: 12, color: '#5a6370'}}>
        <span>confidence: {branch.confidence_label}</span>
        <span>{lineageText}</span>
      </div>
      <div>
        <button
          type="button"
          data-select-primary={branch.branch_id}
          aria-pressed={branch.is_primary}
          disabled={disabled || branch.is_primary}
          onClick={() => onPrimaryBranchSelect?.(branch.branch_id)}
          style={{
            borderRadius: 999,
            border: '1px solid #d5d5d5',
            padding: '6px 10px',
            fontSize: 12,
            background: branch.is_primary ? '#f4fbf7' : '#ffffff',
          }}
        >
          {branch.is_primary ? 'Primary branch' : 'Set as primary'}
        </button>
      </div>
    </article>
  );
}

function BranchLane({
  title,
  branch,
  disabled,
  onPrimaryBranchSelect,
  placeholder,
}: {
  title: string;
  branch: ProposalBranchContract | null;
  disabled: boolean;
  onPrimaryBranchSelect?: (branchId: string) => void;
  placeholder: string;
}): JSX.Element {
  return (
    <section
      style={{
        display: 'grid',
        gap: 8,
        border: '1px solid #d5d5d5',
        borderRadius: 10,
        padding: 10,
        background: '#f8fafc',
      }}
    >
      <strong style={{fontSize: 12}}>{title}</strong>
      {branch ? (
        <HypothesisBranchCard
          branch={branch}
          disabled={disabled}
          onPrimaryBranchSelect={onPrimaryBranchSelect}
        />
      ) : (
        <span style={{fontSize: 12, color: '#5a6370'}}>{placeholder}</span>
      )}
    </section>
  );
}

export function HypothesisBranchTreeScaffold({
  branches,
  selectedPrimaryBranchId,
  disabled = false,
  onPrimaryBranchSelect,
}: {
  branches: ProposalBranchContract[];
  selectedPrimaryBranchId: string | null;
  disabled?: boolean;
  onPrimaryBranchSelect?: (branchId: string) => void;
}): JSX.Element {
  if (branches.length === 0) {
    return (
      <section aria-label="Hypothesis branch tree" style={{fontSize: 13}}>
        No hypothesis branches yet
      </section>
    );
  }

  const normalizedSelectedPrimary = selectedPrimaryBranchId?.trim() || null;
  const branchesWithSelection = normalizedSelectedPrimary
    ? promotePrimaryBranchInMemory(branches, normalizedSelectedPrimary)
    : branches;
  const lanes = deriveHypothesisBranchTreeLanes(branchesWithSelection);
  const primaryBranchId = lanes.h1?.branch_id ?? '';

  return (
    <section
      aria-label="Hypothesis branch tree"
      data-primary-branch-id={primaryBranchId}
      style={{display: 'grid', gap: 10}}
    >
      <header>
        <strong style={{fontSize: 14}}>Hypothesis Branch Tree</strong>
      </header>

      <div
        style={{
          display: 'grid',
          gap: 10,
          gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
        }}
      >
        <BranchLane
          title="H1 (primary)"
          branch={lanes.h1}
          disabled={disabled}
          onPrimaryBranchSelect={onPrimaryBranchSelect}
          placeholder="No H1 branch"
        />
        <BranchLane
          title="H2 (backup)"
          branch={lanes.h2}
          disabled={disabled}
          onPrimaryBranchSelect={onPrimaryBranchSelect}
          placeholder="No H2 branch"
        />
        <section
          style={{
            display: 'grid',
            gap: 8,
            border: '1px solid #d5d5d5',
            borderRadius: 10,
            padding: 10,
            background: '#f8fafc',
          }}
        >
          <strong style={{fontSize: 12}}>Alternatives</strong>
          {lanes.alternatives.length === 0 ? (
            <span style={{fontSize: 12, color: '#5a6370'}}>No alternatives</span>
          ) : (
            <div style={{display: 'grid', gap: 8}}>
              {lanes.alternatives.map((branch) => (
                <HypothesisBranchCard
                  key={branch.branch_id}
                  branch={branch}
                  disabled={disabled}
                  onPrimaryBranchSelect={onPrimaryBranchSelect}
                />
              ))}
            </div>
          )}
        </section>
      </div>
    </section>
  );
}
