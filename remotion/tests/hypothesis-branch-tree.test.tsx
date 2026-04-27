import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {describe, expect, it} from 'vitest';

import {
  HypothesisBranchTreeScaffold,
  deriveHypothesisBranchTreeLanes,
  promotePrimaryBranchInMemory,
} from '../src/proposal-shell/HypothesisBranchTree';
import type {ProposalBranchContract} from '../src/proposal-shell/StageNavigator';

function makeBranch(
  partial: Partial<ProposalBranchContract> & Pick<ProposalBranchContract, 'branch_id' | 'name'>,
): ProposalBranchContract {
  return {
    session_id: 'session-1',
    branch_id: partial.branch_id,
    name: partial.name,
    hypothesis: partial.hypothesis ?? `${partial.name} hypothesis`,
    scorecard: partial.scorecard ?? {
      evidence_support: 0.6,
      feasibility: 0.6,
      risk: 0.3,
      impact: 0.7,
    },
    confidence_label: partial.confidence_label ?? 'medium',
    metadata: partial.metadata ?? {},
    parent_branch_id: partial.parent_branch_id ?? null,
    lineage: partial.lineage ?? [],
    is_primary: partial.is_primary ?? false,
  };
}

describe('hypothesis branch tree lane derivation', () => {
  it('maps primary branch to H1 and strongest non-primary branch to H2', () => {
    const root = makeBranch({
      branch_id: 'branch-root',
      name: 'Root',
      is_primary: true,
      confidence_label: 'medium',
    });
    const challenger = makeBranch({
      branch_id: 'branch-h2',
      name: 'Challenger',
      confidence_label: 'high',
      parent_branch_id: 'branch-root',
      lineage: ['branch-root'],
    });
    const alternative = makeBranch({
      branch_id: 'branch-alt',
      name: 'Alternative',
      confidence_label: 'low',
      parent_branch_id: 'branch-root',
      lineage: ['branch-root'],
    });

    const lanes = deriveHypothesisBranchTreeLanes([alternative, challenger, root]);

    expect(lanes.h1?.branch_id).toBe('branch-root');
    expect(lanes.h2?.branch_id).toBe('branch-h2');
    expect(lanes.alternatives.map((branch) => branch.branch_id)).toEqual(['branch-alt']);
  });

  it('falls back to first sorted branch as H1 when no primary branch is set', () => {
    const branches = [
      makeBranch({branch_id: 'branch-a', name: 'A', confidence_label: 'medium'}),
      makeBranch({branch_id: 'branch-b', name: 'B', confidence_label: 'high'}),
    ];

    const lanes = deriveHypothesisBranchTreeLanes(branches);

    expect(lanes.h1?.branch_id).toBe('branch-b');
    expect(lanes.h2?.branch_id).toBe('branch-a');
  });
});

describe('hypothesis branch tree scaffold', () => {
  it('renders H1/H2/Alternatives columns with confidence labels', () => {
    const markup = renderToStaticMarkup(
      <HypothesisBranchTreeScaffold
        branches={[
          makeBranch({
            branch_id: 'branch-h1',
            name: 'H1 Branch',
            is_primary: true,
            confidence_label: 'high',
          }),
          makeBranch({
            branch_id: 'branch-h2',
            name: 'H2 Branch',
            confidence_label: 'medium',
            parent_branch_id: 'branch-h1',
            lineage: ['branch-h1'],
          }),
          makeBranch({
            branch_id: 'branch-alt',
            name: 'Alternative',
            confidence_label: 'low',
            parent_branch_id: 'branch-h1',
            lineage: ['branch-h1'],
          }),
        ]}
        selectedPrimaryBranchId="branch-h1"
      />,
    );

    expect(markup).toContain('H1 (primary)');
    expect(markup).toContain('H2 (backup)');
    expect(markup).toContain('Alternatives');
    expect(markup).toContain('confidence: high');
    expect(markup).toContain('confidence: medium');
    expect(markup).toContain('Set as primary');
    expect(markup).toContain('data-primary-branch-id="branch-h1"');
  });

  it('renders an empty state when no branches exist', () => {
    const markup = renderToStaticMarkup(
      <HypothesisBranchTreeScaffold branches={[]} selectedPrimaryBranchId={null} />,
    );

    expect(markup).toContain('No hypothesis branches yet');
  });
});

describe('primary branch selection helper', () => {
  it('promotes selected branch and demotes all others', () => {
    const initial = [
      makeBranch({branch_id: 'root', name: 'Root', is_primary: true}),
      makeBranch({branch_id: 'alt', name: 'Alt', parent_branch_id: 'root'}),
      makeBranch({branch_id: 'alt-2', name: 'Alt 2', parent_branch_id: 'root'}),
    ];

    const updated = promotePrimaryBranchInMemory(initial, 'alt');

    expect(updated.find((branch) => branch.branch_id === 'alt')?.is_primary).toBe(true);
    expect(updated.find((branch) => branch.branch_id === 'root')?.is_primary).toBe(false);
    expect(updated.find((branch) => branch.branch_id === 'alt-2')?.is_primary).toBe(false);
  });
});
