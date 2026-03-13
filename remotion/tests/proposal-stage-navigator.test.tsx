import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {describe, expect, it} from 'vitest';
import {
  BranchSelectorScaffold,
  StageNavigatorScaffold,
  deriveStageNavigatorItems,
  type ProposalBranchContract,
  type ProposalSessionContract,
} from '../src/proposal-shell/StageNavigator';

const makeSession = (
  currentStage: string,
  confirmedStages: string[] = [],
): ProposalSessionContract => ({
  session_id: 'session-1',
  version: 3,
  state: {
    current_stage: currentStage,
    stage_states: {
      idea_intake: {
        payload: {},
        confirmed: confirmedStages.includes('idea_intake'),
      },
      logic_refinement: {
        payload: {},
        confirmed: confirmedStages.includes('logic_refinement'),
      },
      evidence_mapping: {
        payload: {},
        confirmed: confirmedStages.includes('evidence_mapping'),
      },
      hypothesis_reshaping: {
        payload: {},
        confirmed: confirmedStages.includes('hypothesis_reshaping'),
      },
      data_feasibility_planning: {
        payload: {},
        confirmed: confirmedStages.includes('data_feasibility_planning'),
      },
      experiment_analysis_design: {
        payload: {},
        confirmed: confirmedStages.includes('experiment_analysis_design'),
      },
      proposal_assembly: {
        payload: {},
        confirmed: confirmedStages.includes('proposal_assembly'),
      },
    },
  },
});

describe('proposal stage navigator', () => {
  it('maps current incomplete stage to needs_input and future stages to draft', () => {
    const items = deriveStageNavigatorItems(makeSession('idea_intake'));

    expect(items[0]?.status).toBe('needs_input');
    expect(items[1]?.status).toBe('draft');
    expect(items[2]?.status).toBe('draft');
  });

  it('maps current confirmed stage to ready', () => {
    const items = deriveStageNavigatorItems(
      makeSession('idea_intake', ['idea_intake']),
    );

    expect(items[0]?.status).toBe('ready');
  });

  it('maps previous stages to locked when workflow has advanced', () => {
    const items = deriveStageNavigatorItems(
      makeSession('logic_refinement', ['idea_intake']),
    );

    expect(items[0]?.status).toBe('locked');
    expect(items[1]?.status).toBe('needs_input');
  });

  it('renders badge labels and stage names for navigator scaffold', () => {
    const markup = renderToStaticMarkup(
      <StageNavigatorScaffold session={makeSession('idea_intake')} />,
    );

    expect(markup).toContain('Idea Intake');
    expect(markup).toContain('needs input');
  });
});

describe('branch selector scaffold', () => {
  it('renders placeholder when no branches are available', () => {
    const markup = renderToStaticMarkup(
      <BranchSelectorScaffold branches={[]} selectedBranchId={null} />,
    );

    expect(markup).toContain('No branches yet');
  });

  it('renders branch options and primary marker', () => {
    const branches: ProposalBranchContract[] = [
      {
        session_id: 'session-1',
        branch_id: 'branch-a',
        name: 'Branch A',
        hypothesis: 'A hypothesis',
        scorecard: {
          evidence_support: 0.8,
          feasibility: 0.6,
          risk: 0.3,
          impact: 0.7,
        },
        confidence_label: 'medium',
        metadata: {},
        parent_branch_id: null,
        lineage: [],
        is_primary: true,
      },
    ];

    const markup = renderToStaticMarkup(
      <BranchSelectorScaffold branches={branches} selectedBranchId="branch-a" />,
    );

    expect(markup).toContain('Branch A (primary)');
    expect(markup).toContain('value="branch-a"');
  });
});
