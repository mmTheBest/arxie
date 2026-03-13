import {describe, expect, it} from 'vitest';
import {
  STAGE_TRANSITION_CONFIRMATION_INITIAL_STATE,
  buildStageTransitionConfirmationCopy,
  stageTransitionConfirmationReducer,
} from '../src/proposal-shell/useStageTransitionConfirmation';

describe('stage transition confirmation model', () => {
  it('opens confirmation state for pending transition', () => {
    const nextState = stageTransitionConfirmationReducer(
      STAGE_TRANSITION_CONFIRMATION_INITIAL_STATE,
      {
        type: 'open',
        transition: {
          fromStage: 'idea_intake',
          toStage: 'logic_refinement',
        },
      },
    );

    expect(nextState.isOpen).toBe(true);
    expect(nextState.pendingTransition?.fromStage).toBe('idea_intake');
    expect(nextState.pendingTransition?.toStage).toBe('logic_refinement');
  });

  it('resets to initial state on cancel', () => {
    const opened = stageTransitionConfirmationReducer(
      STAGE_TRANSITION_CONFIRMATION_INITIAL_STATE,
      {
        type: 'open',
        transition: {
          fromStage: 'idea_intake',
          toStage: 'logic_refinement',
        },
      },
    );

    const cancelled = stageTransitionConfirmationReducer(opened, {
      type: 'cancel',
    });

    expect(cancelled).toEqual(STAGE_TRANSITION_CONFIRMATION_INITIAL_STATE);
  });

  it('builds human-readable copy for modal body', () => {
    const copy = buildStageTransitionConfirmationCopy({
      fromStage: 'idea_intake',
      toStage: 'logic_refinement',
    });

    expect(copy.title).toContain('Move to Logic Refinement?');
    expect(copy.description).toContain('Idea Intake');
    expect(copy.description).toContain('Logic Refinement');
  });
});
