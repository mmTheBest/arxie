import {useCallback, useMemo, useReducer} from 'react';

import {formatStageLabel, type ProposalStageId} from './StageNavigator';

export interface StageTransition {
  fromStage: ProposalStageId;
  toStage: ProposalStageId;
}

export interface StageTransitionConfirmationState {
  isOpen: boolean;
  pendingTransition: StageTransition | null;
}

type StageTransitionConfirmationAction =
  | {type: 'open'; transition: StageTransition}
  | {type: 'confirm'}
  | {type: 'cancel'};

export const STAGE_TRANSITION_CONFIRMATION_INITIAL_STATE: StageTransitionConfirmationState = {
  isOpen: false,
  pendingTransition: null,
};

export function stageTransitionConfirmationReducer(
  state: StageTransitionConfirmationState,
  action: StageTransitionConfirmationAction,
): StageTransitionConfirmationState {
  switch (action.type) {
    case 'open':
      return {
        isOpen: true,
        pendingTransition: action.transition,
      };
    case 'confirm':
    case 'cancel':
      return STAGE_TRANSITION_CONFIRMATION_INITIAL_STATE;
    default:
      return state;
  }
}

export function buildStageTransitionConfirmationCopy(transition: StageTransition): {
  title: string;
  description: string;
  confirmLabel: string;
  cancelLabel: string;
} {
  const fromLabel = formatStageLabel(transition.fromStage);
  const toLabel = formatStageLabel(transition.toStage);

  return {
    title: `Move to ${toLabel}?`,
    description: `This will close ${fromLabel} and switch the workspace to ${toLabel}.`,
    confirmLabel: 'Confirm transition',
    cancelLabel: 'Stay on current stage',
  };
}

export function useStageTransitionConfirmation(
  onConfirm?: (transition: StageTransition) => void,
): {
  state: StageTransitionConfirmationState;
  openConfirmation: (transition: StageTransition) => void;
  cancelConfirmation: () => void;
  confirmTransition: () => void;
  copy: ReturnType<typeof buildStageTransitionConfirmationCopy> | null;
} {
  const [state, dispatch] = useReducer(
    stageTransitionConfirmationReducer,
    STAGE_TRANSITION_CONFIRMATION_INITIAL_STATE,
  );

  const openConfirmation = useCallback((transition: StageTransition) => {
    dispatch({type: 'open', transition});
  }, []);

  const cancelConfirmation = useCallback(() => {
    dispatch({type: 'cancel'});
  }, []);

  const confirmTransition = useCallback(() => {
    if (state.pendingTransition) {
      onConfirm?.(state.pendingTransition);
    }
    dispatch({type: 'confirm'});
  }, [onConfirm, state.pendingTransition]);

  const copy = useMemo(() => {
    if (!state.pendingTransition) {
      return null;
    }

    return buildStageTransitionConfirmationCopy(state.pendingTransition);
  }, [state.pendingTransition]);

  return {
    state,
    openConfirmation,
    cancelConfirmation,
    confirmTransition,
    copy,
  };
}
