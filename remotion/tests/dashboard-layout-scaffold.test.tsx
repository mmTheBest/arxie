import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {describe, expect, it} from 'vitest';

import {
  DashboardLayoutScaffold,
  buildDashboardShellRoutes,
} from '../src/proposal-shell/DashboardLayoutScaffold';

describe('dashboard layout scaffold', () => {
  it('builds deterministic route shell paths from session id', () => {
    expect(buildDashboardShellRoutes('session-1')).toEqual({
      canvas: '/proposal/session-1/canvas',
      artifacts: '/proposal/session-1/artifacts',
      conversation: '/proposal/session-1/conversation',
    });
  });

  it('renders navigator, canvas, inspector, and conversation panes', () => {
    const markup = renderToStaticMarkup(
      <DashboardLayoutScaffold sessionId="session-1" activeRoute="canvas" />,
    );

    expect(markup).toContain('data-pane="navigator"');
    expect(markup).toContain('data-pane="canvas"');
    expect(markup).toContain('data-pane="inspector"');
    expect(markup).toContain('data-pane="conversation"');

    expect(markup).toContain('Workflow Navigator');
    expect(markup).toContain('Canvas Container');
    expect(markup).toContain('Evidence Inspector');
    expect(markup).toContain('Conversation Dock');
  });

  it('marks the active route tab as current page', () => {
    const markup = renderToStaticMarkup(
      <DashboardLayoutScaffold sessionId="session-1" activeRoute="artifacts" />,
    );

    expect(markup).toContain('data-route="artifacts"');
    expect(markup).toContain('aria-current="page"');
    expect(markup).toContain('Artifacts');
  });

  it('renders panel slot overrides without business logic wiring', () => {
    const markup = renderToStaticMarkup(
      <DashboardLayoutScaffold
        sessionId="session-1"
        activeRoute="canvas"
        navigatorSlot={<div>Custom navigator</div>}
        canvasSlot={<div>Custom canvas</div>}
        inspectorSlot={<div>Custom inspector</div>}
        conversationSlot={<div>Custom conversation</div>}
      />,
    );

    expect(markup).toContain('Custom navigator');
    expect(markup).toContain('Custom canvas');
    expect(markup).toContain('Custom inspector');
    expect(markup).toContain('Custom conversation');
  });
});
