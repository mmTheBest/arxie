import React from 'react';

export type DashboardRoute = 'canvas' | 'artifacts' | 'conversation';

export interface DashboardShellRoutes {
  canvas: string;
  artifacts: string;
  conversation: string;
}

const ROUTE_LABELS: Record<DashboardRoute, string> = {
  canvas: 'Canvas',
  artifacts: 'Artifacts',
  conversation: 'Conversation',
};

export function buildDashboardShellRoutes(sessionId: string): DashboardShellRoutes {
  const normalizedSessionId = sessionId.trim() || 'session';

  return {
    canvas: `/proposal/${normalizedSessionId}/canvas`,
    artifacts: `/proposal/${normalizedSessionId}/artifacts`,
    conversation: `/proposal/${normalizedSessionId}/conversation`,
  };
}

function ShellTab({
  route,
  href,
  isActive,
}: {
  route: DashboardRoute;
  href: string;
  isActive: boolean;
}): JSX.Element {
  return (
    <a
      href={href}
      data-route={route}
      aria-current={isActive ? 'page' : undefined}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        padding: '6px 10px',
        borderRadius: 999,
        border: '1px solid #d5d5d5',
        background: isActive ? '#ebf3ff' : '#ffffff',
        color: '#1f2328',
        textDecoration: 'none',
        fontSize: 12,
        fontWeight: 600,
      }}
    >
      {ROUTE_LABELS[route]}
    </a>
  );
}

export function DashboardLayoutScaffold({
  sessionId,
  activeRoute,
  navigatorSlot,
  canvasSlot,
  inspectorSlot,
  conversationSlot,
}: {
  sessionId: string;
  activeRoute: DashboardRoute;
  navigatorSlot?: React.ReactNode;
  canvasSlot?: React.ReactNode;
  inspectorSlot?: React.ReactNode;
  conversationSlot?: React.ReactNode;
}): JSX.Element {
  const routes = buildDashboardShellRoutes(sessionId);

  return (
    <section
      aria-label="Dashboard layout scaffold"
      style={{
        display: 'grid',
        gap: 10,
        width: '100%',
      }}
    >
      <header
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 8,
        }}
      >
        <strong style={{fontSize: 14}}>Proposal Workspace Shell</strong>
        <nav aria-label="Dashboard routes" style={{display: 'inline-flex', gap: 8}}>
          <ShellTab route="canvas" href={routes.canvas} isActive={activeRoute === 'canvas'} />
          <ShellTab
            route="artifacts"
            href={routes.artifacts}
            isActive={activeRoute === 'artifacts'}
          />
          <ShellTab
            route="conversation"
            href={routes.conversation}
            isActive={activeRoute === 'conversation'}
          />
        </nav>
      </header>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '280px minmax(0, 1fr) 320px',
          gridTemplateRows: 'minmax(320px, 1fr) 220px',
          gap: 10,
        }}
      >
        <aside
          data-pane="navigator"
          style={{
            border: '1px solid #d5d5d5',
            borderRadius: 10,
            padding: 12,
            background: '#ffffff',
          }}
        >
          {navigatorSlot ?? <div>Workflow Navigator</div>}
        </aside>

        <main
          data-pane="canvas"
          style={{
            border: '1px solid #d5d5d5',
            borderRadius: 10,
            padding: 12,
            background: '#ffffff',
          }}
        >
          {canvasSlot ?? <div>Canvas Container</div>}
        </main>

        <aside
          data-pane="inspector"
          style={{
            border: '1px solid #d5d5d5',
            borderRadius: 10,
            padding: 12,
            background: '#ffffff',
          }}
        >
          {inspectorSlot ?? <div>Evidence Inspector</div>}
        </aside>

        <section
          data-pane="conversation"
          style={{
            gridColumn: '1 / 4',
            border: '1px solid #d5d5d5',
            borderRadius: 10,
            padding: 12,
            background: '#ffffff',
          }}
        >
          {conversationSlot ?? <div>Conversation Dock</div>}
        </section>
      </div>
    </section>
  );
}
