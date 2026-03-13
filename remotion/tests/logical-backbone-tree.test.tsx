import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {describe, expect, it} from 'vitest';

import {
  LogicalBackboneTreeScaffold,
  buildEvidenceInspectorSelection,
  deriveLogicalBackboneLevels,
  notifyEvidenceInspectorSelection,
  updateLogicalBackboneNodeClaimInMemory,
  type LogicalBackboneNode,
} from '../src/proposal-shell/LogicalBackboneTree';

function makeNode(
  partial: Partial<LogicalBackboneNode> & Pick<LogicalBackboneNode, 'node_id'>,
): LogicalBackboneNode {
  return {
    node_id: partial.node_id,
    claim: partial.claim ?? `Claim ${partial.node_id}`,
    parent_node_id: partial.parent_node_id ?? null,
    linked_paper_ids: partial.linked_paper_ids ?? [],
    confidence_label: partial.confidence_label ?? 'medium',
    weak_link_note: partial.weak_link_note ?? null,
  };
}

describe('logical backbone level derivation', () => {
  it('places roots/orphans first and children in deeper levels', () => {
    const levels = deriveLogicalBackboneLevels([
      makeNode({node_id: 'child-a', parent_node_id: 'root-a'}),
      makeNode({node_id: 'root-a'}),
      makeNode({node_id: 'orphan', parent_node_id: 'missing-root'}),
      makeNode({node_id: 'child-b', parent_node_id: 'root-a'}),
    ]);

    expect(levels).toHaveLength(2);
    expect(levels[0]?.nodes.map((node) => node.node_id)).toEqual(['orphan', 'root-a']);
    expect(levels[1]?.nodes.map((node) => node.node_id)).toEqual(['child-a', 'child-b']);
  });
});

describe('logical backbone in-memory edit helpers', () => {
  it('updates target claim and preserves other nodes', () => {
    const original = [
      makeNode({node_id: 'n-1', claim: 'Original claim'}),
      makeNode({node_id: 'n-2', claim: 'Second claim'}),
    ];

    const updated = updateLogicalBackboneNodeClaimInMemory(original, 'n-1', '  Revised claim  ');

    expect(updated[0]?.claim).toBe('Revised claim');
    expect(updated[1]?.claim).toBe('Second claim');
  });

  it('builds inspector selection payload from linked paper ids', () => {
    const selection = buildEvidenceInspectorSelection(
      [
        makeNode({
          node_id: 'n-1',
          linked_paper_ids: ['paper-1', 'paper-1', '  paper-2  ', ''],
        }),
      ],
      'n-1',
    );

    expect(selection).toEqual({
      nodeId: 'n-1',
      paperIds: ['paper-1', 'paper-2'],
    });
  });

  it('notifies the evidence inspector hook with normalized selection payload', () => {
    const calls: Array<{nodeId: string; paperIds: string[]}> = [];
    const selection = notifyEvidenceInspectorSelection(
      [
        makeNode({
          node_id: 'n-1',
          linked_paper_ids: ['paper-a', 'paper-a', '  paper-b  ', ''],
        }),
      ],
      'n-1',
      (nextSelection) => {
        calls.push(nextSelection);
      },
    );

    expect(selection).toEqual({
      nodeId: 'n-1',
      paperIds: ['paper-a', 'paper-b'],
    });
    expect(calls).toEqual([
      {
        nodeId: 'n-1',
        paperIds: ['paper-a', 'paper-b'],
      },
    ]);
  });

  it('does not notify evidence inspector hook when target node is missing', () => {
    const calls: Array<{nodeId: string; paperIds: string[]}> = [];
    const selection = notifyEvidenceInspectorSelection(
      [makeNode({node_id: 'n-1', linked_paper_ids: ['paper-a']})],
      'missing-node',
      (nextSelection) => {
        calls.push(nextSelection);
      },
    );

    expect(selection).toBeNull();
    expect(calls).toEqual([]);
  });
});

describe('logical backbone scaffold rendering', () => {
  it('renders editable nodes and weak-link annotations', () => {
    const markup = renderToStaticMarkup(
      <LogicalBackboneTreeScaffold
        nodes={[
          makeNode({
            node_id: 'root-1',
            claim: 'Root argument',
            confidence_label: 'high',
          }),
          makeNode({
            node_id: 'weak-1',
            parent_node_id: 'root-1',
            claim: 'Weak supporting step',
            confidence_label: 'low',
            weak_link_note: 'Missing replication evidence',
            linked_paper_ids: ['paper-a'],
          }),
        ]}
        selectedNodeId="weak-1"
      />,
    );

    expect(markup).toContain('Logical Backbone Tree');
    expect(markup).toContain('Root argument');
    expect(markup).toContain('Weak supporting step');
    expect(markup).toContain('Weak link');
    expect(markup).toContain('Missing replication evidence');
    expect(markup).toContain('data-edit-claim="weak-1"');
    expect(markup).toContain('data-node-selected="true"');
    expect(markup).toContain('data-inspector-target="weak-1"');
  });

  it('renders an empty state when no nodes exist', () => {
    const markup = renderToStaticMarkup(
      <LogicalBackboneTreeScaffold nodes={[]} selectedNodeId={null} />,
    );

    expect(markup).toContain('No logical backbone nodes yet');
  });
});
