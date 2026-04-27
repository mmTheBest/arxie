import React from 'react';

export type LogicalConfidenceLabel = 'low' | 'medium' | 'high';

export interface LogicalBackboneNode {
  node_id: string;
  claim: string;
  parent_node_id: string | null;
  linked_paper_ids: string[];
  confidence_label: LogicalConfidenceLabel;
  weak_link_note: string | null;
}

export interface LogicalBackboneLevel {
  depth: number;
  nodes: LogicalBackboneNode[];
}

export interface EvidenceInspectorSelection {
  nodeId: string;
  paperIds: string[];
}

const CONFIDENCE_WEIGHT: Record<LogicalConfidenceLabel, number> = {
  low: 1,
  medium: 2,
  high: 3,
};

function normalizeNodeId(nodeId: string): string {
  return String(nodeId || '').trim();
}

function normalizeLinkedPaperIds(linkedPaperIds: string[]): string[] {
  const deduped: string[] = [];
  const seen = new Set<string>();

  for (const paperId of linkedPaperIds) {
    const normalized = String(paperId || '').trim();
    if (!normalized || seen.has(normalized)) {
      continue;
    }

    deduped.push(normalized);
    seen.add(normalized);
  }

  return deduped;
}

function normalizeConfidenceLabel(value: string): LogicalConfidenceLabel {
  const normalized = String(value || '').trim().toLowerCase();
  if (normalized === 'high' || normalized === 'medium' || normalized === 'low') {
    return normalized;
  }
  return 'medium';
}

function normalizeNode(node: LogicalBackboneNode): LogicalBackboneNode | null {
  const normalizedNodeId = normalizeNodeId(node.node_id);
  if (!normalizedNodeId) {
    return null;
  }

  const normalizedParentNodeId = normalizeNodeId(node.parent_node_id || '');

  return {
    node_id: normalizedNodeId,
    claim: String(node.claim || '').trim() || normalizedNodeId,
    parent_node_id: normalizedParentNodeId || null,
    linked_paper_ids: normalizeLinkedPaperIds(node.linked_paper_ids),
    confidence_label: normalizeConfidenceLabel(node.confidence_label),
    weak_link_note: node.weak_link_note?.trim() || null,
  };
}

function compareNodesForTree(a: LogicalBackboneNode, b: LogicalBackboneNode): number {
  const byConfidence = CONFIDENCE_WEIGHT[b.confidence_label] - CONFIDENCE_WEIGHT[a.confidence_label];
  if (byConfidence !== 0) {
    return byConfidence;
  }

  const byClaim = a.claim.localeCompare(b.claim);
  if (byClaim !== 0) {
    return byClaim;
  }

  return a.node_id.localeCompare(b.node_id);
}

function cloneNode(node: LogicalBackboneNode): LogicalBackboneNode {
  return {
    ...node,
    linked_paper_ids: [...node.linked_paper_ids],
  };
}

function isWeakLink(node: LogicalBackboneNode): boolean {
  return node.confidence_label === 'low' || Boolean(node.weak_link_note);
}

export function deriveLogicalBackboneLevels(nodes: LogicalBackboneNode[]): LogicalBackboneLevel[] {
  const byNodeId = new Map<string, LogicalBackboneNode>();

  for (const node of nodes) {
    const normalized = normalizeNode(node);
    if (!normalized || byNodeId.has(normalized.node_id)) {
      continue;
    }
    byNodeId.set(normalized.node_id, normalized);
  }

  if (byNodeId.size === 0) {
    return [];
  }

  const childrenByParentNodeId = new Map<string, string[]>();
  const rootNodeIds: string[] = [];

  for (const node of byNodeId.values()) {
    const parentNodeId = node.parent_node_id;
    const hasValidParent = Boolean(parentNodeId && byNodeId.has(parentNodeId));

    if (!hasValidParent) {
      rootNodeIds.push(node.node_id);
      continue;
    }

    const siblings = childrenByParentNodeId.get(parentNodeId as string) ?? [];
    siblings.push(node.node_id);
    childrenByParentNodeId.set(parentNodeId as string, siblings);
  }

  const levels: LogicalBackboneLevel[] = [];
  const visitedNodeIds = new Set<string>();
  let cursorNodeIds = rootNodeIds.sort((a, b) => a.localeCompare(b));
  let depth = 0;

  while (cursorNodeIds.length > 0) {
    const currentNodes = cursorNodeIds
      .map((nodeId) => byNodeId.get(nodeId))
      .filter((node): node is LogicalBackboneNode => Boolean(node))
      .sort(compareNodesForTree);

    if (currentNodes.length === 0) {
      break;
    }

    levels.push({
      depth,
      nodes: currentNodes,
    });

    for (const node of currentNodes) {
      visitedNodeIds.add(node.node_id);
    }

    const nextNodeIds: string[] = [];
    for (const node of currentNodes) {
      const childNodeIds = childrenByParentNodeId.get(node.node_id) ?? [];
      for (const childNodeId of childNodeIds) {
        if (!visitedNodeIds.has(childNodeId)) {
          nextNodeIds.push(childNodeId);
        }
      }
    }

    cursorNodeIds = nextNodeIds.sort((a, b) => a.localeCompare(b));
    depth += 1;
  }

  const remainingNodes = [...byNodeId.values()]
    .filter((node) => !visitedNodeIds.has(node.node_id))
    .sort(compareNodesForTree);

  if (remainingNodes.length > 0) {
    levels.push({
      depth,
      nodes: remainingNodes,
    });
  }

  return levels;
}

export function updateLogicalBackboneNodeClaimInMemory(
  nodes: LogicalBackboneNode[],
  targetNodeId: string,
  nextClaim: string,
): LogicalBackboneNode[] {
  const normalizedTargetNodeId = normalizeNodeId(targetNodeId);
  const normalizedClaim = String(nextClaim || '').trim();

  return nodes.map((node) => {
    const cloned = cloneNode(node);
    if (normalizeNodeId(cloned.node_id) !== normalizedTargetNodeId) {
      return cloned;
    }

    if (!normalizedClaim) {
      return cloned;
    }

    return {
      ...cloned,
      claim: normalizedClaim,
    };
  });
}

export function buildEvidenceInspectorSelection(
  nodes: LogicalBackboneNode[],
  targetNodeId: string,
): EvidenceInspectorSelection | null {
  const normalizedTargetNodeId = normalizeNodeId(targetNodeId);
  if (!normalizedTargetNodeId) {
    return null;
  }

  for (const node of nodes) {
    if (normalizeNodeId(node.node_id) !== normalizedTargetNodeId) {
      continue;
    }

    return {
      nodeId: normalizeNodeId(node.node_id),
      paperIds: normalizeLinkedPaperIds(node.linked_paper_ids),
    };
  }

  return null;
}

export function LogicalBackboneTreeScaffold({
  nodes,
  selectedNodeId,
  disabled = false,
  onNodeClaimChange,
  onNodeSelectForInspector,
}: {
  nodes: LogicalBackboneNode[];
  selectedNodeId: string | null;
  disabled?: boolean;
  onNodeClaimChange?: (
    nodeId: string,
    nextClaim: string,
    nextNodes: LogicalBackboneNode[],
  ) => void;
  onNodeSelectForInspector?: (selection: EvidenceInspectorSelection) => void;
}): JSX.Element {
  const normalizedSelectedNodeId = normalizeNodeId(selectedNodeId || '');
  const levels = deriveLogicalBackboneLevels(nodes);

  if (levels.length === 0) {
    return (
      <section aria-label="Logical backbone tree" style={{fontSize: 13}}>
        No logical backbone nodes yet
      </section>
    );
  }

  return (
    <section aria-label="Logical backbone tree" style={{display: 'grid', gap: 10}}>
      <header>
        <strong style={{fontSize: 14}}>Logical Backbone Tree</strong>
      </header>

      <div style={{display: 'grid', gap: 10}}>
        {levels.map((level) => (
          <section
            key={`logical-depth-${level.depth}`}
            data-logical-depth={String(level.depth)}
            style={{
              display: 'grid',
              gap: 8,
              border: '1px solid #d5d5d5',
              borderRadius: 10,
              padding: 10,
              background: '#f8fafc',
            }}
          >
            <strong style={{fontSize: 12}}>Layer {level.depth + 1}</strong>

            <div style={{display: 'grid', gap: 8}}>
              {level.nodes.map((node) => {
                const selected = node.node_id === normalizedSelectedNodeId;
                const weakLink = isWeakLink(node);

                return (
                  <article
                    key={node.node_id}
                    data-logical-node={node.node_id}
                    data-node-selected={selected ? 'true' : 'false'}
                    style={{
                      border: '1px solid #d5d5d5',
                      borderRadius: 10,
                      padding: 10,
                      background: selected ? '#f4fbf7' : '#ffffff',
                      display: 'grid',
                      gap: 8,
                    }}
                  >
                    <header style={{display: 'flex', justifyContent: 'space-between', gap: 8}}>
                      <span style={{fontSize: 12, color: '#5a6370'}}>node: {node.node_id}</span>
                      <span style={{fontSize: 12, color: '#5a6370'}}>
                        confidence: {node.confidence_label}
                      </span>
                    </header>

                    {weakLink ? (
                      <div style={{display: 'grid', gap: 2}}>
                        <span
                          data-weak-link={node.node_id}
                          style={{fontSize: 12, fontWeight: 600, color: '#9f1239'}}
                        >
                          Weak link
                        </span>
                        {node.weak_link_note ? (
                          <span style={{fontSize: 12, color: '#5a6370'}}>{node.weak_link_note}</span>
                        ) : (
                          <span style={{fontSize: 12, color: '#5a6370'}}>
                            Low confidence requires supporting evidence.
                          </span>
                        )}
                      </div>
                    ) : null}

                    <textarea
                      data-edit-claim={node.node_id}
                      defaultValue={node.claim}
                      disabled={disabled}
                      onChange={(event) => {
                        const nextNodes = updateLogicalBackboneNodeClaimInMemory(
                          nodes,
                          node.node_id,
                          event.target.value,
                        );
                        onNodeClaimChange?.(node.node_id, event.target.value, nextNodes);
                      }}
                      style={{
                        minHeight: 76,
                        borderRadius: 8,
                        border: '1px solid #d5d5d5',
                        padding: '8px 10px',
                        fontSize: 13,
                        resize: 'vertical',
                        fontFamily: 'inherit',
                      }}
                    />

                    <div style={{display: 'flex', justifyContent: 'space-between', gap: 8}}>
                      <button
                        type="button"
                        data-inspector-target={node.node_id}
                        disabled={disabled}
                        onClick={() => {
                          const selection = buildEvidenceInspectorSelection(nodes, node.node_id);
                          if (selection) {
                            onNodeSelectForInspector?.(selection);
                          }
                        }}
                        style={{
                          borderRadius: 999,
                          border: '1px solid #d5d5d5',
                          padding: '6px 10px',
                          fontSize: 12,
                          background: '#ffffff',
                        }}
                      >
                        Inspect evidence ({node.linked_paper_ids.length})
                      </button>
                      <span style={{fontSize: 12, color: '#5a6370'}}>
                        parent: {node.parent_node_id ?? 'root'}
                      </span>
                    </div>
                  </article>
                );
              })}
            </div>
          </section>
        ))}
      </div>
    </section>
  );
}
