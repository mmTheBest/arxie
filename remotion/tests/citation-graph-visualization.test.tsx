import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {describe, expect, it} from 'vitest';
import {
  CitationGraphVisualization,
  buildMermaidDefinition,
  computeCitationGraphLayout,
  type CitationGraphData,
} from '../src/CitationGraphVisualization';

const sampleGraph: CitationGraphData = {
  nodes: [
    {id: 'p1', label: 'Paper A', year: 2020},
    {id: 'p2', label: 'Paper B', year: 2022},
    {id: 'p3', label: 'Paper C', year: 2024},
  ],
  edges: [
    {from: 'p1', to: 'p2'},
    {from: 'p2', to: 'p3'},
  ],
};

describe('citation graph visualization', () => {
  it('builds a Mermaid directed graph definition from paper relationships', () => {
    const definition = buildMermaidDefinition(sampleGraph, 'TD');

    expect(definition).toContain('graph TD');
    expect(definition).toContain('p1["Paper A (2020)"]');
    expect(definition).toContain('p2["Paper B (2022)"]');
    expect(definition).toContain('p1 --> p2');
    expect(definition).toContain('p2 --> p3');
  });

  it('computes bounded d3 coordinates for each paper node', () => {
    const layout = computeCitationGraphLayout(sampleGraph, {
      width: 640,
      height: 320,
      iterations: 40,
    });

    expect(layout.nodes).toHaveLength(sampleGraph.nodes.length);
    expect(layout.edges).toHaveLength(sampleGraph.edges.length);

    for (const node of layout.nodes) {
      expect(node.x).toBeGreaterThanOrEqual(0);
      expect(node.x).toBeLessThanOrEqual(640);
      expect(node.y).toBeGreaterThanOrEqual(0);
      expect(node.y).toBeLessThanOrEqual(320);
    }
  });

  it('renders an SVG directed graph using d3 layout output', () => {
    const markup = renderToStaticMarkup(
      <CitationGraphVisualization
        data={sampleGraph}
        width={640}
        height={320}
        title="Citation Influence"
      />,
    );

    expect(markup).toContain('<svg');
    expect(markup).toContain('marker-end="url(#citation-arrowhead)"');
    expect(markup).toContain('Paper A');
    expect(markup).toContain('Paper B');
    expect(markup).toContain('Paper C');
  });

  it('supports Mermaid view mode for downstream rendering', () => {
    const markup = renderToStaticMarkup(
      <CitationGraphVisualization
        data={sampleGraph}
        width={640}
        height={320}
        mode="mermaid"
      />,
    );

    expect(markup).toContain('data-mermaid-graph="true"');
    expect(markup).toContain('graph LR');
    expect(markup).toContain('p1 --&gt; p2');
  });
});
