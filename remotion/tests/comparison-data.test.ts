import {describe, expect, it} from 'vitest';
import {arxieCitations, citationGraph, gpt4oCitations} from '../src/data';

describe('citation profiles', () => {
  it('flags GPT-4o citations as hallucinated', () => {
    expect(gpt4oCitations.length).toBeGreaterThan(0);
    expect(gpt4oCitations.every((citation) => citation.hallucinated)).toBe(true);
  });

  it('requires all Arxie citations to be verified with stable ids', () => {
    expect(arxieCitations.length).toBeGreaterThan(0);
    expect(
      arxieCitations.every(
        (citation) => citation.verified && Boolean(citation.identifier.trim()),
      ),
    ).toBe(true);
  });

  it('links graph edges to existing graph nodes', () => {
    const nodeIds = new Set(citationGraph.nodes.map((node) => node.id));

    expect(citationGraph.edges.length).toBeGreaterThan(0);
    expect(citationGraph.edges.every((edge) => nodeIds.has(edge.from))).toBe(true);
    expect(citationGraph.edges.every((edge) => nodeIds.has(edge.to))).toBe(true);
  });
});
