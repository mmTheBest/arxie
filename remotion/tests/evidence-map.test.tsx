import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {describe, expect, it} from 'vitest';

import {
  EvidenceMapScaffold,
  buildEvidenceMapBuckets,
  togglePinnedPaperInMemory,
  type EvidenceMapItem,
} from '../src/proposal-shell/EvidenceMap';

function makeItem(partial: Partial<EvidenceMapItem> & Pick<EvidenceMapItem, 'paper_id'>): EvidenceMapItem {
  return {
    paper_id: partial.paper_id,
    title: partial.title ?? partial.paper_id,
    bucket: partial.bucket ?? 'adjacent',
    relevance_score: partial.relevance_score ?? 0.5,
    provenance_link: partial.provenance_link ?? null,
  };
}

describe('evidence map bucket model', () => {
  it('normalizes support/contradict labels into support/contradict/adjacent buckets', () => {
    const items = [
      makeItem({paper_id: 'p-supporting', bucket: 'supporting', relevance_score: 0.72}),
      makeItem({paper_id: 'p-support', bucket: 'support', relevance_score: 0.91}),
      makeItem({paper_id: 'p-contradicting', bucket: 'contradicting', relevance_score: 0.64}),
      makeItem({paper_id: 'p-contradict', bucket: 'contradict', relevance_score: 0.38}),
      makeItem({paper_id: 'p-unknown', bucket: 'unclassified', relevance_score: 0.2}),
    ];

    const buckets = buildEvidenceMapBuckets(items, ['p-supporting']);

    expect(buckets.support.map((item) => item.paper_id)).toEqual(['p-supporting', 'p-support']);
    expect(buckets.contradict.map((item) => item.paper_id)).toEqual([
      'p-contradicting',
      'p-contradict',
    ]);
    expect(buckets.adjacent.map((item) => item.paper_id)).toEqual(['p-unknown']);
  });
});

describe('evidence map scaffold rendering', () => {
  it('renders support/contradict/adjacent sections with provenance links and pin states', () => {
    const markup = renderToStaticMarkup(
      <EvidenceMapScaffold
        items={[
          makeItem({
            paper_id: 'paper-1',
            title: 'Transformer support study',
            bucket: 'support',
            provenance_link: 'https://doi.org/10.1000/xyz123',
          }),
          makeItem({
            paper_id: 'paper-2',
            title: 'Contradictory replication',
            bucket: 'contradicting',
            provenance_link: 'https://example.org/replication',
          }),
          makeItem({
            paper_id: 'paper-3',
            title: 'Adjacent systems review',
            bucket: 'adjacent',
          }),
        ]}
        pinnedPaperIds={['paper-1']}
      />, 
    );

    expect(markup).toContain('Support');
    expect(markup).toContain('Contradict');
    expect(markup).toContain('Adjacent');

    expect(markup).toContain('Transformer support study');
    expect(markup).toContain('Contradictory replication');
    expect(markup).toContain('Adjacent systems review');

    expect(markup).toContain('href="https://doi.org/10.1000/xyz123"');
    expect(markup).toContain('href="https://example.org/replication"');
    expect(markup).toContain('No provenance link');

    expect(markup).toContain('Unpin reference');
    expect(markup).toContain('Pin reference');
    expect(markup).toContain('data-pin-toggle="paper-1"');
  });

  it('renders empty placeholders for each bucket when no evidence exists', () => {
    const markup = renderToStaticMarkup(<EvidenceMapScaffold items={[]} pinnedPaperIds={[]} />);

    expect(markup).toContain('No support papers yet');
    expect(markup).toContain('No contradict papers yet');
    expect(markup).toContain('No adjacent papers yet');
  });
});

describe('pinned paper in-memory toggle helper', () => {
  it('adds unpinned ids and removes pinned ids deterministically', () => {
    expect(togglePinnedPaperInMemory([], 'paper-1')).toEqual(['paper-1']);
    expect(togglePinnedPaperInMemory(['paper-1'], 'paper-1')).toEqual([]);
    expect(togglePinnedPaperInMemory(['paper-1'], 'paper-2')).toEqual(['paper-1', 'paper-2']);
  });

  it('ignores empty ids and de-duplicates incoming pinned values', () => {
    const next = togglePinnedPaperInMemory(['paper-1', 'paper-1', '  ', 'paper-2'], '  ');
    expect(next).toEqual(['paper-1', 'paper-2']);
  });
});
