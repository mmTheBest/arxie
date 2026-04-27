import React from 'react';

export type EvidenceBucketId = 'support' | 'contradict' | 'adjacent';

export interface EvidenceMapItem {
  paper_id: string;
  title: string;
  bucket: string;
  relevance_score: number;
  provenance_link: string | null;
}

export interface EvidenceMapCard extends EvidenceMapItem {
  bucket: EvidenceBucketId;
  pinned: boolean;
}

export interface EvidenceMapBuckets {
  support: EvidenceMapCard[];
  contradict: EvidenceMapCard[];
  adjacent: EvidenceMapCard[];
}

const BUCKET_LABELS: Record<EvidenceBucketId, string> = {
  support: 'Support',
  contradict: 'Contradict',
  adjacent: 'Adjacent',
};

function normalizePaperId(paperId: string): string {
  return String(paperId || '').trim();
}

function normalizeBucket(bucket: string): EvidenceBucketId {
  const normalized = String(bucket || '').trim().toLowerCase();

  if (normalized === 'support' || normalized === 'supports' || normalized === 'supporting') {
    return 'support';
  }

  if (
    normalized === 'contradict' ||
    normalized === 'contradicts' ||
    normalized === 'contradicting' ||
    normalized === 'opposing' ||
    normalized === 'oppose'
  ) {
    return 'contradict';
  }

  return 'adjacent';
}

function normalizePinnedPaperIds(pinnedPaperIds: string[]): string[] {
  const deduped: string[] = [];
  const seen = new Set<string>();

  for (const paperId of pinnedPaperIds) {
    const normalized = normalizePaperId(paperId);
    if (!normalized || seen.has(normalized)) {
      continue;
    }

    deduped.push(normalized);
    seen.add(normalized);
  }

  return deduped;
}

function compareCards(a: EvidenceMapCard, b: EvidenceMapCard): number {
  const pinnedDiff = Number(b.pinned) - Number(a.pinned);
  if (pinnedDiff !== 0) {
    return pinnedDiff;
  }

  const relevanceDiff = b.relevance_score - a.relevance_score;
  if (relevanceDiff !== 0) {
    return relevanceDiff;
  }

  const byTitle = a.title.localeCompare(b.title);
  if (byTitle !== 0) {
    return byTitle;
  }

  return a.paper_id.localeCompare(b.paper_id);
}

export function buildEvidenceMapBuckets(
  items: EvidenceMapItem[],
  pinnedPaperIds: string[] = [],
): EvidenceMapBuckets {
  const pinnedSet = new Set(normalizePinnedPaperIds(pinnedPaperIds));

  const buckets: EvidenceMapBuckets = {
    support: [],
    contradict: [],
    adjacent: [],
  };

  for (const item of items) {
    const paperId = normalizePaperId(item.paper_id);
    if (!paperId) {
      continue;
    }

    const bucket = normalizeBucket(item.bucket);

    const card: EvidenceMapCard = {
      ...item,
      paper_id: paperId,
      title: String(item.title || '').trim() || paperId,
      bucket,
      relevance_score: Number.isFinite(item.relevance_score) ? item.relevance_score : 0,
      provenance_link: item.provenance_link,
      pinned: pinnedSet.has(paperId),
    };

    buckets[bucket].push(card);
  }

  buckets.support.sort(compareCards);
  buckets.contradict.sort(compareCards);
  buckets.adjacent.sort(compareCards);

  return buckets;
}

export function togglePinnedPaperInMemory(
  currentPinnedPaperIds: string[],
  targetPaperId: string,
): string[] {
  const normalizedCurrent = normalizePinnedPaperIds(currentPinnedPaperIds);
  const normalizedTarget = normalizePaperId(targetPaperId);

  if (!normalizedTarget) {
    return normalizedCurrent;
  }

  if (normalizedCurrent.includes(normalizedTarget)) {
    return normalizedCurrent.filter((paperId) => paperId !== normalizedTarget);
  }

  return [...normalizedCurrent, normalizedTarget];
}

function EvidenceCard({
  card,
  pinnedPaperIds,
  disabled,
  onTogglePin,
}: {
  card: EvidenceMapCard;
  pinnedPaperIds: string[];
  disabled: boolean;
  onTogglePin?: (paperId: string, nextPinnedPaperIds: string[]) => void;
}): JSX.Element {
  const isPinned = pinnedPaperIds.includes(card.paper_id);

  return (
    <article
      data-paper-id={card.paper_id}
      data-bucket={card.bucket}
      style={{
        border: '1px solid #d5d5d5',
        borderRadius: 10,
        padding: 10,
        background: '#ffffff',
        display: 'grid',
        gap: 8,
      }}
    >
      <header style={{display: 'grid', gap: 2}}>
        <strong style={{fontSize: 13}}>{card.title}</strong>
        <span style={{fontSize: 12, color: '#5a6370'}}>paper_id: {card.paper_id}</span>
      </header>

      <div style={{display: 'grid', gap: 2, fontSize: 12, color: '#5a6370'}}>
        <span>relevance: {card.relevance_score.toFixed(2)}</span>
        {card.provenance_link ? (
          <a href={card.provenance_link} target="_blank" rel="noreferrer" style={{color: '#2563eb'}}>
            View provenance
          </a>
        ) : (
          <span>No provenance link</span>
        )}
      </div>

      <div>
        <button
          type="button"
          data-pin-toggle={card.paper_id}
          aria-pressed={isPinned}
          disabled={disabled}
          onClick={() => {
            const next = togglePinnedPaperInMemory(pinnedPaperIds, card.paper_id);
            onTogglePin?.(card.paper_id, next);
          }}
          style={{
            borderRadius: 999,
            border: '1px solid #d5d5d5',
            padding: '6px 10px',
            fontSize: 12,
            background: isPinned ? '#f4fbf7' : '#ffffff',
          }}
        >
          {isPinned ? 'Unpin reference' : 'Pin reference'}
        </button>
      </div>
    </article>
  );
}

function EvidenceBucketSection({
  bucket,
  cards,
  pinnedPaperIds,
  disabled,
  onTogglePin,
}: {
  bucket: EvidenceBucketId;
  cards: EvidenceMapCard[];
  pinnedPaperIds: string[];
  disabled: boolean;
  onTogglePin?: (paperId: string, nextPinnedPaperIds: string[]) => void;
}): JSX.Element {
  return (
    <section
      data-evidence-bucket={bucket}
      style={{
        display: 'grid',
        gap: 8,
        border: '1px solid #d5d5d5',
        borderRadius: 10,
        padding: 10,
        background: '#f8fafc',
      }}
    >
      <header style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
        <strong style={{fontSize: 12}}>{BUCKET_LABELS[bucket]}</strong>
        <span style={{fontSize: 12, color: '#5a6370'}}>{cards.length}</span>
      </header>

      {cards.length === 0 ? (
        <span style={{fontSize: 12, color: '#5a6370'}}>No {bucket} papers yet</span>
      ) : (
        <div style={{display: 'grid', gap: 8}}>
          {cards.map((card) => (
            <EvidenceCard
              key={card.paper_id}
              card={card}
              pinnedPaperIds={pinnedPaperIds}
              disabled={disabled}
              onTogglePin={onTogglePin}
            />
          ))}
        </div>
      )}
    </section>
  );
}

export function EvidenceMapScaffold({
  items,
  pinnedPaperIds,
  disabled = false,
  onTogglePin,
}: {
  items: EvidenceMapItem[];
  pinnedPaperIds: string[];
  disabled?: boolean;
  onTogglePin?: (paperId: string, nextPinnedPaperIds: string[]) => void;
}): JSX.Element {
  const normalizedPinnedPaperIds = normalizePinnedPaperIds(pinnedPaperIds);
  const buckets = buildEvidenceMapBuckets(items, normalizedPinnedPaperIds);

  return (
    <section aria-label="Evidence map scaffold" style={{display: 'grid', gap: 10}}>
      <header>
        <strong style={{fontSize: 14}}>Evidence Map</strong>
      </header>

      <div
        style={{
          display: 'grid',
          gap: 10,
          gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
        }}
      >
        <EvidenceBucketSection
          bucket="support"
          cards={buckets.support}
          pinnedPaperIds={normalizedPinnedPaperIds}
          disabled={disabled}
          onTogglePin={onTogglePin}
        />
        <EvidenceBucketSection
          bucket="contradict"
          cards={buckets.contradict}
          pinnedPaperIds={normalizedPinnedPaperIds}
          disabled={disabled}
          onTogglePin={onTogglePin}
        />
        <EvidenceBucketSection
          bucket="adjacent"
          cards={buckets.adjacent}
          pinnedPaperIds={normalizedPinnedPaperIds}
          disabled={disabled}
          onTogglePin={onTogglePin}
        />
      </div>
    </section>
  );
}
