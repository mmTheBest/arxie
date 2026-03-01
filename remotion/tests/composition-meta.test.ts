import {describe, expect, it} from 'vitest';
import {COMPOSITION_META, TIMELINE} from '../src/config';

describe('demo composition metadata', () => {
  it('uses a 1080p landscape composition for side-by-side panels', () => {
    expect(COMPOSITION_META.width).toBe(1920);
    expect(COMPOSITION_META.height).toBe(1080);
  });

  it('keeps timeline checkpoints ordered within video duration', () => {
    expect(TIMELINE.introStart).toBe(0);
    expect(TIMELINE.comparisonStart).toBeGreaterThan(TIMELINE.introStart);
    expect(TIMELINE.graphRevealStart).toBeGreaterThan(TIMELINE.comparisonStart);
    expect(TIMELINE.outroStart).toBeGreaterThan(TIMELINE.graphRevealStart);
    expect(TIMELINE.outroStart).toBeLessThan(COMPOSITION_META.durationInFrames);
  });
});
