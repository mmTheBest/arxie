import React from 'react';
import {Composition, Folder} from 'remotion';
import {ArxieVsGpt4o} from './ArxieVsGpt4o';
import {COMPOSITION_META} from './config';

export const RemotionRoot: React.FC = () => {
  return (
    <Folder name="Arxie-Comparisons">
      <Composition
        id={COMPOSITION_META.id}
        component={ArxieVsGpt4o}
        durationInFrames={COMPOSITION_META.durationInFrames}
        fps={COMPOSITION_META.fps}
        width={COMPOSITION_META.width}
        height={COMPOSITION_META.height}
      />
    </Folder>
  );
};
