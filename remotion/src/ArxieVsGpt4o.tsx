import {loadFont as loadHeadline} from '@remotion/google-fonts/BebasNeue';
import {loadFont as loadBody} from '@remotion/google-fonts/Barlow';
import React from 'react';
import {
  AbsoluteFill,
  Easing,
  Sequence,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {CitationGraphVisualization} from './CitationGraphVisualization';
import {TIMELINE} from './config';
import {
  arxieAnswer,
  arxieCitations,
  citationGraph,
  gpt4oAnswer,
  gpt4oCitations,
  question,
} from './data';

const headlineFont = loadHeadline('normal', {weights: ['400']}).fontFamily;
const bodyFont = loadBody('normal', {weights: ['400', '600', '700']}).fontFamily;

const cardStyle: React.CSSProperties = {
  borderRadius: 26,
  padding: '30px 34px',
  background: 'rgba(13, 18, 27, 0.78)',
  backdropFilter: 'blur(8px)',
  border: '1px solid rgba(255,255,255,0.15)',
  boxShadow: '0 26px 48px rgba(0,0,0,0.28)',
};

const panelTitleStyle: React.CSSProperties = {
  fontFamily: headlineFont,
  fontSize: 56,
  letterSpacing: '0.08em',
  marginBottom: 8,
};

const responseStyle: React.CSSProperties = {
  fontSize: 28,
  lineHeight: 1.42,
  color: '#ebeff5',
  marginTop: 16,
};

const CitationList: React.FC<{
  items: typeof gpt4oCitations;
  statusColor: string;
  label: string;
  frame: number;
}> = ({items, statusColor, label, frame}) => {
  return (
    <div style={{marginTop: 18, display: 'flex', flexDirection: 'column', gap: 10}}>
      {items.map((citation, index) => {
        const itemProgress = spring({
          frame: frame - index * 8,
          fps: 30,
          config: {damping: 200},
        });

        return (
          <div
            key={citation.identifier}
            style={{
              borderLeft: `4px solid ${statusColor}`,
              borderRadius: 10,
              padding: '10px 12px',
              background: 'rgba(255,255,255,0.04)',
              transform: `translateX(${interpolate(itemProgress, [0, 1], [22, 0])}px)`,
              opacity: itemProgress,
            }}
          >
            <div style={{fontSize: 17, color: '#edf1f7'}}>{citation.text}</div>
            <div style={{fontSize: 15, color: '#b8c8da', marginTop: 3}}>{citation.identifier}</div>
            <div style={{fontSize: 14, color: statusColor, marginTop: 5, letterSpacing: '0.03em'}}>
              {label}
            </div>
          </div>
        );
      })}
    </div>
  );
};

export const ArxieVsGpt4o: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();

  const introProgress = spring({
    frame,
    fps,
    config: {damping: 200},
  });

  const splitReveal = interpolate(
    frame,
    [TIMELINE.comparisonStart, TIMELINE.comparisonStart + 22],
    [1.06, 1],
    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.out(Easing.cubic)},
  );

  const graphProgress = interpolate(
    frame,
    [TIMELINE.graphRevealStart, TIMELINE.graphRevealStart + 35],
    [0, 1],
    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
  );

  const outroOpacity = interpolate(
    frame,
    [TIMELINE.outroStart, TIMELINE.outroStart + 18],
    [0, 1],
    {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
  );

  return (
    <AbsoluteFill
      style={{
        fontFamily: bodyFont,
        background:
          'radial-gradient(circle at 15% 20%, #2f3f69 0%, rgba(47,63,105,0.15) 36%), radial-gradient(circle at 80% 75%, #25594f 0%, rgba(37,89,79,0.18) 42%), linear-gradient(135deg, #0f1624 0%, #101d2f 55%, #121827 100%)',
        color: '#f4f8ff',
      }}
    >
      <Sequence from={TIMELINE.introStart} durationInFrames={TIMELINE.comparisonStart} premountFor={12}>
        <AbsoluteFill
          style={{
            justifyContent: 'center',
            alignItems: 'center',
            transform: `scale(${interpolate(introProgress, [0, 1], [0.96, 1])})`,
            opacity: interpolate(
              frame,
              [TIMELINE.introStart, TIMELINE.comparisonStart - 6],
              [0, 1],
              {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
            ),
          }}
        >
          <div style={{...cardStyle, width: 1380, textAlign: 'center', padding: '42px 54px'}}>
            <div style={{fontFamily: headlineFont, fontSize: 96, letterSpacing: '0.12em'}}>CITATION RELIABILITY</div>
            <div style={{fontSize: 42, color: '#b8d9ff', marginTop: 10}}>GPT-4o vs Arxie</div>
            <div style={{fontSize: 28, lineHeight: 1.35, marginTop: 18, color: '#dce7f7'}}>{question}</div>
          </div>
        </AbsoluteFill>
      </Sequence>

      <Sequence
        from={TIMELINE.comparisonStart}
        durationInFrames={TIMELINE.outroStart - TIMELINE.comparisonStart + 1}
        premountFor={15}
      >
        <AbsoluteFill style={{padding: '40px 48px'}}>
          <div style={{display: 'flex', gap: 28, transform: `scale(${splitReveal})`}}>
            <div style={{...cardStyle, width: 900, minHeight: 758}}>
              <div style={{...panelTitleStyle, color: '#f26f82'}}>GPT-4o</div>
              <div style={{fontSize: 20, color: '#f8a8b4', letterSpacing: '0.06em'}}>UNVERIFIED CITATION OUTPUT</div>
              <div style={responseStyle}>{gpt4oAnswer}</div>
              <CitationList
                items={gpt4oCitations}
                statusColor="#ff8aa0"
                label="NOT RESOLVABLE"
                frame={frame - TIMELINE.comparisonStart}
              />
            </div>

            <div style={{...cardStyle, width: 900, minHeight: 758}}>
              <div style={{...panelTitleStyle, color: '#79f0d1'}}>Arxie</div>
              <div style={{fontSize: 20, color: '#9de8d7', letterSpacing: '0.06em'}}>VERIFIED CITATIONS + GRAPH</div>
              <div style={responseStyle}>{arxieAnswer}</div>
              <CitationList
                items={arxieCitations}
                statusColor="#77ebcb"
                label="VERIFIED SOURCE"
                frame={frame - TIMELINE.comparisonStart + 10}
              />
              <div style={{opacity: graphProgress, transform: `translateY(${(1 - graphProgress) * 20}px)`}}>
                <div
                  style={{
                    ...cardStyle,
                    marginTop: 22,
                    height: 240,
                    position: 'relative',
                    overflow: 'hidden',
                    background:
                      'linear-gradient(135deg, rgba(10, 34, 34, 0.9), rgba(5, 17, 37, 0.94))',
                  }}
                >
                  <CitationGraphVisualization
                    data={citationGraph}
                    width={760}
                    height={240}
                    progress={graphProgress}
                    title="Citation Influence Graph"
                  />
                </div>
              </div>
            </div>
          </div>
        </AbsoluteFill>
      </Sequence>

      <Sequence
        from={TIMELINE.outroStart}
        durationInFrames={90}
        premountFor={10}
      >
        <AbsoluteFill style={{justifyContent: 'center', alignItems: 'center', opacity: outroOpacity}}>
          <div style={{...cardStyle, width: 1500, textAlign: 'center', padding: '34px 48px'}}>
            <div style={{fontFamily: headlineFont, fontSize: 78, letterSpacing: '0.08em', color: '#eaf8ff'}}>
              VERIFIED PROVENANCE WINS
            </div>
            <div style={{fontSize: 34, marginTop: 8, color: '#b9e8dc'}}>
              Arxie links claims to resolvable evidence and shows the citation lineage.
            </div>
          </div>
        </AbsoluteFill>
      </Sequence>
    </AbsoluteFill>
  );
};
