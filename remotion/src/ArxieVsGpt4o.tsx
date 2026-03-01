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

const graphNodeRadius = 16;

const CitationGraph: React.FC<{progress: number}> = ({progress}) => {
  const width = 760;
  const height = 240;

  const nodeMap = new Map(citationGraph.nodes.map((node) => [node.id, node]));

  return (
    <div
      style={{
        ...cardStyle,
        marginTop: 22,
        height,
        position: 'relative',
        overflow: 'hidden',
        background:
          'linear-gradient(135deg, rgba(10, 34, 34, 0.9), rgba(5, 17, 37, 0.94))',
      }}
    >
      <div style={{fontSize: 22, color: '#9de8d7', marginBottom: 12}}>Citation Influence Graph</div>
      <svg width={width} height={height - 34} viewBox={`0 0 ${width} ${height - 34}`}>
        {citationGraph.edges.map((edge, index) => {
          const from = nodeMap.get(edge.from);
          const to = nodeMap.get(edge.to);
          if (!from || !to) {
            return null;
          }

          const x1 = (from.x / 100) * width;
          const y1 = (from.y / 100) * (height - 34);
          const x2 = (to.x / 100) * width;
          const y2 = (to.y / 100) * (height - 34);

          const strokeProgress = interpolate(
            progress,
            [index * 0.16, index * 0.16 + 0.22],
            [0, 1],
            {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
          );

          return (
            <line
              key={`${edge.from}-${edge.to}`}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke="#6de9ce"
              strokeWidth={3}
              strokeDasharray="1"
              strokeDashoffset={1 - strokeProgress}
              opacity={strokeProgress}
            />
          );
        })}
        {citationGraph.nodes.map((node, index) => {
          const nodeProgress = interpolate(
            progress,
            [index * 0.14, index * 0.14 + 0.2],
            [0, 1],
            {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'},
          );

          const x = (node.x / 100) * width;
          const y = (node.y / 100) * (height - 34);

          return (
            <g key={node.id} opacity={nodeProgress}>
              <circle cx={x} cy={y} r={graphNodeRadius} fill="#122f3f" stroke="#90f4dc" strokeWidth={2.5} />
              <text
                x={x}
                y={y + 5}
                textAnchor="middle"
                fill="#d4fff3"
                fontSize={10}
                fontFamily={bodyFont}
              >
                {node.year}
              </text>
              <text
                x={x + 22}
                y={y + 4}
                fill="#eff8fb"
                fontSize={13}
                fontFamily={bodyFont}
              >
                {node.label}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
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
                <CitationGraph progress={graphProgress} />
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
