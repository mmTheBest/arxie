import React, {useMemo} from 'react';
import {forceCenter, forceLink, forceManyBody, forceSimulation} from 'd3-force';

export type CitationGraphNode = {
  id: string;
  label: string;
  year: number;
};

export type CitationGraphEdge = {
  from: string;
  to: string;
};

export type CitationGraphData = {
  nodes: CitationGraphNode[];
  edges: CitationGraphEdge[];
};

type PositionedNode = CitationGraphNode & {
  x: number;
  y: number;
};

type PositionedEdge = CitationGraphEdge & {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export type CitationGraphLayout = {
  nodes: PositionedNode[];
  edges: PositionedEdge[];
};

export type MermaidDirection = 'LR' | 'RL' | 'TB' | 'BT' | 'TD';

type SimulationNode = {
  id: string;
  x: number;
  y: number;
};

type SimulationLink = {
  source: string;
  target: string;
};

const clamp = (value: number, min: number, max: number): number => {
  if (value < min) {
    return min;
  }

  if (value > max) {
    return max;
  }

  return value;
};

const normalizedMermaidId = (id: string): string => id.replace(/[^A-Za-z0-9_]/g, '_');

const escapedMermaidLabel = (label: string): string => label.replace(/"/g, '\\"');

export const buildMermaidDefinition = (
  data: CitationGraphData,
  direction: MermaidDirection = 'LR',
): string => {
  const lines: string[] = [`graph ${direction}`];

  data.nodes.forEach((node) => {
    const safeId = normalizedMermaidId(node.id);
    lines.push(`  ${safeId}["${escapedMermaidLabel(node.label)} (${node.year})"]`);
  });

  data.edges.forEach((edge) => {
    const from = normalizedMermaidId(edge.from);
    const to = normalizedMermaidId(edge.to);
    lines.push(`  ${from} --> ${to}`);
  });

  return lines.join('\n');
};

export const computeCitationGraphLayout = (
  data: CitationGraphData,
  options: {width: number; height: number; iterations?: number},
): CitationGraphLayout => {
  const {width, height, iterations = 90} = options;

  const simulationNodes: SimulationNode[] = data.nodes.map((node, index) => ({
    id: node.id,
    x: ((index + 1) / (data.nodes.length + 1)) * width,
    y: (height * 0.3) + (((index + 1) % 3) * (height * 0.2)),
  }));

  const nodeIds = new Set(simulationNodes.map((node) => node.id));
  const simulationLinks: SimulationLink[] = data.edges
    .filter((edge) => nodeIds.has(edge.from) && nodeIds.has(edge.to))
    .map((edge) => ({source: edge.from, target: edge.to}));

  const simulation = forceSimulation(simulationNodes)
    .force('charge', forceManyBody().strength(-220))
    .force('center', forceCenter(width / 2, height / 2))
    .force(
      'link',
      forceLink(simulationLinks)
        .id((node) => node.id)
        .distance(Math.max(width, height) * 0.34)
        .strength(0.65),
    )
    .stop();

  for (let tick = 0; tick < iterations; tick += 1) {
    simulation.tick();
  }

  simulation.stop();

  const positionedNodeById = new Map(
    simulationNodes.map((node) => [
      node.id,
      {
        x: clamp(node.x, 0, width),
        y: clamp(node.y, 0, height),
      },
    ]),
  );

  const positionedNodes: PositionedNode[] = data.nodes
    .map((node) => {
      const position = positionedNodeById.get(node.id);
      if (!position) {
        return null;
      }

      return {
        ...node,
        x: position.x,
        y: position.y,
      };
    })
    .filter((node): node is PositionedNode => Boolean(node));

  const positionedEdges: PositionedEdge[] = data.edges
    .map((edge) => {
      const from = positionedNodeById.get(edge.from);
      const to = positionedNodeById.get(edge.to);
      if (!from || !to) {
        return null;
      }

      return {
        ...edge,
        x1: from.x,
        y1: from.y,
        x2: to.x,
        y2: to.y,
      };
    })
    .filter((edge): edge is PositionedEdge => Boolean(edge));

  return {
    nodes: positionedNodes,
    edges: positionedEdges,
  };
};

export type CitationGraphVisualizationProps = {
  data: CitationGraphData;
  width: number;
  height: number;
  title?: string;
  mode?: 'd3' | 'mermaid';
  progress?: number;
};

const nodeRadius = 16;

export const CitationGraphVisualization: React.FC<CitationGraphVisualizationProps> = ({
  data,
  width,
  height,
  title,
  mode = 'd3',
  progress = 1,
}) => {
  const layout = useMemo(
    () => computeCitationGraphLayout(data, {width, height, iterations: 95}),
    [data, width, height],
  );

  const mermaidDefinition = useMemo(() => buildMermaidDefinition(data, 'LR'), [data]);

  if (mode === 'mermaid') {
    return (
      <div style={{width, height, overflow: 'auto'}}>
        {title ? <div style={{fontSize: 18, marginBottom: 8}}>{title}</div> : null}
        <pre
          data-mermaid-graph="true"
          style={{
            margin: 0,
            whiteSpace: 'pre-wrap',
            fontFamily: 'Menlo, Monaco, monospace',
            fontSize: 13,
            color: '#d4fff3',
          }}
        >
          {mermaidDefinition}
        </pre>
      </div>
    );
  }

  const visibleEdges = Math.max(0, Math.min(1, progress));

  return (
    <div style={{width, height, position: 'relative', overflow: 'hidden'}}>
      {title ? <div style={{fontSize: 22, color: '#9de8d7', marginBottom: 12}}>{title}</div> : null}
      <svg width={width} height={height - 34} viewBox={`0 0 ${width} ${height - 34}`}>
        <defs>
          <marker
            id="citation-arrowhead"
            markerWidth="8"
            markerHeight="6"
            refX="7"
            refY="3"
            orient="auto"
            markerUnits="strokeWidth"
          >
            <path d="M0,0 L8,3 L0,6 Z" fill="#6de9ce" />
          </marker>
        </defs>

        {layout.edges.map((edge, index) => {
          const edgeProgress = clamp((visibleEdges * layout.edges.length) - index, 0, 1);

          return (
            <line
              key={`${edge.from}-${edge.to}`}
              x1={edge.x1}
              y1={edge.y1}
              x2={edge.x2}
              y2={edge.y2}
              stroke="#6de9ce"
              strokeWidth={3}
              markerEnd="url(#citation-arrowhead)"
              opacity={edgeProgress}
            />
          );
        })}

        {layout.nodes.map((node, index) => {
          const nodeProgress = clamp((visibleEdges * layout.nodes.length) - index, 0, 1);

          return (
            <g key={node.id} opacity={nodeProgress}>
              <circle cx={node.x} cy={node.y} r={nodeRadius} fill="#122f3f" stroke="#90f4dc" strokeWidth={2.5} />
              <text x={node.x} y={node.y + 4} textAnchor="middle" fill="#d4fff3" fontSize={10}>
                {node.year}
              </text>
              <text x={node.x + 24} y={node.y + 4} fill="#eff8fb" fontSize={13}>
                {node.label}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
};
