export type CitationItem = {
  text: string;
  identifier: string;
  verified: boolean;
  hallucinated: boolean;
};

export type GraphNode = {
  id: string;
  label: string;
  year: number;
  x: number;
  y: number;
};

export type GraphEdge = {
  from: string;
  to: string;
};

export const question =
  'What is the strongest evidence that retrieval-augmented generation improves factual consistency in biomedical QA?';

export const gpt4oAnswer =
  'RAG improves biomedical QA by reducing unsupported claims and increasing citation density, but this response includes fabricated references that cannot be resolved to real papers.';

export const arxieAnswer =
  'Arxie cross-checks each claim against retriever output, only emits resolvable references, and surfaces influence chains so the user can inspect provenance beyond flat citation lists.';

export const gpt4oCitations: CitationItem[] = [
  {
    text: 'Li et al. (2024). Biomedical Hallucination Suppression via Neural Grounding.',
    identifier: 'DOI:10.5555/biomed.404404',
    verified: false,
    hallucinated: true,
  },
  {
    text: 'Patel and Gomez (2023). Universal Retrieval Certainty for Clinical LLMs.',
    identifier: 'arXiv:2310.99999',
    verified: false,
    hallucinated: true,
  },
  {
    text: 'Wu et al. (2025). Cited Truthfulness Benchmarks for GPT-4o.',
    identifier: 'DOI:10.4242/imaginary.2025.1',
    verified: false,
    hallucinated: true,
  },
];

export const arxieCitations: CitationItem[] = [
  {
    text: 'Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP.',
    identifier: 'arXiv:2005.11401',
    verified: true,
    hallucinated: false,
  },
  {
    text: 'Luu et al. (2024). Citation-Enhanced Generation for Evidence-Based QA.',
    identifier: 'DOI:10.48550/arXiv.2401.10245',
    verified: true,
    hallucinated: false,
  },
  {
    text: 'Kang et al. (2023). Improving Factuality with Retrieval and Verification Loops.',
    identifier: 'DOI:10.48550/arXiv.2311.06789',
    verified: true,
    hallucinated: false,
  },
];

export const citationGraph: {nodes: GraphNode[]; edges: GraphEdge[]} = {
  nodes: [
    {id: 'n1', label: 'RAG (2020)', year: 2020, x: 12, y: 72},
    {id: 'n2', label: 'BioMed-RAG (2022)', year: 2022, x: 35, y: 44},
    {id: 'n3', label: 'FactCheck Loop (2023)', year: 2023, x: 58, y: 62},
    {id: 'n4', label: 'Citation QA (2024)', year: 2024, x: 82, y: 34},
    {id: 'n5', label: 'Arxie Synthesis (2026)', year: 2026, x: 82, y: 78},
  ],
  edges: [
    {from: 'n1', to: 'n2'},
    {from: 'n2', to: 'n3'},
    {from: 'n3', to: 'n4'},
    {from: 'n3', to: 'n5'},
  ],
};
