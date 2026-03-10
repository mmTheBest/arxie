# Arxie v0.2 Pre-PRD (Discussion Draft)

## Purpose
Define the next product step after Arxie v0.1.0: evolve from a strong literature assistant into a collaborative **research proposal co-creation workspace**.

This is a pre-PRD for discussion, not a final requirements lock.

---

## Product North Star
Arxie should help a solo researcher turn an early idea into a structured, evidence-grounded research proposal through iterative dialogue and visual reasoning tools.

**Core principle:** Arxie should **not** generate the whole proposal from one simple prompt by default.
It should co-develop the proposal with the researcher step by step.

---

## Target User (v0.2)
- **Primary user:** solo researcher (e.g., PhD student)
- **Collaboration:** out of scope for this version (single-user only)
- **Domain scope:** domain-agnostic

---

## Current State (v0.1.0)
Arxie already provides:
- paper search + citation-grounded QA
- deep search / multi-hop paper analysis
- literature review generation
- citation influence tracing
- confidence annotations
- conversational mode

Main gap: outputs are strong, but workflow is still command/chat-centric and lacks a structured visual proposal-building environment.

---

## v0.2 Product Objective
Deliver a dashboard-based workflow that supports iterative proposal development with multiple linked artifacts:
- mindmaps
- logical backbone trees
- evidence maps (support/contradict)
- method/analysis plans
- expected-outcome comparisons against prior work

All artifacts should stay synchronized with the evolving conversation and user edits.

---

## User Workflow (Iterative)

### Stage 1 — Problem framing
- user states research intent
- Arxie asks clarifying questions (scope, population, constraints)
- output: problem statement + boundaries

### Stage 2 — Evidence landscape
- Arxie maps existing work and contradictions
- output: evidence graph + "known/unknown/contested" summary

### Stage 3 — Hypothesis development
- Arxie proposes candidate hypotheses and assumptions
- user revises hypotheses iteratively
- output: hypothesis tree with testability/falsifiability notes

### Stage 4 — Study blueprint
- data plan, analysis plan, controls/confounders
- output: method pipeline diagram + analysis checklist

### Stage 5 — Expected outcomes
- expected result ranges and interpretation rules
- comparison vs related prior work
- output: outcome matrix + decision criteria

### Stage 6 — Proposal draft assembly
- build coherent proposal sections from prior stages
- output: editable structured proposal draft

---

## UX / Dashboard Requirements

### Information architecture (v0.2)
1. **Conversation panel** (driver)
2. **Proposal canvas** (editable structured sections)
3. **Evidence panel** (papers, citation links, contradiction signals)
4. **Visual tabs**
   - mindmap
   - logical backbone tree
   - evidence map
   - method pipeline
   - outcome comparison matrix

### Behavioral requirements
- no single-shot full artifact generation by default
- stage-gated progression with explicit user confirmation
- editable nodes in any artifact
- cross-artifact propagation (editing hypothesis triggers method/outcome refresh flags)
- provenance visibility (which claims are evidence-backed vs speculative)

---

## Functional Requirements (Draft)

### FR-1 Iterative dialogue engine
Arxie asks targeted follow-ups and moves proposal state forward incrementally.

### FR-2 Proposal state model
Persistent state object stores:
- question framing
- evidence summaries
- hypothesis tree
- method plan
- expected outcomes
- unresolved assumptions

### FR-3 Multi-artifact generation
Generate and update visual artifacts from current proposal state.

### FR-4 Synchronized editing
User edits in one artifact update related artifacts and the draft.

### FR-5 Evidence traceability
Every evidence-backed claim in proposal artifacts links to source papers.

### FR-6 Draft export
Export structured proposal as markdown/PDF (format polish later if needed).

---

## Non-Goals (v0.2)
- team collaboration / multi-user editing
- grant-specific formatting packs by institution
- autonomous experiment execution
- end-to-end paper writing assistance beyond proposal stage

---

## Risks & Open Questions
1. **Cognitive overload risk:** too many visual artifacts at once
   - mitigation: guided stage mode + progressive disclosure
2. **Consistency risk:** artifact drift when edits happen in multiple tabs
   - mitigation: shared state + dependency graph + refresh markers
3. **Domain-agnostic depth tradeoff:** broad support but shallow templates
   - mitigation: generic baseline templates + domain packs later
4. **Open question:** default artifact order and tab priority
5. **Open question:** how much auto-refresh vs user-controlled refresh

---

## Acceptance Criteria for v0.2 (Draft)
- User can complete all 6 stages in one session with editable intermediate artifacts
- User can revise a hypothesis and see linked artifacts marked/updated
- Proposal draft includes: framing, evidence summary, hypothesis, method, expected outcomes, risks
- Evidence-backed statements in draft link to references
- Workflow remains iterative and does not collapse into one-shot generation by default

---

## Suggested Build Milestones
1. State model + stage engine
2. Dashboard shell + conversation/canvas/evidence panes
3. First two visuals (logical tree + evidence map)
4. Remaining visuals + cross-artifact sync
5. Proposal assembly + export
6. Usability polish for solo researcher flow
