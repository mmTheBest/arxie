# Arxie PRD v0.2 — Research Proposal Co-Creation Workspace

## 1) Product Overview

### Product vision
Arxie helps a solo researcher turn a rough idea into a structured, evidence-grounded research proposal through iterative dialogue and visual reasoning workflows.

### Core principle
Arxie should **not** generate a full proposal from one simple prompt by default. It should co-develop the proposal with the user stage by stage.

### Version scope
- Current shipped product: `v0.1.0`
- This PRD defines: `v0.2.0`

---

## 2) Target User & Context

### Primary user
- Solo researcher (e.g., PhD student)

### Usage context
- Early-stage idea exploration
- Proposal refinement before running experiments
- Building defensible logic with supporting and opposing evidence

### In-scope (v0.2)
- Single-user workspace
- Domain-agnostic proposal workflow
- Multi-artifact dashboard
- Branch-based hypothesis and design exploration

### Out-of-scope (v0.2)
- Multi-user collaboration
- Institution-specific grant templates
- Autonomous experiment execution
- Full manuscript writing beyond proposal scope

---

## 3) Product Objectives

1. Convert rough ideas into testable research logic.
2. Make evidence support/contradiction explicit and traceable.
3. Support iterative hypothesis and study design branching.
4. Provide a dashboard workflow where artifacts stay synchronized.
5. Assemble proposal drafts from approved workflow state.

---

## 4) End-to-End Workflow (Stage-Gated)

## Stage 0 — Idea Intake
**Input:** rough free-text idea from user.

**Arxie actions:**
- Parse key components: problem, target, mechanism, expected outcome.
- Detect missing information and ambiguities.

**Outputs:**
- Idea Parse Card (editable)
- Missing Info Checklist

**Exit criteria:** user confirms/refines parsed idea.

---

## Stage 1 — Logic Refinement
**Goal:** build a testable logic backbone.

**Arxie actions:**
- Build logical tree:
  - Problem → Gap → Mechanism → Expected effect → Testable hypothesis
- Flag weak logic nodes and unclear assumptions.

**Outputs:**
- Logical Backbone Tree
- Assumption/weak-link annotations

**Exit criteria:** user approves backbone or edits nodes.

---

## Stage 2 — Evidence Mapping
**Goal:** establish current literature landscape.

**Arxie actions:**
- Retrieve and cluster relevant papers.
- Separate evidence into:
  - Supporting
  - Contradicting
  - Adjacent/related
- Summarize consensus, controversy, and unknowns.

**Outputs:**
- Evidence Map
- Landscape Summary

**Exit criteria:** user pins core references and accepts evidence baseline.

---

## Stage 3 — Hypothesis Reshaping
**Goal:** generate and refine candidate hypotheses after literature grounding.

**Arxie actions:**
- Propose H1/H2/alternatives.
- Explicitly list assumptions and falsification criteria.
- Mark confidence level by evidence grounding.

**Outputs:**
- Hypothesis Branch Tree
- Falsification Criteria Table

**Exit criteria:** user selects primary branch (optional backup branch).

---

## Stage 4 — Data & Feasibility Planning
**Goal:** map feasible data paths for selected branch(es).

**Arxie actions:**
- Recommend candidate datasets.
- Map required variables and preprocessing needs.
- Surface feasibility constraints (coverage, bias, missingness, scale).

**Outputs:**
- Data Options Table
- Feasibility Scorecard

**Exit criteria:** user selects preferred data strategy.

---

## Stage 5 — Experiment & Analysis Design
**Goal:** define what experiment to run and how to analyze outcomes.

**Arxie actions:**
- Identify analogous study designs from prior work.
- Propose experiment templates and analysis plans.
- Surface controls, confounders, and evaluation metrics.

**Outputs:**
- Experiment Flow Diagram
- Analysis Plan Tree
- Confounder & Control Checklist

**Exit criteria:** user confirms experiment + analysis plan.

---

## Stage 6 — Proposal Assembly
**Goal:** compile a structured proposal draft from approved state.

**Arxie actions:**
- Assemble:
  - Background / Gap
  - Hypothesis
  - Data plan
  - Experiment design
  - Analysis plan
  - Expected outcomes
  - Risks/limitations
  - Supporting and opposing evidence

**Outputs:**
- Editable Proposal Draft
- Exportable markdown/PDF

**Exit criteria:** user accepts draft for export.

---

## 5) Interface & Information Architecture

### Layout (Dashboard)
1. **Left panel:** workflow navigator (stages + status + branch selector)
2. **Center panel:** editable canvas (current stage + artifact editor)
3. **Right panel:** evidence/provenance inspector
4. **Bottom panel:** conversation thread (interaction driver)

### Visual artifact tabs (v0.2)
- Mindmap
- Logical Backbone Tree
- Evidence Map (support vs contradict)
- Method/Analysis Tree
- Experiment Flow Diagram
- Outcome Comparison Matrix

### Interaction model
- Conversation-first, artifact-synchronized
- Stage-gated progression
- Explicit user confirmation at stage exits

---

## 6) Branching Model

Branching is first-class in v0.2.

### Branch points
- Hypothesis branch
- Dataset branch
- Method/analysis branch

### Branch operations
- Create branch from any approved node
- Compare branches on:
  - Evidence support strength
  - Feasibility
  - Risk
  - Expected contribution
- Promote one branch as “primary”

---

## 7) Functional Requirements

### FR-1 Iterative dialogue engine
System asks targeted follow-ups and updates state incrementally.

### FR-2 Proposal state engine
Persistent structured state for all stages and artifacts.

### FR-3 Multi-artifact generation
Generate and update visual artifacts from shared state.

### FR-4 Cross-artifact synchronization
Edits in one artifact propagate dependency updates to others.

### FR-5 Evidence traceability
Each evidence-backed claim links to source references.

### FR-6 Branch management
Support create/compare/select branch workflows.

### FR-7 Proposal assembly and export
Generate editable draft and export in markdown/PDF.

---

## 8) Non-Functional Requirements

1. **Responsiveness:**
   - chat turn p95 <= 5s for non-heavy operations
   - artifact refresh p95 <= 10s for standard state size
2. **Reliability:** autosave + recoverable session state
3. **Consistency:** deterministic state updates for same action
4. **Transparency:** provenance and confidence labels visible per claim/node
5. **Privacy:** user project data isolated per workspace/session

---

## 9) Success Criteria (v0.2)

### Product quality metrics
- User can complete all 6 stages in one guided session.
- >=90% of proposal claims marked evidence-backed or explicitly speculative.
- Hypothesis revision triggers synchronized updates across dependent artifacts.
- User can generate at least 2 branch alternatives and compare them.
- Final draft includes all required sections with linked references.

### Researcher-outcome metrics (pilot)
- Reduced time-to-first-structured-proposal draft.
- Increased perceived clarity of research logic.
- Improved confidence in evidence traceability.

---

## 10) Risks & Mitigations

1. **UI complexity overload**
   - Mitigation: guided mode + progressive disclosure
2. **Artifact drift / inconsistency**
   - Mitigation: shared state graph + dependency invalidation flags
3. **Low-quality branch explosion**
   - Mitigation: branch quality scoring + capped active branches
4. **Evidence noise in domain-agnostic setup**
   - Mitigation: explicit relevance thresholds + user pinning

---

## 11) Milestone Plan

### M1 — Workflow + state foundation
- stage engine
- state schema
- conversation-to-state updates

### M2 — Dashboard shell
- 3-pane layout + stage navigator
- evidence inspector

### M3 — Core artifacts
- logical tree + evidence map + hypothesis tree

### M4 — Planning artifacts
- data/feasibility view + experiment/analysis diagram + outcome matrix

### M5 — Branching + sync
- branch operations + cross-artifact dependency refresh

### M6 — Proposal output
- draft assembly + export + polish

---

## 12) Open Questions

1. Default tab order for first-time users.
2. Auto-refresh vs manual-refresh preference for heavy artifacts.
3. How strict stage locking should be after user confirmation.
4. Whether v0.2 includes lightweight template presets by domain.

---

## 13) Summary

Arxie v0.2 shifts from "answering research questions" to "co-creating research proposals" through iterative conversation, evidence-aware reasoning, and synchronized visual artifacts in a dashboard workflow.
