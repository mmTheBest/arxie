"""Proposal domain package."""

from ra.proposal.branching import (
    BranchAlreadyExistsError,
    BranchComparisonItem,
    BranchComparisonResult,
    BranchConfidenceLabel,
    BranchNotFoundError,
    BranchScorecard,
    HypothesisBranch,
    HypothesisBranchManager,
)
from ra.proposal.artifact_sync import (
    ArtifactNode,
    ArtifactNodeNotFoundError,
    ArtifactSyncManager,
    ProposalArtifact,
    ProvenanceNotFoundError,
)
from ra.proposal.evidence_mapper import (
    EvidenceItem,
    EvidenceMapper,
    EvidenceMappingResult,
    LandscapeSummary,
)
from ra.proposal.models import ProposalStage, ProposalWorkflowState, StageState
from ra.proposal.session_service import ProposalSessionService
from ra.proposal.stage_engine import (
    ProposalStageEngine,
    StageDiagnostics,
    StageTransitionError,
    StageTransitionReason,
)
from ra.proposal.store import (
    InMemoryProposalSessionStore,
    ProposalSessionSnapshot,
    SessionAlreadyExistsError,
    SessionNotFoundError,
    SessionVersionConflictError,
)

__all__ = [
    "ProposalStage",
    "ProposalWorkflowState",
    "StageState",
    "BranchScorecard",
    "HypothesisBranch",
    "BranchComparisonItem",
    "BranchComparisonResult",
    "BranchConfidenceLabel",
    "BranchNotFoundError",
    "BranchAlreadyExistsError",
    "HypothesisBranchManager",
    "ProposalArtifact",
    "ArtifactNode",
    "ArtifactSyncManager",
    "ArtifactNodeNotFoundError",
    "ProvenanceNotFoundError",
    "EvidenceItem",
    "EvidenceMappingResult",
    "LandscapeSummary",
    "EvidenceMapper",
    "ProposalSessionSnapshot",
    "InMemoryProposalSessionStore",
    "SessionAlreadyExistsError",
    "SessionNotFoundError",
    "SessionVersionConflictError",
    "ProposalSessionService",
    "ProposalStageEngine",
    "StageDiagnostics",
    "StageTransitionError",
    "StageTransitionReason",
]
