"""
Phase 2 data models extending Tier 1 architecture.

New additions:
- Memory models (Episodic, Semantic, Procedural)
- Quality scoring models
- 10-agent state tracking
"""

from typing import List, Optional, Dict
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from datetime import datetime

# Import existing models
from src.models import (
    AgentState,  # Will extend this
    Thesis, Antithesis, Synthesis,
    Evidence, ConflictingEvidence
)


# =============================================================================
# Quality Scoring Models
# =============================================================================

class PaperQualityScoreV2(BaseModel):
    """
    Multi-factor quality assessment for academic papers.

    Attributes:
        citation_count: From Semantic Scholar API
        venue_impact_factor: Manual lookup for top venues (0-100)
        is_peer_reviewed: True if journal/conference, False if preprint
        recency_score: Exponential decay from publication date (0-100)
        methodology_score: From Methodologist agent critique (0-100)
        overall_score: Weighted composite (0-100)
    """
    citation_count: int = Field(ge=0)
    venue_impact_factor: float = Field(ge=0.0, le=100.0, default=50.0)
    is_peer_reviewed: bool
    recency_score: float = Field(ge=0.0, le=100.0)
    methodology_score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    overall_score: float = Field(ge=0.0, le=100.0)

    @classmethod
    def calculate_overall(cls, citation_count: int, venue_if: float,
                         peer_reviewed: bool, recency: float) -> float:
        """
        Weighted composite score.

        Formula: 0.3*citations + 0.25*venue + 0.25*peer + 0.2*recency
        """
        # Normalize citation count (log scale, cap at 1000)
        citation_score = min(100.0, (citation_count / 10.0) * 10)
        peer_score = 100.0 if peer_reviewed else 50.0

        overall = (
            0.30 * citation_score +
            0.25 * venue_if +
            0.25 * peer_score +
            0.20 * recency
        )
        return round(overall, 2)


# =============================================================================
# Memory Models
# =============================================================================

class EpisodicMemory(BaseModel):
    """
    Memory of a past research session (episodic).

    Stores "what happened when" - session summaries for temporal retrieval.
    """
    memory_id: str = Field(description="UUID for this memory")
    session_id: str = Field(description="LangGraph thread_id")
    timestamp: datetime
    query: str = Field(min_length=3)
    synthesis_claim_id: str = Field(description="UUID of final synthesis claim")
    key_papers: List[str] = Field(description="URLs of important papers from session")
    insights: List[str] = Field(description="Key takeaways (1-3 sentences each)")
    embedding: Optional[List[float]] = Field(default=None, description="Query embedding for similarity")


class SemanticMemory(BaseModel):
    """
    Long-term factual knowledge (semantic).

    Stores "what is true" - facts extracted from syntheses with confidence.
    """
    fact_id: str = Field(description="UUID for this fact")
    claim: str = Field(min_length=20, description="The factual claim")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0-1)")
    sources: List[str] = Field(min_length=1, description="Paper URLs supporting this fact")
    created_at: datetime
    last_verified: datetime
    last_accessed: Optional[datetime] = None
    embedding: Optional[List[float]] = Field(default=None, description="Claim embedding")

    def decay_confidence(self, days_since_access: int) -> float:
        """
        Apply time-based decay to confidence.

        Formula: confidence * exp(-days/90)
        After 90 days without access, confidence halves.
        """
        import math
        decay_factor = math.exp(-days_since_access / 90.0)
        return self.confidence * decay_factor


class ProceduralMemory(BaseModel):
    """
    Learned research strategies (procedural).

    Stores "how to do X" - successful patterns for future use.
    """
    strategy_id: str
    description: str = Field(min_length=20, description="What this strategy does")
    success_rate: float = Field(ge=0.0, le=1.0, description="Historical success (0-1)")
    usage_count: int = Field(ge=0, description="Times used")
    last_used: datetime


# =============================================================================
# AgentStateV2 - Extended for 10-Agent Architecture
# =============================================================================

class AgentContribution(TypedDict):
    """
    Tracks one agent's contribution to the debate.
    """
    agent_name: str  # "Literature Surveyor", "Pattern Recognizer", etc.
    round_number: int
    content: str  # The agent's output (claim, critique, etc.)
    papers_cited: List[str]  # URLs
    timestamp: datetime


class AgentStateV2(TypedDict):
    """
    Extended state for 10-agent architecture with persistent memory.

    Inherits all fields from AgentState, adds:
    - Agent contribution tracking
    - Memory retrieval fields
    - Quality scores for discovered papers
    - Coordinator decisions
    """
    # =========================================================================
    # Tier 1 Fields (Inherited from AgentState)
    # =========================================================================
    messages: List  # LangGraph add_messages reducer
    original_query: str
    current_thesis: Optional[Thesis]
    current_antithesis: Optional[Antithesis]
    final_synthesis: Optional[Synthesis]
    contradiction_report: str
    iteration_count: int
    procedural_memory: str  # Legacy Tier 3 field
    debate_memory: Dict  # Rejected claims, objections, weak URLs
    current_claim_id: str
    synthesis_mode: Optional[str]
    consecutive_high_similarity_count: int
    last_similarity_score: Optional[float]
    conversation_history: List
    current_round_papers_analyst: List[str]
    current_round_papers_skeptic: List[str]

    # =========================================================================
    # Phase 2 New Fields
    # =========================================================================

    # Agent Coordination
    agent_contributions: List[AgentContribution]  # Full debate transcript
    active_agents: List[str]  # Which agents Coordinator activated this round
    coordinator_reasoning: str  # Why Coordinator chose these agents

    # Memory Retrieval
    retrieved_episodic_memories: List[EpisodicMemory]  # Past sessions
    retrieved_semantic_facts: List[SemanticMemory]  # Long-term knowledge
    memory_context_summary: str  # "System remembers: [summary]"

    # Quality Tracking
    paper_quality_scores: Dict[str, PaperQualityScoreV2]  # URL → quality
    high_quality_papers: List[str]  # URLs with score >80
    low_quality_papers: List[str]  # URLs with score <40

    # PDF Processing
    pdf_extraction_status: Dict[str, str]  # URL → "success"|"failed"|"pending"
    extracted_text_chunks: Dict[str, List[str]]  # URL → list of text chunks
