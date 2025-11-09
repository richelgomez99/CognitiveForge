"""
Data models for the CognitiveForge dialectical synthesis system.

This module defines Pydantic models for structured LLM outputs and the AgentState TypedDict
for LangGraph state management.
"""

from typing import List, Optional, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, field_validator
from langgraph.graph import add_messages


# =============================================================================
# Pydantic Models for Structured LLM Outputs
# =============================================================================

class Evidence(BaseModel):
    """
    Evidence supporting a claim in the dialectical process.
    
    Attributes:
        source_url: URL of the source document or research paper
        snippet: Relevant text snippet from the source
        relevance_score: Optional relevance score (0.0-1.0)
    """
    source_url: str = Field(description="URL of the source document")
    snippet: str = Field(description="Relevant text excerpt from source", min_length=10)
    relevance_score: Optional[float] = Field(default=None, description="Relevance score (0.0-1.0)", ge=0.0, le=1.0)
    
    @field_validator('source_url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Ensure source_url is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_url cannot be empty")
        return v.strip()


class ConflictingEvidence(BaseModel):
    """
    Counter-evidence discovered by Skeptic that contradicts thesis claims.
    
    Tier 1: Used by Skeptic agent for active counter-research (US3).
    
    Attributes:
        source_url: URL of the contradicting paper
        snippet: Relevant excerpt explaining the contradiction
        relevance_score: How relevant this counter-evidence is (0.0-1.0)
        discovered_by: Which agent discovered this (e.g., "skeptic_counter")
    """
    source_url: str = Field(description="URL of the contradicting paper", min_length=10)
    snippet: str = Field(description="Relevant excerpt from paper", max_length=300)
    relevance_score: float = Field(description="Relevance score (0.0-1.0)", ge=0.0, le=1.0)
    discovered_by: str = Field(default="skeptic_counter", description="Discovery method")


class Thesis(BaseModel):
    """
    Analyst's initial claim with supporting evidence.
    
    Attributes:
        claim: The main claim or hypothesis
        reasoning: Explanation of how evidence supports the claim
        evidence: List of Evidence objects (minimum 2 required)
    """
    claim: str = Field(description="Main claim or hypothesis", min_length=20)
    reasoning: str = Field(description="Explanation connecting evidence to claim", min_length=50)
    evidence: List[Evidence] = Field(description="Supporting evidence (min 2)", min_length=2)
    
    @field_validator('evidence')
    @classmethod
    def validate_evidence_count(cls, v: List[Evidence]) -> List[Evidence]:
        """Ensure at least 2 evidence items are provided."""
        if len(v) < 2:
            raise ValueError("Thesis must have at least 2 evidence items")
        return v


class Antithesis(BaseModel):
    """
    Skeptic's evaluation of the Thesis, identifying contradictions or weaknesses.
    
    Attributes:
        contradiction_found: Whether contradictions or weaknesses were identified
        counter_claim: Alternative interpretation or refutation (if contradiction found)
        conflicting_evidence: List of ConflictingEvidence contradicting the Thesis (Tier 1: US3)
        critique: Detailed explanation of identified weaknesses
    """
    contradiction_found: bool = Field(description="Whether contradictions were identified")
    counter_claim: Optional[str] = Field(default=None, description="Alternative interpretation or refutation")
    conflicting_evidence: List[ConflictingEvidence] = Field(default_factory=list, description="Counter-evidence contradicting the Thesis")
    critique: str = Field(description="Detailed explanation of weaknesses or contradictions", min_length=30)
    
    @field_validator('counter_claim')
    @classmethod
    def validate_counter_claim(cls, v: Optional[str], info) -> Optional[str]:
        """Ensure counter_claim exists if contradiction_found is True."""
        if info.data.get('contradiction_found') and not v:
            raise ValueError("counter_claim is required when contradiction_found is True")
        return v


class Synthesis(BaseModel):
    """
    Synthesizer's novel insight integrating Thesis and Antithesis.
    
    Attributes:
        novel_insight: The synthesized novel insight or conclusion
        supporting_claims: List of claims from thesis/antithesis that support synthesis
        evidence_lineage: List of all unique source URLs (min 3)
        confidence_score: Confidence in the synthesis (0.0-1.0)
        novelty_score: Self-assessed novelty (0.0-1.0), evaluated by Synthesizer
        reasoning: Explanation of synthesis derivation
    """
    novel_insight: str = Field(description="Synthesized novel insight or conclusion", min_length=50)
    supporting_claims: List[str] = Field(description="Claims supporting the synthesis")
    evidence_lineage: List[str] = Field(description="All unique source URLs (min 3)", min_length=3)
    confidence_score: float = Field(description="Confidence in synthesis (0.0-1.0)", ge=0.0, le=1.0)
    novelty_score: float = Field(
        description="Self-assessed novelty (0.0-1.0) generated by Synthesizer in same LLM call; evaluates how novel the synthesis is compared to input thesis and antithesis",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(description="Explanation of synthesis derivation", min_length=50)
    
    @field_validator('evidence_lineage')
    @classmethod
    def validate_evidence_lineage(cls, v: List[str]) -> List[str]:
        """Ensure at least 3 unique source URLs."""
        unique_urls = list(set(v))
        if len(unique_urls) < 3:
            raise ValueError("evidence_lineage must contain at least 3 unique source URLs")
        return unique_urls
    
    @field_validator('confidence_score', 'novelty_score')
    @classmethod
    def validate_scores(cls, v: float) -> float:
        """Ensure scores are within valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Scores must be between 0.0 and 1.0")
        return v


# =============================================================================
# Knowledge Discovery Models (for Paper Discovery Engine)
# =============================================================================

class DiscoveryStrategy(BaseModel):
    """
    AI-determined strategy for how many papers to discover (Tier 1: US4).
    
    Used by: Analyst agent to adaptively determine paper limits based on claim complexity.
    
    Attributes:
        initial_papers: Number of papers to fetch initially (2-10)
        follow_up_needed: Whether follow-up discovery is needed after initial results
        follow_up_papers: Number of additional papers if follow-up needed (0-5)
        reasoning: Explanation of why this strategy was chosen
    """
    initial_papers: int = Field(
        ge=2,
        le=10,
        description="Number of papers to fetch in initial discovery"
    )
    follow_up_needed: bool = Field(
        description="Whether follow-up discovery is needed after initial results"
    )
    follow_up_papers: int = Field(
        ge=0,
        le=5,
        description="Number of additional papers if follow-up needed"
    )
    reasoning: str = Field(
        min_length=20,
        max_length=300,
        description="Explanation of why this strategy was chosen"
    )


class KeywordExtractionOutput(BaseModel):
    """
    Structured output from LLM keyword extraction (Tier 1: US1).
    
    Used by: Analyst agent to extract targeted search keywords from thesis claim.
    
    Attributes:
        keywords: List of 3-5 multi-word search phrases
        reasoning: Explanation of keyword selection
    """
    keywords: List[str] = Field(
        description="List of 3-5 multi-word search phrases",
        min_length=3,
        max_length=5
    )
    reasoning: str = Field(
        description="Explanation of keyword selection",
        min_length=20,
        max_length=300
    )
    
    @field_validator('keywords')
    @classmethod
    def validate_keywords(cls, v: List[str]) -> List[str]:
        """Ensure keywords are multi-word phrases."""
        for keyword in v:
            if len(keyword.split()) < 2:
                raise ValueError(f"Keyword '{keyword}' must be a multi-word phrase (e.g., 'neural correlates consciousness')")
        return v


class PaperMetadata(BaseModel):
    """
    Metadata for a discovered academic paper.
    
    Used for:
    - Parsing API responses from arXiv/Semantic Scholar
    - Validating data before Neo4j insertion
    - Returning search results to UI
    
    Attributes:
        title: Paper title
        url: Unique paper URL (primary key)
        abstract: Paper abstract
        authors: List of author names
        published: Publication date (ISO 8601)
        source: Discovery source ("arxiv" | "semantic_scholar" | "manual")
        citation_count: Number of citations
        fields_of_study: Subject classification tags
    """
    title: str = Field(description="Paper title", min_length=10)
    url: str = Field(description="Unique paper URL (primary key)")
    abstract: str = Field(description="Paper abstract", min_length=50)
    authors: List[str] = Field(description="List of author names", min_length=1)
    published: str = Field(description="Publication date (ISO 8601 format)")
    source: str = Field(description="Discovery source", pattern="^(arxiv|semantic_scholar|manual)$")
    citation_count: Optional[int] = Field(default=0, description="Number of citations", ge=0)
    fields_of_study: Optional[List[str]] = Field(default_factory=list, description="Subject classification")


class DiscoveryRequest(BaseModel):
    """
    Request for manual paper discovery.
    
    Used by: POST /discover endpoint
    
    Attributes:
        query: Search query (keywords)
        source: Search source ("arxiv" | "semantic_scholar")
        max_results: Maximum papers to return (1-50)
    """
    query: str = Field(description="Search query (keywords)", min_length=3)
    source: str = Field(description="Search source", pattern="^(arxiv|semantic_scholar)$")
    max_results: int = Field(default=10, description="Maximum papers to return", ge=1, le=50)


class DiscoveryResponse(BaseModel):
    """
    Response for manual paper discovery.
    
    Used by: POST /discover endpoint
    
    Attributes:
        papers: List of discovered papers
        count: Number of papers returned
        source: Source that was searched
        query: Original search query
    """
    papers: List[PaperMetadata] = Field(description="List of discovered papers")
    count: int = Field(description="Number of papers returned", ge=0)
    source: str = Field(description="Source that was searched")
    query: str = Field(description="Original search query")


class AddPapersRequest(BaseModel):
    """
    Request to add papers to Neo4j knowledge graph.
    
    Used by: POST /add_papers endpoint
    
    Attributes:
        papers: Full paper metadata objects to add (preferred)
        paper_urls: List of paper URLs (fallback, deprecated)
        discovered_by: Discovery method identifier
    """
    papers: Optional[List['PaperMetadata']] = Field(default=None, description="Full paper metadata objects")
    paper_urls: Optional[List[str]] = Field(default=None, description="List of paper URLs (fallback)")
    discovered_by: str = Field(default="manual", description="Discovery method identifier")
    
    @field_validator('papers', 'paper_urls')
    @classmethod
    def validate_at_least_one(cls, v, info):
        """Ensure either papers or paper_urls is provided."""
        # This validator will be called for each field
        # We'll do the actual validation in model_validator
        return v
    
    def model_post_init(self, __context):
        """Validate that at least one of papers or paper_urls is provided."""
        if not self.papers and not self.paper_urls:
            raise ValueError("Either 'papers' or 'paper_urls' must be provided")


class AddPapersResponse(BaseModel):
    """
    Response for adding papers to knowledge graph.
    
    Used by: POST /add_papers endpoint
    
    Attributes:
        added: Number of papers successfully added
        skipped: Number of papers skipped (duplicates)
        failed: Number of papers that failed to add
        details: Detailed messages (e.g., duplicate URLs)
    """
    added: int = Field(description="Number of papers successfully added", ge=0)
    skipped: int = Field(description="Number of papers skipped (duplicates)", ge=0)
    failed: int = Field(description="Number of papers that failed to add", ge=0)
    details: List[str] = Field(default_factory=list, description="Detailed messages")


# =============================================================================
# Tier 1: Debate Memory
# =============================================================================

class DebateMemory(TypedDict):
    """
    Memory of rejected claims and objections within a debate session (Tier 1: US2).
    
    Used to prevent circular arguments by tracking what has been rejected and why.
    FIFO eviction when max items (10) exceeded.
    
    Attributes:
        rejected_claims: List of thesis claims that Skeptic rejected
        skeptic_objections: List of Skeptic critique summaries
        weak_evidence_urls: List of paper URLs flagged as low quality
    """
    rejected_claims: List[str]
    skeptic_objections: List[str]
    weak_evidence_urls: List[str]


# =============================================================================
# AgentState for LangGraph
# =============================================================================

class AgentState(TypedDict):
    """
    Central state for the dialectical synthesis LangGraph.
    
    This TypedDict is passed between all agent nodes and accumulates state
    throughout the graph execution.
    
    Attributes:
        messages: Accumulated messages using LangGraph's add_messages reducer
        original_query: The user's research query
        current_thesis: Latest Thesis from Analyst
        current_antithesis: Latest Antithesis from Skeptic
        final_synthesis: Final Synthesis from Synthesizer
        contradiction_report: Description of contradictions (from Skeptic)
        iteration_count: Number of iterations through the debate cycle
        procedural_memory: Procedural heuristics for agent learning (Tier 3)
        debate_memory: Memory of rejected claims and objections (Tier 1: US2)
        current_claim_id: UUID for current claim/iteration (Tier 1: US1)
    """
    messages: Annotated[list, add_messages]
    original_query: str
    current_thesis: Optional[Thesis]
    current_antithesis: Optional[Antithesis]
    final_synthesis: Optional[Synthesis]
    contradiction_report: str
    iteration_count: int
    procedural_memory: str  # For Tier 3 compatibility
    debate_memory: DebateMemory  # Tier 1: Memory for circular argument prevention
    current_claim_id: str  # Tier 1: UUID for claim-specific paper discovery

