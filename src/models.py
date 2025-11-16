"""
Data models for the CognitiveForge dialectical synthesis system.

This module defines Pydantic models for structured LLM outputs and the AgentState TypedDict
for LangGraph state management.
"""

from typing import List, Optional, Annotated, Dict, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, field_validator, EmailStr
from langgraph.graph import add_messages
from datetime import datetime
from enum import Enum
import uuid


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


# =============================================================================
# Tier 2: Paper Quality Assessment Models
# =============================================================================

class PaperQualityScore(BaseModel):
    """
    Quality assessment for academic papers (Tier 2: US4).
    
    Attributes:
        citation_count: Number of citations from Semantic Scholar
        venue_impact_factor: Journal/venue impact factor (0-100), default 5.0 for unknown
        is_peer_reviewed: True if peer-reviewed journal/conference, False if preprint
        star_rating: Visual quality score (1-5 stars)
        quality_score: Computed overall quality (0-100)
    """
    citation_count: int = Field(ge=0, description="Number of citations from Semantic Scholar")
    venue_impact_factor: float = Field(
        ge=0.0, 
        le=100.0, 
        default=5.0, 
        description="Journal/venue impact factor (0-100). Default 5.0 for unknown venues."
    )
    is_peer_reviewed: bool = Field(description="True if peer-reviewed, False if preprint (arXiv, bioRxiv)")
    star_rating: int = Field(ge=1, le=5, description="Visual quality score (1-5 stars)")
    quality_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Overall quality score (0-100) computed from citations + venue + peer-review status"
    )


# =============================================================================
# Tier 2: Comprehensive Synthesis Models (Nested)
# =============================================================================

class RoundSummary(BaseModel):
    """
    Summary of one dialectical round for comprehensive synthesis (Tier 2: US5).
    
    Simplified for Gemini structured output - constraints removed to avoid schema complexity.
    """
    round_number: int
    thesis_claim: str
    rejection_reason: Optional[str] = None
    key_insights: List[str]


class EvidenceItem(BaseModel):
    """
    Paper supporting the synthesis with explanation (Tier 2: US5).
    
    Simplified for Gemini structured output - constraints removed.
    """
    url: str
    title: str
    how_it_supports: str


class CounterEvidenceResolution(BaseModel):
    """
    Counter-evidence and how synthesis addresses it (Tier 2: US5).
    
    Simplified for Gemini structured output - constraints removed.
    """
    url: str
    title: str
    contradiction: str
    resolution: str


class PaperSummary(BaseModel):
    """
    Key paper with full context for comprehensive synthesis (Tier 2: US5).
    
    Simplified for Gemini structured output - constraints removed.
    """
    url: str
    title: str
    authors: List[str]
    year: Optional[int] = None
    venue: Optional[str] = None
    role_in_synthesis: str


# =============================================================================
# Tier 2: Expanded Synthesis Model (Research Report Quality)
# =============================================================================

class SynthesisLLMOutput(BaseModel):
    """
    Simplified synthesis model for LLM structured output (Tier 2).
    
    This version has relaxed constraints to avoid Gemini schema complexity limits.
    After generation, it's validated and converted to the full Synthesis model.
    """
    # Core insight
    novel_insight: str
    
    # Dialectical journey
    dialectical_summary: str
    rounds_explored: List[RoundSummary]
    
    # Evidence synthesis
    supporting_evidence: List[EvidenceItem]
    contradicting_evidence_addressed: List[CounterEvidenceResolution] = []
    
    # Comprehensive reasoning
    synthesis_reasoning: str
    
    # Quality metrics
    confidence_score: float = Field(ge=0.0, le=100.0)
    confidence_justification: str
    novelty_score: float = Field(ge=0.0, le=100.0)
    novelty_justification: str
    
    # Practical implications
    practical_implications: List[str]
    testable_predictions: List[str]
    open_questions: List[str]
    
    # References
    evidence_lineage: List[str]
    key_papers: List[PaperSummary]
    
    # Legacy fields
    supporting_claims: List[str]
    reasoning: str


class Synthesis(BaseModel):
    """
    Comprehensive research synthesis report (Tier 2: Expanded for US5).
    
    This model was expanded from a brief 200-word summary to a comprehensive 800-1500 word
    research report including dialectical journey, evidence synthesis, and practical implications.
    
    Attributes:
        # Core insight (Tier 1 - preserved)
        novel_insight: Main synthesis claim (2-3 sentences)
        
        # Dialectical journey (Tier 2: NEW)
        dialectical_summary: Summary of debate evolution (300-500 words)
        rounds_explored: List of round summaries (what was explored and why rejected)
        
        # Evidence synthesis (Tier 2: EXPANDED)
        supporting_evidence: Papers supporting synthesis with explanations
        contradicting_evidence_addressed: Counter-evidence and how synthesis resolves it
        
        # Comprehensive reasoning (Tier 2: EXPANDED)
        synthesis_reasoning: How this resolves tensions (400-600 words)
        
        # Quality metrics (Tier 1 - preserved, Tier 2: added justifications)
        confidence_score: Confidence score (0-100, was 0-1 in Tier 1)
        confidence_justification: Why this confidence level
        novelty_score: Novelty score (0-100, was 0-1 in Tier 1)
        novelty_justification: What's novel about this
        
        # Practical implications (Tier 2: NEW)
        practical_implications: Practical applications (2-5 items)
        testable_predictions: Testable predictions (2-5 items)
        open_questions: Remaining open questions (2-5 items)
        
        # References (Tier 1 - preserved, Tier 2: annotated)
        evidence_lineage: All URLs cited (min 3)
        key_papers: Top 5-10 papers with full context
        
        # Legacy fields (Tier 1 - preserved for backward compatibility)
        supporting_claims: List of claims from thesis/antithesis (Tier 1)
        reasoning: Brief synthesis reasoning (Tier 1, superseded by synthesis_reasoning)
    """
    # Core insight (Tier 1 - preserved)
    novel_insight: str = Field(
        description="Main synthesis claim (2-3 sentences, 50-1000 characters for Tier 2 comprehensive reports)",
        min_length=50,
        max_length=1000
    )
    
    # Dialectical journey (Tier 2: NEW)
    dialectical_summary: str = Field(
        description="Summary of debate evolution showing how rounds led to final insight (300-500 words)",
        min_length=300,
        max_length=3000
    )
    rounds_explored: List[RoundSummary] = Field(
        description="Summaries of dialectical rounds (what was explored and why rejected)",
        min_length=1,
        max_length=10
    )
    
    # Evidence synthesis (Tier 2: EXPANDED)
    supporting_evidence: List[EvidenceItem] = Field(
        description="Papers supporting synthesis with specific roles",
        min_length=2,
        max_length=15
    )
    contradicting_evidence_addressed: List[CounterEvidenceResolution] = Field(
        description="Counter-evidence and how synthesis resolves tensions",
        min_length=0,
        max_length=10
    )
    
    # Comprehensive reasoning (Tier 2: EXPANDED)
    synthesis_reasoning: str = Field(
        description="Comprehensive explanation of how this synthesis resolves tensions (400-600 words)",
        min_length=400,
        max_length=3500
    )
    
    # Quality metrics (Tier 1 - preserved, Tier 2: expanded with justifications)
    confidence_score: float = Field(
        description="Confidence in synthesis (0-100, was 0-1 in Tier 1)",
        ge=0.0,
        le=100.0
    )
    confidence_justification: str = Field(
        description="Why this confidence level (100-300 characters)",
        min_length=100,
        max_length=300
    )
    novelty_score: float = Field(
        description="Self-assessed novelty (0-100, was 0-1 in Tier 1)",
        ge=0.0,
        le=100.0
    )
    novelty_justification: str = Field(
        description="What's novel about this synthesis (100-300 characters)",
        min_length=100,
        max_length=300
    )
    
    # Practical implications (Tier 2: NEW)
    practical_implications: List[str] = Field(
        description="Practical applications of this synthesis",
        min_length=2,
        max_length=5
    )
    testable_predictions: List[str] = Field(
        description="Testable predictions derived from synthesis",
        min_length=2,
        max_length=5
    )
    open_questions: List[str] = Field(
        description="Remaining open questions for future research",
        min_length=2,
        max_length=5
    )
    
    # References (Tier 1 - preserved, Tier 2: annotated)
    evidence_lineage: List[str] = Field(
        description="All unique source URLs cited (min 3)",
        min_length=3
    )
    key_papers: List[PaperSummary] = Field(
        description="Top 3-10 papers with full context and role in synthesis (minimum 3 for impasse/limited evidence cases)",
        min_length=3,
        max_length=10
    )
    
    # Legacy fields (Tier 1 - preserved for backward compatibility)
    supporting_claims: List[str] = Field(
        description="Claims from thesis/antithesis supporting synthesis (Tier 1 legacy)"
    )
    reasoning: str = Field(
        description="Brief synthesis reasoning (Tier 1 legacy, superseded by synthesis_reasoning)",
        min_length=50
    )
    
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
        """Ensure scores are within valid range (Tier 2: updated to 0-100 scale)."""
        if not 0.0 <= v <= 100.0:
            raise ValueError("Scores must be between 0.0 and 100.0")
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

class ConversationRound(TypedDict):
    """
    One round of dialectical conversation for UI display (Tier 2: US1).
    
    Attributes:
        round_number: Round number (1-indexed)
        thesis: Analyst's thesis for this round
        antithesis: Skeptic's response for this round
        papers_analyst: Papers discovered by Analyst in this round
        papers_skeptic: Papers discovered by Skeptic in this round
    """
    round_number: int
    thesis: Thesis
    antithesis: Antithesis
    papers_analyst: List[str]  # URLs
    papers_skeptic: List[str]  # URLs


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
        synthesis_mode: Mode for synthesis ("standard" | "impasse" | "exhausted_attempts") (Tier 1: T089)
        consecutive_high_similarity_count: Count of consecutive rejections with high similarity (>0.75)
        last_similarity_score: Most recent similarity score from circular argument check
        
        # Tier 2: Visualization & UX
        conversation_history: List of conversation rounds for UI display (Tier 2: US1)
        current_round_papers_analyst: Papers discovered by Analyst in current round (Tier 2: US1)
        current_round_papers_skeptic: Papers discovered by Skeptic in current round (Tier 2: US1)
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
    synthesis_mode: Optional[str]  # Tier 1: T089 - "standard" | "impasse" | "exhausted_attempts"
    consecutive_high_similarity_count: int  # Natural termination: track consecutive high-similarity rejections
    last_similarity_score: Optional[float]  # Natural termination: most recent similarity score
    
    # Tier 2: Visualization & UX
    conversation_history: List[ConversationRound]  # Tier 2: US1 - Conversational thread view
    current_round_papers_analyst: List[str]  # Tier 2: US1 - Papers discovered by Analyst in current round
    current_round_papers_skeptic: List[str]  # Tier 2: US1 - Papers discovered by Skeptic in current round

    # Epic 5: 10-Agent Architecture - New agent outputs
    curated_papers: Optional[List[str]]  # Paper Curator: Ranked paper URLs
    evidence_validation_report: Optional[Dict[str, Any]]  # Evidence Validator: Alignment scores
    bias_detection_report: Optional[Dict[str, Any]]  # Bias Detector: Detected biases
    consistency_check_report: Optional[Dict[str, Any]]  # Consistency Checker: Historical consistency
    counter_perspectives: Optional[List[str]]  # Counter-Perspective: Alternative viewpoints
    novelty_assessment: Optional[Dict[str, Any]]  # Novelty Assessor: Innovation scores
    synthesis_review: Optional[Dict[str, Any]]  # Synthesis Reviewer: QA results


# =============================================================================
# Epic 5: 10-Agent Architecture - Agent Output Models
# =============================================================================

class PaperScore(BaseModel):
    """Score for a curated paper."""
    url: str = Field(description="Paper URL")
    title: str = Field(description="Paper title")
    relevance_score: float = Field(description="Relevance to query (0-1)", ge=0.0, le=1.0)
    quality_score: float = Field(description="Quality score (0-1)", ge=0.0, le=1.0)
    citation_count: int = Field(description="Citation count", ge=0)
    rank: int = Field(description="Rank in curated list", ge=0)  # 0 = unranked
    combined_score: float = Field(default=0.0, description="Combined relevance + quality score", ge=0.0, le=1.0)


class CurationReport(BaseModel):
    """Output from Paper Curator agent."""
    curated_papers: List[PaperScore] = Field(description="Ranked papers")
    total_discovered: int = Field(description="Total papers discovered", ge=0)
    filtered_count: int = Field(description="Papers filtered out", ge=0)
    avg_quality: float = Field(description="Average quality score", ge=0.0, le=1.0)


class EvidenceAlignment(BaseModel):
    """Alignment assessment for one evidence item."""
    evidence_url: str = Field(description="Evidence source URL")
    claim_excerpt: str = Field(description="Relevant claim excerpt", max_length=200)
    alignment_score: float = Field(description="Alignment strength (0-1)", ge=0.0, le=1.0)
    alignment_category: str = Field(description="strong | medium | weak")
    reasoning: str = Field(description="Alignment reasoning", max_length=300)


class ValidationReport(BaseModel):
    """Output from Evidence Validator agent."""
    alignments: List[EvidenceAlignment] = Field(description="Per-evidence alignments")
    overall_strength: float = Field(description="Overall evidence strength (0-1)", ge=0.0, le=1.0)
    strong_count: int = Field(description="Count of strong alignments", ge=0)
    weak_count: int = Field(description="Count of weak alignments", ge=0)
    recommendation: str = Field(description="Validation recommendation", max_length=200)


class DetectedBias(BaseModel):
    """Single detected bias."""
    bias_type: str = Field(description="Type of bias (selection, methodological, etc.)")
    confidence: float = Field(description="Detection confidence (0-1)", ge=0.0, le=1.0)
    description: str = Field(description="Bias description", max_length=300)
    affected_papers: List[str] = Field(description="URLs of affected papers")


class BiasReport(BaseModel):
    """Output from Bias Detector agent."""
    detected_biases: List[DetectedBias] = Field(description="Detected biases")
    bias_score: float = Field(description="Overall bias score (0-1, higher=more bias)", ge=0.0, le=1.0)
    recommendation: str = Field(description="Balancing recommendation", max_length=200)


class ConsistencyCheck(BaseModel):
    """Output from Consistency Checker agent."""
    similar_past_claims: List[str] = Field(description="Similar claims from past sessions")
    contradiction_found: bool = Field(description="Whether contradicts past claims")
    consistency_score: float = Field(description="Consistency score (0-1)", ge=0.0, le=1.0)
    recommendation: str = Field(description="Consistency recommendation", max_length=200)


class CounterPerspectiveItem(BaseModel):
    """Single counter-perspective."""
    perspective: str = Field(description="Alternative viewpoint", max_length=500)
    supporting_evidence: List[str] = Field(description="URLs supporting this perspective")
    strength: float = Field(description="Perspective strength (0-1)", ge=0.0, le=1.0)


class NoveltyAssessment(BaseModel):
    """Output from Novelty Assessor agent."""
    novelty_score: float = Field(description="Independent novelty score (0-100)", ge=0.0, le=100.0)
    innovation_factors: List[str] = Field(description="Factors contributing to novelty")
    comparison_to_baseline: str = Field(description="How this differs from existing work", max_length=300)
    assessment_confidence: float = Field(description="Assessment confidence (0-1)", ge=0.0, le=1.0)


class SynthesisReview(BaseModel):
    """Output from Synthesis Reviewer agent."""
    qa_passed: bool = Field(description="Whether synthesis passes QA")
    quality_score: float = Field(description="Overall quality (0-100)", ge=0.0, le=100.0)
    consistency_check: bool = Field(description="Internal consistency validated")
    evidence_check: bool = Field(description="Evidence lineage validated")
    issues_found: List[str] = Field(description="Issues requiring attention")
    recommendation: str = Field(description="Accept | Revise | Reject")


# =============================================================================
# Epic 4: Persistent Memory System - Enums
# =============================================================================

class SessionStatus(str, Enum):
    """Status of a debate session."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    DELETED = "deleted"


class Permission(str, Enum):
    """Permission levels for workspace access."""
    VIEW = "view"
    EDIT = "edit"
    ADMIN = "admin"


class MemoryPatternType(str, Enum):
    """Types of memory patterns recognized across sessions."""
    CLAIM_STRUCTURE = "claim_structure"
    SKEPTIC_OBJECTION = "skeptic_objection_type"
    DEBATE_STYLE = "debate_style"
    EVIDENCE_QUALITY = "evidence_quality"
    CONVERGENCE_PATTERN = "convergence_pattern"


# =============================================================================
# Epic 4: Persistent Memory System - User & Workspace Models
# =============================================================================

class UserProfile(BaseModel):
    """
    User profile for authentication and workspace isolation (Epic 4: Task 2).

    Attributes:
        user_id: Unique identifier (UUID)
        username: Display name
        email: Email address
        created_at: Account creation timestamp
        last_login: Last login timestamp
        preferences: User settings (theme, notification preferences, etc.)
    """
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique user identifier")
    username: str = Field(description="Username (3-50 characters)", min_length=3, max_length=50)
    email: EmailStr = Field(description="Email address")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Account creation timestamp")
    last_login: Optional[datetime] = Field(default=None, description="Last login timestamp")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences (JSON)")


class Workspace(BaseModel):
    """
    Workspace for organizing sessions (Epic 4: Task 2).

    Attributes:
        workspace_id: Unique identifier (UUID)
        name: Workspace name
        owner_id: User ID of owner
        created_at: Creation timestamp
        description: Workspace description
        is_public: Whether workspace is publicly accessible
        settings: Workspace-specific settings
    """
    workspace_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique workspace identifier")
    name: str = Field(description="Workspace name (3-100 characters)", min_length=3, max_length=100)
    owner_id: str = Field(description="User ID of workspace owner")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    description: Optional[str] = Field(default=None, description="Workspace description", max_length=500)
    is_public: bool = Field(default=False, description="Whether workspace is publicly accessible")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Workspace settings (JSON)")


class WorkspacePermission(BaseModel):
    """
    Permission mapping for workspace access (Epic 4: Task 2).

    Attributes:
        workspace_id: Workspace identifier
        user_id: User identifier
        permission: Permission level (view, edit, admin)
        granted_at: When permission was granted
        granted_by: User ID who granted permission
    """
    workspace_id: str = Field(description="Workspace identifier")
    user_id: str = Field(description="User identifier")
    permission: Permission = Field(description="Permission level")
    granted_at: datetime = Field(default_factory=datetime.utcnow, description="Permission grant timestamp")
    granted_by: str = Field(description="User ID who granted this permission")


# =============================================================================
# Epic 4: Persistent Memory System - Session Models
# =============================================================================

class SessionMetadata(BaseModel):
    """
    Metadata for a debate session (Epic 4: Task 1).

    Attributes:
        session_id: Unique identifier (UUID)
        workspace_id: Parent workspace identifier
        thread_id: LangGraph thread identifier (for checkpoint compatibility)
        title: User-provided session title
        original_query: Original research question
        status: Current session status
        created_at: Session creation timestamp
        updated_at: Last update timestamp
        completed_at: Completion timestamp (if completed)
        iteration_count: Number of debate iterations
        final_synthesis_id: ID of final synthesis (if completed)
        tags: Session tags for categorization
        metadata: Custom metadata fields
    """
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique session identifier")
    workspace_id: str = Field(description="Parent workspace identifier")
    thread_id: str = Field(description="LangGraph thread identifier")
    title: str = Field(description="Session title (3-200 characters)", min_length=3, max_length=200)
    original_query: str = Field(description="Original research question", min_length=10)
    status: SessionStatus = Field(default=SessionStatus.ACTIVE, description="Session status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    iteration_count: int = Field(default=0, description="Number of debate iterations", ge=0)
    final_synthesis_id: Optional[str] = Field(default=None, description="ID of final synthesis")
    tags: List[str] = Field(default_factory=list, description="Session tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")


class SessionRecord(BaseModel):
    """
    Full session record including state snapshot (Epic 4: Task 1).

    Attributes:
        session_metadata: Session metadata
        state_snapshot: Optional snapshot of AgentState
        debate_moments: List of debate moments in this session
        created_by: User ID who created the session
    """
    session_metadata: SessionMetadata = Field(description="Session metadata")
    state_snapshot: Optional[Dict[str, Any]] = Field(default=None, description="AgentState snapshot")
    debate_moments: List['DebateMoment'] = Field(default_factory=list, description="Debate moments")
    created_by: str = Field(description="User ID who created session")


class CreateSessionRequest(BaseModel):
    """
    Request to create a new session (Epic 4: Task 1).

    Attributes:
        workspace_id: Workspace to create session in
        title: Session title
        query: Research question
        tags: Optional session tags
    """
    workspace_id: str = Field(description="Workspace identifier")
    title: str = Field(description="Session title (3-200 characters)", min_length=3, max_length=200)
    query: str = Field(description="Research question", min_length=10)
    tags: List[str] = Field(default_factory=list, description="Session tags")


class UpdateSessionRequest(BaseModel):
    """
    Request to update session metadata (Epic 4: Task 1).

    Attributes:
        title: Updated session title
        status: Updated session status
        tags: Updated session tags
    """
    title: Optional[str] = Field(default=None, description="Updated session title", min_length=3, max_length=200)
    status: Optional[SessionStatus] = Field(default=None, description="Updated session status")
    tags: Optional[List[str]] = Field(default=None, description="Updated session tags")


class SessionListResponse(BaseModel):
    """
    Response for listing sessions (Epic 4: Task 1).

    Attributes:
        sessions: List of session metadata
        total: Total number of sessions
        page: Current page number
        page_size: Page size
    """
    sessions: List[SessionMetadata] = Field(description="List of session metadata")
    total: int = Field(description="Total number of sessions", ge=0)
    page: int = Field(description="Current page number", ge=1)
    page_size: int = Field(description="Page size", ge=1, le=100)


# =============================================================================
# Epic 4: Persistent Memory System - Memory Models
# =============================================================================

class DebateMoment(BaseModel):
    """
    A single moment in a debate session (Epic 4: Task 3).

    Represents one agent's contribution (thesis, antithesis, or synthesis).

    Attributes:
        moment_id: Unique identifier (UUID)
        session_id: Parent session identifier
        round_number: Debate round number (1-indexed)
        agent_type: Which agent produced this moment
        content: The actual content (claim, critique, or synthesis)
        timestamp: When this moment occurred
        similarity_score: Similarity to previous moments (for circular detection)
        is_rejected: Whether this was rejected by the next agent
        paper_urls: Papers cited in this moment
        embedding: Vector embedding for semantic search
    """
    moment_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique moment identifier")
    session_id: str = Field(description="Parent session identifier")
    round_number: int = Field(description="Debate round number", ge=1)
    agent_type: str = Field(description="Agent type (analyst, skeptic, synthesizer)")
    content: str = Field(description="Moment content", min_length=10)
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Moment timestamp")
    similarity_score: Optional[float] = Field(default=None, description="Similarity to previous moments", ge=0.0, le=1.0)
    is_rejected: bool = Field(default=False, description="Whether this was rejected")
    paper_urls: List[str] = Field(default_factory=list, description="Papers cited in this moment")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding for semantic search")


class MemoryPattern(BaseModel):
    """
    Recognized pattern across multiple debate sessions (Epic 4: Task 3).

    Patterns are extracted from past debates to improve future reasoning.

    Attributes:
        pattern_id: Unique identifier (UUID)
        workspace_id: Parent workspace identifier
        pattern_type: Type of pattern (claim structure, objection type, etc.)
        description: Human-readable pattern description
        frequency: How often this pattern appears
        last_seen: When this pattern was last observed
        example_sessions: Session IDs where this pattern appeared
        embedding: Vector embedding for semantic matching
        metadata: Additional pattern-specific data
    """
    pattern_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique pattern identifier")
    workspace_id: str = Field(description="Parent workspace identifier")
    pattern_type: MemoryPatternType = Field(description="Pattern type")
    description: str = Field(description="Pattern description", min_length=20, max_length=500)
    frequency: int = Field(default=1, description="Pattern frequency", ge=1)
    last_seen: datetime = Field(default_factory=datetime.utcnow, description="Last observation timestamp")
    example_sessions: List[str] = Field(default_factory=list, description="Example session IDs", max_length=10)
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Pattern-specific metadata")


class MemorySearchRequest(BaseModel):
    """
    Request for semantic memory search (Epic 4: Task 4).

    Attributes:
        query: Search query (claim, question, or keywords)
        workspace_id: Workspace to search within
        limit: Maximum results to return
        similarity_threshold: Minimum similarity score (0-1)
        pattern_types: Filter by pattern types
    """
    query: str = Field(description="Search query", min_length=3)
    workspace_id: str = Field(description="Workspace identifier")
    limit: int = Field(default=10, description="Maximum results", ge=1, le=100)
    similarity_threshold: float = Field(default=0.75, description="Minimum similarity", ge=0.0, le=1.0)
    pattern_types: Optional[List[MemoryPatternType]] = Field(default=None, description="Filter by pattern types")


class MemorySearchResult(BaseModel):
    """
    Result from semantic memory search (Epic 4: Task 4).

    Attributes:
        session_id: Matching session identifier
        moment: Matching debate moment
        similarity_score: Similarity to query (0-1)
        context: Additional context from the session
    """
    session_id: str = Field(description="Session identifier")
    moment: DebateMoment = Field(description="Matching debate moment")
    similarity_score: float = Field(description="Similarity score", ge=0.0, le=1.0)
    context: Optional[str] = Field(default=None, description="Additional context")


class MemorySearchResponse(BaseModel):
    """
    Response for semantic memory search (Epic 4: Task 4).

    Attributes:
        results: List of matching results
        query: Original search query
        total_found: Total number of matches
    """
    results: List[MemorySearchResult] = Field(description="Search results")
    query: str = Field(description="Original query")
    total_found: int = Field(description="Total matches", ge=0)

