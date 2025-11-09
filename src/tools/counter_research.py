"""
Counter-research tools for Skeptic agent (Tier 1 US3).

This module enables the Skeptic to actively search for papers that contradict
thesis claims, grounding critiques in counter-evidence.
"""

import os
import logging
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.models import PaperMetadata, ConflictingEvidence
from src.tools.paper_discovery import discover_papers_for_keywords_parallel

load_dotenv()
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================

class CounterQueryOutput(BaseModel):
    """
    Structured output for counter-query generation (Tier 1: T061).
    """
    counter_queries: List[str] = Field(
        description="2-3 search queries designed to find contradicting evidence",
        min_length=2,
        max_length=3
    )
    reasoning: str = Field(
        description="Explanation of why these queries target contradictions",
        min_length=20
    )
    
    @classmethod
    def field_validator(cls, v: List[str]) -> List[str]:
        """Ensure queries include negation/challenge terms"""
        for query in v:
            if len(query) < 5:
                raise ValueError(f"Query too short: '{query}'")
        return v


# =============================================================================
# Counter-Query Generation (T059-T061)
# =============================================================================

def generate_counter_queries(thesis_claim: str, reasoning: str) -> List[str]:
    """
    Generate 2-3 targeted search queries to find contradicting evidence.
    
    Tier 1: T059-T061 - LLM-based counter-query generation for Skeptic.
    
    Args:
        thesis_claim: The claim to find contradictions for
        reasoning: The supporting reasoning for the claim
    
    Returns:
        List of 2-3 search query strings targeting contradictions
    
    Example:
        >>> queries = generate_counter_queries(
        ...     "Consciousness is purely computational",
        ...     "Information processing theory suggests..."
        ... )
        >>> len(queries)
        3
        >>> "limitations" in queries[0].lower()
        True
    
    Tier 1: Uses negation terms ('limitations', 'challenges', 'refutation') to
    find papers that contradict rather than support the thesis.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY not set, using fallback counter-queries")
        return _fallback_counter_queries(thesis_claim)
    
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL_SKEPTIC", "gemini-2.5-flash-lite"),
        google_api_key=api_key,
        temperature=0.2
    )
    
    parser = PydanticOutputParser(pydantic_object=CounterQueryOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a critical research assistant tasked with finding evidence that CONTRADICTS or CHALLENGES a given thesis claim.

Your goal is to generate 2-3 targeted search queries that will find papers presenting:
- Contradictory evidence
- Alternative theories
- Limitations and challenges
- Counter-arguments

Key strategies:
1. Use negation terms: "limitations of", "challenges to", "refutation of", "critique of"
2. Search for alternative frameworks: "substrate-dependent", "non-computational"
3. Look for boundary conditions: "when X fails", "constraints of"
4. Target specific claims in the reasoning that can be challenged

{format_instructions}

Example of GOOD counter-queries:
- "limitations of computational theory of consciousness"
- "substrate-dependent consciousness biological necessity"
- "information processing insufficient for consciousness"

Example of BAD counter-queries (too supportive):
- "computational consciousness research"
- "information processing mind"
"""),
        ("human", "Thesis Claim: {claim}\n\nSupporting Reasoning: {reasoning}\n\nGenerate 2-3 search queries to find CONTRADICTING evidence.")
    ]).partial(format_instructions=parser.get_format_instructions())
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({"claim": thesis_claim, "reasoning": reasoning})
        logger.info(f"✅ Generated {len(result.counter_queries)} counter-queries")
        for i, query in enumerate(result.counter_queries, 1):
            logger.info(f"   {i}. {query}")
        return result.counter_queries
    except Exception as e:
        logger.error(f"❌ Counter-query generation failed: {e}")
        logger.warning("Falling back to heuristic counter-queries")
        return _fallback_counter_queries(thesis_claim)


def _fallback_counter_queries(claim: str) -> List[str]:
    """
    Fallback counter-query generation when LLM fails.
    
    Tier 1: T061 - Graceful degradation for counter-research.
    """
    logger.warning(f"Using fallback counter-queries for: {claim[:50]}...")
    
    # Extract key concepts (simple heuristic: first 3-5 words)
    words = claim.split()[:5]
    key_concept = " ".join(words)
    
    # Generate queries with negation terms
    fallback_queries = [
        f"limitations of {key_concept}",
        f"challenges to {key_concept}",
        f"critique {key_concept}"
    ]
    
    return fallback_queries


# =============================================================================
# Counter-Evidence Discovery (T062-T065)
# =============================================================================

async def discover_counter_evidence(
    counter_queries: List[str],
    claim_id: str,
    max_papers_per_query: int = 2
) -> List[PaperMetadata]:
    """
    Discover papers that contradict the thesis using counter-queries.
    
    Tier 1: T062-T065 - Active counter-research for Skeptic agent.
    
    Args:
        counter_queries: List of 2-3 search queries targeting contradictions
        claim_id: UUID of the claim for Neo4j tagging
        max_papers_per_query: Papers to fetch per query (default: 2, total: 4-6)
    
    Returns:
        List of PaperMetadata objects (4-6 papers total)
    
    Example:
        >>> queries = ["limitations of neural networks", "challenges to deep learning"]
        >>> papers = await discover_counter_evidence(queries, claim_id="abc-123")
        >>> len(papers)
        4
        >>> papers[0].discovered_by
        'skeptic_counter'
    
    Tier 1: Uses parallel discovery with deduplication (T064).
    Tags papers with 'skeptic_counter' for tracking (T063).
    Limits to 4-6 total papers to prevent explosion (T064).
    """
    logger.info(f"🔍 Discovering counter-evidence for {len(counter_queries)} queries...")
    logger.info(f"   Max papers per query: {max_papers_per_query} (total: {len(counter_queries) * max_papers_per_query * 2})")
    
    # T062-T063: Use existing discover_papers_for_keywords_parallel
    # This will tag papers with discovered_by="skeptic_counter" when we add them to Neo4j
    unique_papers, url_to_keywords = await discover_papers_for_keywords_parallel(
        keywords=counter_queries,
        max_results_per_keyword=max_papers_per_query,
        claim_id=claim_id
    )
    
    # T064: Limit total counter-papers to 6 (in case of many duplicates)
    if len(unique_papers) > 6:
        logger.info(f"   Limiting counter-papers from {len(unique_papers)} to 6")
        unique_papers = unique_papers[:6]
    
    logger.info(f"✅ Discovered {len(unique_papers)} counter-papers")
    
    return unique_papers


def papers_to_conflicting_evidence(
    papers: List[PaperMetadata],
    discovered_by: str = "skeptic_counter"
) -> List[ConflictingEvidence]:
    """
    Convert PaperMetadata to ConflictingEvidence objects for Antithesis.
    
    Tier 1: T068 - Helper function for Skeptic integration.
    
    Args:
        papers: List of PaperMetadata from counter-research
        discovered_by: Discovery method tag (default: "skeptic_counter")
    
    Returns:
        List of ConflictingEvidence objects with snippets from abstracts
    
    Example:
        >>> papers = [PaperMetadata(...)]
        >>> evidence = papers_to_conflicting_evidence(papers)
        >>> evidence[0].discovered_by
        'skeptic_counter'
        >>> len(evidence[0].snippet)
        <= 300
    """
    conflicting_evidence = []
    
    for paper in papers:
        # Extract snippet from abstract (truncate to 300 chars per ConflictingEvidence validation)
        if paper.abstract:
            snippet = paper.abstract[:300]
        else:
            snippet = paper.title[:300]
        
        # Set relevance score (default 0.8 for counter-papers, as they're specifically searched)
        # Higher score than regular papers since they're targeted contradictions
        relevance_score = 0.8
        
        evidence = ConflictingEvidence(
            source_url=paper.url,
            snippet=snippet,
            relevance_score=relevance_score,
            discovered_by=discovered_by
        )
        
        conflicting_evidence.append(evidence)
    
    logger.info(f"   Converted {len(papers)} papers to ConflictingEvidence objects")
    
    return conflicting_evidence


# =============================================================================
# Deduplication Helper (T065)
# =============================================================================

def filter_already_in_kg(
    papers: List[PaperMetadata],
    existing_papers_in_kg: List[str]
) -> List[PaperMetadata]:
    """
    Filter out papers that are already in the knowledge graph.
    
    Tier 1: T065 - Avoid re-adding Analyst's papers as counter-evidence.
    
    Args:
        papers: List of newly discovered counter-papers
        existing_papers_in_kg: List of URLs already in Neo4j
    
    Returns:
        Filtered list of papers (only new ones)
    
    Example:
        >>> papers = [PaperMetadata(url="https://arxiv.org/abs/1234"), ...]
        >>> existing = ["https://arxiv.org/abs/1234"]
        >>> filtered = filter_already_in_kg(papers, existing)
        >>> len(filtered)
        0  # Paper was already in KG
    """
    filtered = [p for p in papers if p.url not in existing_papers_in_kg]
    
    if len(filtered) < len(papers):
        logger.info(f"   Filtered {len(papers) - len(filtered)} papers already in KG")
    
    return filtered

