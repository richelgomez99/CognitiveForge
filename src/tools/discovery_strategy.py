"""
AI-Adaptive Discovery Strategy for Tier 1: Intelligent Paper Limits (US4).

This module uses LLM to intelligently determine how many papers to discover based on
claim complexity, reducing API calls for simple claims while ensuring thoroughness for complex ones.

Tier 1: T022-T025
"""

import os
import logging
from typing import List, Optional
from google import genai
from google.genai import types
from src.models import DiscoveryStrategy, Evidence

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Use lightweight model for strategy determination (fast, cost-effective)
STRATEGY_MODEL = os.getenv("GEMINI_MODEL_STRATEGY", "gemini-2.5-flash-lite")

# Default strategy if LLM fails
# DEMO MODE: Use higher limits for arXiv-only (no rate limits)
_arxiv_only = os.getenv("ARXIV_ONLY", "false").lower() == "true"
DEFAULT_STRATEGY = DiscoveryStrategy(
    initial_papers=6 if _arxiv_only else 3,  # Higher for arXiv-only demo
    follow_up_needed=False,
    follow_up_papers=0,
    reasoning=f"Default strategy: {'arXiv-only demo (6 papers)' if _arxiv_only else 'moderate initial discovery (3 papers)'}, no follow-up"
)

# =============================================================================
# Discovery Strategy Determination
# =============================================================================

def determine_discovery_strategy(claim: str, existing_evidence: List[Evidence]) -> DiscoveryStrategy:
    """
    Determine adaptive paper discovery limits based on claim complexity.
    
    Uses Gemini Flash-Lite to analyze claim and decide:
    - How many papers to fetch initially (2-10)
    - Whether follow-up discovery is needed (bool)
    - How many additional papers for follow-up (0-5)
    
    Strategy Logic:
        - Simple, well-established claims: 2-3 papers (fast)
        - Complex, contentious claims: 5-8 papers (thorough)
        - Novel, exploratory claims: 8-10 papers (comprehensive)
        - Follow-up triggered if initial paper quality < 0.6 avg relevance
    
    Args:
        claim: The thesis claim to evaluate
        existing_evidence: List of Evidence objects already available (for context)
    
    Returns:
        DiscoveryStrategy with adaptive paper limits and reasoning
    
    Fallback:
        If LLM fails, returns default strategy (3 initial, no follow-up)
    
    Examples:
        >>> # Simple, established claim
        >>> strategy = determine_discovery_strategy(
        ...     claim="Water is H2O",
        ...     existing_evidence=[]
        ... )
        >>> strategy.initial_papers
        2
        
        >>> # Complex, contentious claim
        >>> strategy = determine_discovery_strategy(
        ...     claim="Consciousness is an emergent property",
        ...     existing_evidence=[]
        ... )
        >>> strategy.initial_papers
        6
        >>> strategy.follow_up_needed
        True
    
    Tier 1: T022-T025 - AI-adaptive discovery limits
    """
    try:
        logger.info(f"🎯 Determining discovery strategy for claim: {claim[:100]}...")
        
        # Initialize Gemini client
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY not set")
            return _fallback_strategy()
        
        client = genai.Client(api_key=api_key)
        
        # Analyze existing evidence strength
        evidence_context = _build_evidence_context(existing_evidence)
        
        # T023: Implement LLM prompt for strategy determination
        prompt = f"""You are a research strategy assistant determining how many academic papers to discover.

Task: Analyze the claim complexity and determine an adaptive discovery strategy.

Claim to Analyze:
{claim}

Existing Evidence Context:
{evidence_context}

Guidelines:

1. SIMPLE & WELL-ESTABLISHED claims (3-5 papers):
   - Basic facts, widely accepted concepts
   - Strong existing evidence already available
   - Examples: "Water is H2O", "DNA encodes genetic information"

2. COMPLEX & CONTENTIOUS claims (6-10 papers):
   - Competing theories, active debate
   - Limited or weak existing evidence
   - Examples: "Consciousness is emergent", "Dark matter is self-interacting"

3. NOVEL & EXPLORATORY claims (8-10 papers):
   - Cutting-edge research, speculative
   - Very little existing evidence
   - Examples: "Quantum effects enable free will", "AI can exhibit genuine consciousness"

4. FOLLOW-UP DISCOVERY:
   - Needed if: Complex/novel claim AND existing evidence is weak/absent
   - Not needed if: Simple claim OR strong existing evidence
   - Follow-up papers: 2-5 (adds depth if initial discovery is insufficient)

Return JSON:
{{
    "initial_papers": 3,  // 2-10 based on complexity
    "follow_up_needed": false,  // true if complex/novel with weak evidence
    "follow_up_papers": 0,  // 0-5 if follow_up_needed
    "reasoning": "Brief explanation (20-300 chars) of why this strategy was chosen"
}}

Be conservative with paper counts to minimize API costs. Start lower unless claim is clearly complex/novel."""
        
        # T024: Add structured output parsing
        response = client.models.generate_content(
            model=STRATEGY_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DiscoveryStrategy
            )
        )
        
        # Parse response
        strategy = DiscoveryStrategy.model_validate_json(response.text)
        
        # T076: Enhanced logging with complexity reasoning
        complexity_level = _infer_complexity_level(strategy.initial_papers)
        logger.info(f"✅ Strategy determined: Claim complexity: {complexity_level} → {strategy.initial_papers} initial papers")
        logger.info(f"   Follow-up needed: {strategy.follow_up_needed}")
        if strategy.follow_up_needed:
            logger.info(f"   Follow-up papers: {strategy.follow_up_papers}")
        logger.info(f"   Reasoning: {strategy.reasoning}")
        
        return strategy
    
    except Exception as e:
        logger.error(f"❌ Strategy determination failed: {e}")
        logger.warning("Falling back to default strategy")
        # T025: Add fallback
        return _fallback_strategy()


def _build_evidence_context(existing_evidence: List[Evidence]) -> str:
    """
    Build context string from existing evidence for strategy determination.
    
    Args:
        existing_evidence: List of Evidence objects
    
    Returns:
        Human-readable summary of existing evidence strength
    
    Tier 1: Helper function
    """
    if not existing_evidence:
        return "No existing evidence available."
    
    # Summarize evidence
    evidence_count = len(existing_evidence)
    avg_relevance = sum(e.relevance_score or 0.5 for e in existing_evidence) / evidence_count
    
    sources = [e.source_url for e in existing_evidence]
    unique_sources = len(set(sources))
    
    return f"""Existing Evidence:
- Count: {evidence_count} pieces of evidence
- Average Relevance: {avg_relevance:.2f} / 1.0
- Unique Sources: {unique_sources}
- Evidence Strength: {"Strong" if avg_relevance > 0.7 and evidence_count >= 3 else "Weak"}"""


def _fallback_strategy() -> DiscoveryStrategy:
    """
    Fallback strategy when LLM fails.
    
    Returns moderate default strategy (3 initial papers, no follow-up).
    
    Returns:
        DiscoveryStrategy with default values
    
    Tier 1: T025 - Graceful degradation
    """
    logger.warning("Using fallback default strategy")
    return DEFAULT_STRATEGY


# =============================================================================
# Strategy Validation (Optional Enhancement)
# =============================================================================

def should_trigger_follow_up(strategy: DiscoveryStrategy, discovered_papers: list, avg_relevance: float) -> bool:
    """
    Determine if follow-up discovery should be triggered based on initial results.
    
    Follow-up logic:
        - If strategy.follow_up_needed AND avg_relevance < 0.6, trigger follow-up
        - Fetch strategy.follow_up_papers additional papers
    
    Args:
        strategy: The DiscoveryStrategy from LLM
        discovered_papers: List of papers from initial discovery
        avg_relevance: Average relevance score of discovered papers (0.0-1.0)
    
    Returns:
        True if follow-up should be triggered, False otherwise
    
    Tier 1: T033 - Follow-up discovery logic
    """
    if not strategy.follow_up_needed:
        logger.debug("Follow-up not needed (strategy indicates simple claim)")
        return False
    
    if len(discovered_papers) == 0:
        logger.warning("No papers discovered in initial search, skipping follow-up")
        return False
    
    if avg_relevance < 0.6:
        logger.info(f"✅ Triggering follow-up discovery (avg relevance {avg_relevance:.2f} < 0.6 threshold)")
        return True
    
    logger.debug(f"Follow-up not triggered (avg relevance {avg_relevance:.2f} >= 0.6)")
    return False


def _infer_complexity_level(initial_papers: int) -> str:
    """
    Infer human-readable complexity level from paper count.
    
    Tier 1: T076 - Enhanced logging for strategy reasoning.
    
    Args:
        initial_papers: Number of initial papers recommended
    
    Returns:
        Complexity level string (LOW, MEDIUM, HIGH)
    
    Examples:
        >>> _infer_complexity_level(2)
        'LOW'
        >>> _infer_complexity_level(7)
        'HIGH'
    """
    if initial_papers <= 3:
        return "LOW"
    elif initial_papers <= 6:
        return "MEDIUM"
    else:
        return "HIGH"

