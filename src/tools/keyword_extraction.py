"""
Keyword Extraction for Tier 1: Multi-Keyword Discovery Per Claim (US1).

This module uses LLM to extract 3-5 targeted multi-word search keywords from thesis claims,
enabling more relevant paper discovery than generic query-based searches.

Tier 1: T018-T021
"""

import os
import logging
from typing import List
from google import genai
from google.genai import types
from src.models import KeywordExtractionOutput

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Use lightweight model for keyword extraction (fast, cost-effective)
KEYWORD_MODEL = os.getenv("GEMINI_MODEL_KEYWORD", "gemini-2.5-flash-lite")

# =============================================================================
# Keyword Extraction
# =============================================================================

def extract_keywords(claim: str, reasoning: str) -> KeywordExtractionOutput:
    """
    Extract 3-5 targeted search keywords from a thesis claim using LLM.
    
    Uses Gemini Flash-Lite for fast, cost-effective keyword generation.
    Keywords are multi-word phrases (e.g., "neural correlates consciousness") optimized
    for academic paper search APIs (arXiv, Semantic Scholar).
    
    Args:
        claim: The main claim or hypothesis (e.g., "Consciousness is emergent from neural substrates")
        reasoning: Brief context/reasoning for the claim (first 200 chars recommended)
    
    Returns:
        KeywordExtractionOutput with:
            - keywords: List of 3-5 multi-word search phrases
            - reasoning: Explanation of keyword selection
    
    Fallback:
        If LLM fails, returns claim as single keyword (graceful degradation)
    
    Examples:
        >>> result = extract_keywords(
        ...     claim="Consciousness is emergent from biological neural substrates",
        ...     reasoning="IIT suggests consciousness arises from integrated information..."
        ... )
        >>> result.keywords
        ['integrated information theory consciousness', 'neural correlates consciousness biological',
         'emergent properties neural networks', 'substrate dependence cognition']
    
    Tier 1: T018-T021 - LLM-based keyword extraction with structured output
    """
    try:
        logger.info(f"🔍 Extracting keywords from claim: {claim[:100]}...")
        
        # Initialize Gemini client
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY not set")
            return _fallback_keywords(claim)
        
        client = genai.Client(api_key=api_key)
        
        # T019: Implement LLM prompt for keyword extraction
        prompt = f"""You are a research assistant extracting search keywords for academic paper discovery.

Task: Extract 3-5 targeted multi-word search keywords from the given thesis claim.

Requirements:
- Each keyword must be a multi-word phrase (2-4 words), NOT single words
- Keywords should be optimized for academic paper search (arXiv, Semantic Scholar)
- Include technical terminology and domain-specific concepts
- Focus on the core concepts and mechanisms mentioned in the claim
- Avoid overly generic terms (e.g., "neural networks" → "integrated information neural networks")

Thesis Claim:
{claim}

Reasoning Context (first 200 chars):
{reasoning[:200]}

Examples of GOOD keywords:
- "integrated information theory consciousness"
- "neural correlates consciousness biological"
- "emergent properties complex systems"
- "substrate dependence cognition"

Examples of BAD keywords (too generic, single words):
- "consciousness" (too generic)
- "brain" (single word, too broad)
- "networks" (single word)

Return a JSON object with:
{{
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
    "reasoning": "Brief explanation of why these keywords were chosen (20-300 chars)"
}}"""
        
        # T020: Add structured output parsing using Pydantic model
        response = client.models.generate_content(
            model=KEYWORD_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=KeywordExtractionOutput
            )
        )
        
        # Parse response
        extraction_result = KeywordExtractionOutput.model_validate_json(response.text)
        
        logger.info(f"✅ Extracted {len(extraction_result.keywords)} keywords: {extraction_result.keywords}")
        logger.debug(f"   Reasoning: {extraction_result.reasoning}")
        
        return extraction_result
    
    except Exception as e:
        logger.error(f"❌ Keyword extraction failed: {e}")
        logger.warning("Falling back to claim as single keyword")
        # T021: Add fallback logic
        return _fallback_keywords(claim)


def _fallback_keywords(claim: str) -> KeywordExtractionOutput:
    """
    Fallback keyword extraction when LLM fails.
    
    Returns the claim split into 3 keyword variants for graceful degradation.
    Ensures min_length=3 constraint is met.
    
    Args:
        claim: The original claim text
    
    Returns:
        KeywordExtractionOutput with 3 keyword variants
    
    Tier 1: T021 - Graceful degradation
    """
    logger.warning(f"Using fallback: claim as multiple keyword variants")
    
    # Use the first 100 chars of claim as primary keyword
    primary_keyword = claim[:100].strip()
    words = primary_keyword.split()
    
    # Handle very short claims (1-2 words) - pad to make multi-word
    if len(words) == 1:
        # Single-word claims: pad to make them multi-word
        keywords = [
            f"{words[0]} research",
            f"{words[0]} analysis",
            f"{words[0]} theory"
        ]
    elif len(words) == 2:
        # Two-word claims: use as-is and create variants
        keywords = [
            primary_keyword,
            f"{words[0]} research analysis",
            f"{words[1]} theory applications"
        ]
    elif len(words) >= 4:
        # Longer claims: split into meaningful chunks
        mid = len(words) // 2
        keywords = [
            primary_keyword,
            " ".join(words[:mid]),
            " ".join(words[mid:])
        ]
    else:
        # 3-word claims: use as-is and create variants
        keywords = [
            primary_keyword,
            f"{words[0]} {words[1]}",
            f"{words[1]} {words[2]}" if len(words) > 2 else f"{words[0]} research"
        ]
    
    return KeywordExtractionOutput(
        keywords=keywords,
        reasoning="Fallback: LLM keyword extraction failed, using claim variants as search queries"
    )


# =============================================================================
# Keyword Validation (Optional Enhancement)
# =============================================================================

def validate_keyword_quality(keywords: List[str]) -> tuple[bool, str]:
    """
    Validate that extracted keywords meet quality criteria.
    
    Checks:
        - All keywords are multi-word (2+ words)
        - No overly generic single words
        - Reasonable length (5-60 chars)
    
    Args:
        keywords: List of extracted keywords
    
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if all checks pass
        - error_message: Description of validation failure (if any)
    
    Tier 1: Optional quality gate
    """
    for keyword in keywords:
        # Check if multi-word
        word_count = len(keyword.split())
        if word_count < 2:
            return (False, f"Keyword '{keyword}' is not a multi-word phrase (only {word_count} word)")
        
        # Check length
        if len(keyword) < 5:
            return (False, f"Keyword '{keyword}' is too short ({len(keyword)} chars)")
        if len(keyword) > 60:
            return (False, f"Keyword '{keyword}' is too long ({len(keyword)} chars)")
    
    return (True, "")

