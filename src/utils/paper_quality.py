"""
Paper quality assessment utilities for Tier 2 (US4).

This module provides functions to:
1. Fetch citation counts from Semantic Scholar
2. Calculate quality scores based on citations + venue + peer-review status
3. Assign star ratings (1-5 stars) for visual quality indicators

Quality Score Formula:
    quality_score = (citation_score * 0.6) + (venue_score * 0.3) + (peer_review_bonus * 0.1)
    
Star Rating Thresholds:
    5 stars: Quality score >= 80 (highly cited + top venue + peer-reviewed)
    4 stars: Quality score >= 60
    3 stars: Quality score >= 40
    2 stars: Quality score >= 20
    1 star:  Quality score < 20
"""

import logging
import os
import time
from typing import Optional, Dict, Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models import PaperQualityScore


logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# VENUE IMPACT FACTORS (Top Journals/Conferences)
# ────────────────────────────────────────────────────────────────

VENUE_IMPACT_FACTORS = {
    # Top-tier journals (Impact Factor > 30)
    "Nature": 49.962,
    "Science": 47.728,
    "Cell": 66.850,
    "Nature Reviews Neuroscience": 38.2,
    "Nature Neuroscience": 24.884,
    "Nature Reviews Molecular Cell Biology": 94.444,
    
    # High-tier journals (Impact Factor 10-30)
    "PLOS Computational Biology": 4.7,
    "Neural Computation": 2.783,
    "Neuron": 16.2,
    "Trends in Cognitive Sciences": 21.5,
    "Annual Review of Neuroscience": 13.4,
    
    # Mid-tier journals (Impact Factor 5-10)
    "Journal of Neuroscience": 5.3,
    "Frontiers in Neuroscience": 4.3,
    "Neural Networks": 7.8,
    
    # Top-tier CS conferences (equiv. to high-impact journals)
    "NeurIPS": 25.0,  # Equivalent impact
    "ICML": 25.0,
    "ICLR": 25.0,
    "CVPR": 20.0,
    "ICCV": 20.0,
    "AAAI": 15.0,
    "IJCAI": 15.0,
    
    # Preprints (no peer review)
    "arXiv": 0.0,
    "bioRxiv": 0.0,
    "medRxiv": 0.0,
}


def get_venue_impact_factor(venue: Optional[str]) -> float:
    """
    Get venue impact factor from predefined list.
    
    Args:
        venue: Venue name (journal or conference)
    
    Returns:
        Impact factor (0-100), default 5.0 for unknown venues
    """
    if not venue:
        return 5.0
    
    # Normalize venue name (case-insensitive, strip whitespace)
    venue_normalized = venue.strip().lower()
    
    # Check for exact match
    for known_venue, impact in VENUE_IMPACT_FACTORS.items():
        if known_venue.lower() == venue_normalized:
            return impact
    
    # Check for partial match (e.g., "Nature Reviews" matches "Nature Reviews Neuroscience")
    for known_venue, impact in VENUE_IMPACT_FACTORS.items():
        if known_venue.lower() in venue_normalized or venue_normalized in known_venue.lower():
            return impact
    
    # Unknown venue: default 5.0
    logger.debug(f"Unknown venue '{venue}', using default impact factor 5.0")
    return 5.0


def is_peer_reviewed(venue: Optional[str]) -> bool:
    """
    Determine if venue is peer-reviewed.
    
    Args:
        venue: Venue name (journal or conference)
    
    Returns:
        True if peer-reviewed, False if preprint
    """
    if not venue:
        return False
    
    venue_lower = venue.lower()
    
    # Preprints (not peer-reviewed)
    preprint_indicators = ["arxiv", "biorxiv", "medrxiv", "preprint"]
    if any(indicator in venue_lower for indicator in preprint_indicators):
        return False
    
    # Everything else assumed peer-reviewed
    return True


# ────────────────────────────────────────────────────────────────
# SEMANTIC SCHOLAR CITATION FETCHING
# ────────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True
)
def fetch_citation_count(paper_url: str) -> Optional[int]:
    """
    Fetch citation count from Semantic Scholar API.
    
    This function uses the Semantic Scholar API to retrieve citation counts.
    Rate limit: 100 requests/5 minutes (without API key), 5000 requests/5 minutes (with API key).
    
    Args:
        paper_url: URL of the paper (arXiv or Semantic Scholar)
    
    Returns:
        Citation count (int), or None if unavailable
    """
    # Extract paper ID from URL
    paper_id = _extract_paper_id(paper_url)
    if not paper_id:
        logger.warning(f"Could not extract paper ID from URL: {paper_url}")
        return None
    
    # Semantic Scholar API endpoint
    api_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
    
    # Request only citation count field
    params = {"fields": "citationCount"}
    
    # Add API key if available (increases rate limit)
    headers = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key and api_key != "your_semantic_scholar_api_key_here":
        headers["x-api-key"] = api_key
    
    try:
        logger.debug(f"Fetching citation count for {paper_id}...")
        response = requests.get(api_url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            citation_count = data.get("citationCount", 0)
            logger.debug(f"Citation count for {paper_id}: {citation_count}")
            return citation_count
        
        elif response.status_code == 404:
            logger.debug(f"Paper not found in Semantic Scholar: {paper_id}")
            return 0
        
        elif response.status_code == 429:
            logger.warning("Semantic Scholar rate limit hit. Retrying...")
            time.sleep(5)  # Wait before retry
            raise Exception("Rate limit exceeded")  # Trigger retry
        
        else:
            logger.warning(f"Semantic Scholar API error {response.status_code}: {response.text}")
            return None
    
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout fetching citation count for {paper_url}")
        return None
    
    except Exception as e:
        logger.warning(f"Error fetching citation count for {paper_url}: {e}")
        return None


def _extract_paper_id(paper_url: str) -> Optional[str]:
    """
    Extract paper ID from URL for Semantic Scholar API.
    
    Supports:
    - arXiv URLs: http://arxiv.org/abs/2301.12345 → arXiv:2301.12345
    - Semantic Scholar URLs: https://www.semanticscholar.org/paper/abc123 → abc123
    - DOI URLs: https://doi.org/10.1234/xyz → DOI:10.1234/xyz
    
    Args:
        paper_url: Paper URL
    
    Returns:
        Paper ID for Semantic Scholar API, or None if unsupported
    """
    url_lower = paper_url.lower()
    
    # arXiv
    if "arxiv.org" in url_lower:
        # Extract paper ID from arXiv URL
        # Example: http://arxiv.org/abs/2301.12345 → arXiv:2301.12345
        if "/abs/" in url_lower:
            paper_id = paper_url.split("/abs/")[-1]
            # Remove version suffix (v1, v2, etc.)
            paper_id = paper_id.split("v")[0] if "v" in paper_id else paper_id
            return f"arXiv:{paper_id}"
    
    # Semantic Scholar
    elif "semanticscholar.org" in url_lower:
        # Extract paper ID from Semantic Scholar URL
        # Example: https://www.semanticscholar.org/paper/abc123 → abc123
        parts = paper_url.split("/paper/")
        if len(parts) > 1:
            return parts[-1]
    
    # DOI
    elif "doi.org" in url_lower:
        # Extract DOI
        # Example: https://doi.org/10.1234/xyz → DOI:10.1234/xyz
        doi = paper_url.split("doi.org/")[-1]
        return f"DOI:{doi}"
    
    # Unsupported URL
    return None


# ────────────────────────────────────────────────────────────────
# QUALITY SCORE CALCULATION
# ────────────────────────────────────────────────────────────────

def calculate_quality_score(
    citation_count: int,
    venue_impact_factor: float,
    is_peer_reviewed_flag: bool
) -> float:
    """
    Calculate overall quality score (0-100) from citations, venue, and peer-review status.
    
    Formula:
        quality_score = (citation_score * 0.6) + (venue_score * 0.3) + (peer_review_bonus * 0.1)
    
    Citation Score (0-100):
        - 0-10 citations: 0-20
        - 10-50 citations: 20-40
        - 50-100 citations: 40-60
        - 100-500 citations: 60-80
        - 500+ citations: 80-100
    
    Venue Score (0-100):
        - Top-tier (IF > 30): 80-100
        - High-tier (IF 10-30): 60-80
        - Mid-tier (IF 5-10): 40-60
        - Low-tier (IF < 5): 20-40
        - Preprint (IF = 0): 0
    
    Peer-Review Bonus (0-10):
        - Peer-reviewed: +10
        - Preprint: +0
    
    Args:
        citation_count: Number of citations
        venue_impact_factor: Venue impact factor (0-100)
        is_peer_reviewed_flag: True if peer-reviewed
    
    Returns:
        Quality score (0-100)
    """
    # Citation score (0-100)
    if citation_count >= 500:
        citation_score = 100.0
    elif citation_count >= 100:
        citation_score = 60.0 + (citation_count - 100) / 400 * 20  # 60-80
    elif citation_count >= 50:
        citation_score = 40.0 + (citation_count - 50) / 50 * 20  # 40-60
    elif citation_count >= 10:
        citation_score = 20.0 + (citation_count - 10) / 40 * 20  # 20-40
    else:
        citation_score = citation_count / 10 * 20  # 0-20
    
    # Venue score (0-100)
    if venue_impact_factor >= 30:
        venue_score = 80.0 + min((venue_impact_factor - 30) / 70 * 20, 20)  # 80-100
    elif venue_impact_factor >= 10:
        venue_score = 60.0 + (venue_impact_factor - 10) / 20 * 20  # 60-80
    elif venue_impact_factor >= 5:
        venue_score = 40.0 + (venue_impact_factor - 5) / 5 * 20  # 40-60
    elif venue_impact_factor > 0:
        venue_score = venue_impact_factor / 5 * 40  # 0-40
    else:
        venue_score = 0.0  # Preprint
    
    # Peer-review bonus (0-10)
    peer_review_bonus = 10.0 if is_peer_reviewed_flag else 0.0
    
    # Weighted average
    quality_score = (citation_score * 0.6) + (venue_score * 0.3) + (peer_review_bonus * 0.1)
    
    # Clamp to [0, 100]
    return max(0.0, min(100.0, quality_score))


def get_star_rating(quality_score: float) -> int:
    """
    Convert quality score to star rating (1-5 stars).
    
    Thresholds:
        5 stars: Quality score >= 80 (highly cited + top venue + peer-reviewed)
        4 stars: Quality score >= 60
        3 stars: Quality score >= 40
        2 stars: Quality score >= 20
        1 star:  Quality score < 20
    
    Args:
        quality_score: Quality score (0-100)
    
    Returns:
        Star rating (1-5)
    """
    if quality_score >= 80:
        return 5
    elif quality_score >= 60:
        return 4
    elif quality_score >= 40:
        return 3
    elif quality_score >= 20:
        return 2
    else:
        return 1


# ────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTION
# ────────────────────────────────────────────────────────────────

def assess_paper_quality(
    paper_url: str,
    venue: Optional[str] = None,
    cached_citation_count: Optional[int] = None
) -> PaperQualityScore:
    """
    Assess paper quality by fetching citation count and calculating scores.
    
    This is a convenience function that:
    1. Fetches citation count from Semantic Scholar (if not cached)
    2. Determines venue impact factor
    3. Determines peer-review status
    4. Calculates quality score
    5. Assigns star rating
    
    Args:
        paper_url: URL of the paper
        venue: Optional venue name (journal or conference)
        cached_citation_count: Optional cached citation count (skip API call)
    
    Returns:
        PaperQualityScore object with all fields populated
    """
    # Fetch citation count (or use cached)
    if cached_citation_count is not None:
        citation_count = cached_citation_count
    else:
        citation_count = fetch_citation_count(paper_url) or 0
    
    # Get venue impact factor
    venue_impact_factor = get_venue_impact_factor(venue)
    
    # Determine peer-review status
    is_peer_reviewed_flag = is_peer_reviewed(venue)
    
    # Calculate quality score
    quality_score = calculate_quality_score(
        citation_count,
        venue_impact_factor,
        is_peer_reviewed_flag
    )
    
    # Assign star rating
    star_rating = get_star_rating(quality_score)
    
    return PaperQualityScore(
        citation_count=citation_count,
        venue_impact_factor=venue_impact_factor,
        is_peer_reviewed=is_peer_reviewed_flag,
        star_rating=star_rating,
        quality_score=quality_score
    )

