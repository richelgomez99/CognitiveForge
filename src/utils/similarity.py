"""
Semantic similarity detection for Tier 1: Iterative Memory & Learning.

This module uses sentence-transformers to compute semantic similarity between claims
to detect circular arguments and prevent repeated rejected theses.

Tier 1: US2 - Similarity-based auto-rejection
"""

import os
import logging
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Similarity threshold for auto-rejection (configurable via env)
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.80"))

# Debug logging for similarity operations
DEBUG_MEMORY = os.getenv("DEBUG_MEMORY", "false").lower() == "true"

# =============================================================================
# Model Caching
# =============================================================================

_model_cache: Optional[SentenceTransformer] = None


def get_similarity_model() -> SentenceTransformer:
    """
    Load and cache the sentence-transformers model.
    
    Uses all-MiniLM-L6-v2 (22M params, 80MB) for fast CPU inference.
    Model is loaded once and reused for all similarity computations.
    
    Returns:
        SentenceTransformer model instance
    
    Performance:
        - CPU inference: 30-50ms for 2 sentences
        - GPU inference: 10-15ms for 2 sentences
    
    Tier 1: T014 - Model caching for reuse
    """
    global _model_cache
    
    if _model_cache is None:
        logger.info("Loading sentence-transformers model: all-MiniLM-L6-v2 (22M params, 80MB)")
        _model_cache = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ Similarity model loaded and cached")
    
    return _model_cache


# =============================================================================
# Similarity Computation
# =============================================================================

def compute_similarity(claim1: str, claim2: str) -> float:
    """
    Compute semantic similarity between two claims using sentence embeddings.
    
    Args:
        claim1: First claim text
        claim2: Second claim text
    
    Returns:
        Cosine similarity score (0.0-1.0)
        - 0.0 = completely different
        - 1.0 = identical/very similar
        - >0.80 = typically indicates circular argument (auto-reject threshold)
    
    Examples:
        >>> compute_similarity(
        ...     "Consciousness is an emergent property of biological neural networks",
        ...     "Consciousness emerges from biological neural substrates"
        ... )
        0.89  # HIGH similarity - likely circular argument
        
        >>> compute_similarity(
        ...     "Consciousness emerges from neural networks",
        ...     "Consciousness requires quantum effects in microtubules"
        ... )
        0.52  # LOW similarity - genuinely different claim
    
    Performance:
        - Target: <500ms (per spec)
        - Typical: 30-50ms on CPU, 10-15ms on GPU
    
    Tier 1: T012 - Core similarity computation
    """
    if not claim1 or not claim2:
        logger.warning("Empty claim provided to similarity computation")
        return 0.0
    
    # Strip whitespace
    claim1 = claim1.strip()
    claim2 = claim2.strip()
    
    # Load model (cached after first call)
    model = get_similarity_model()
    
    # Encode both claims into 768-dim vectors
    embeddings = model.encode([claim1, claim2])
    
    # Compute cosine similarity
    cos_sim = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )
    
    similarity_score = float(cos_sim)
    
    if DEBUG_MEMORY:
        logger.debug(f"Similarity computed: {similarity_score:.3f}")
        logger.debug(f"  Claim 1: {claim1[:100]}...")
        logger.debug(f"  Claim 2: {claim2[:100]}...")
    
    return similarity_score


def is_circular_argument(new_claim: str, rejected_claims: list[str]) -> tuple[bool, Optional[str], Optional[float]]:
    """
    Check if a new claim is too similar to previously rejected claims.
    
    Args:
        new_claim: The new thesis claim to check
        rejected_claims: List of previously rejected thesis claims from debate_memory
    
    Returns:
        Tuple of (is_circular, most_similar_claim, similarity_score)
        - is_circular: True if similarity exceeds SIMILARITY_THRESHOLD
        - most_similar_claim: The rejected claim with highest similarity (if circular)
        - similarity_score: The highest similarity score found
    
    Examples:
        >>> memory = ["Consciousness requires biological substrate", "IIT implies weak emergence"]
        >>> is_circular_argument("Consciousness needs biological substrate", memory)
        (True, "Consciousness requires biological substrate", 0.91)
        
        >>> is_circular_argument("Quantum effects enable free will", memory)
        (False, None, None)
    
    Tier 1: T047 - Similarity check for auto-rejection
    """
    if not rejected_claims:
        return (False, None, None)
    
    if DEBUG_MEMORY:
        logger.debug(f"Checking {len(rejected_claims)} rejected claims for circular argument")
    
    max_similarity = 0.0
    most_similar_claim = None
    
    for rejected_claim in rejected_claims:
        similarity = compute_similarity(new_claim, rejected_claim)
        
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_claim = rejected_claim
        
        if similarity > SIMILARITY_THRESHOLD:
            logger.info(f"🔄 Circular argument detected! Similarity: {similarity:.3f}")
            logger.info(f"   New claim: {new_claim[:100]}...")
            logger.info(f"   Rejected claim: {rejected_claim[:100]}...")
            return (True, rejected_claim, similarity)
    
    if DEBUG_MEMORY:
        logger.debug(f"Max similarity: {max_similarity:.3f} (threshold: {SIMILARITY_THRESHOLD})")
    
    return (False, None, None)

