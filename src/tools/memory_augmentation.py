"""
Epic 4: Task 3 - Memory Augmentation Module

Provides cross-session learning capabilities:
1. Pattern recognition - Identify recurring claim structures, objection types
2. Memory compression - Summarize old sessions to reduce storage
3. Context injection - Retrieve relevant past insights for new debates
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import numpy as np

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import models and storage
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import DebateMoment, MemoryPattern, MemoryPatternType
from tools import memory_store


# Global embedding model (lazy loaded)
_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    Get or initialize the sentence embedding model.

    Uses all-MiniLM-L6-v2 by default (configurable via EMBEDDING_MODEL env var).

    Returns:
        SentenceTransformer model for generating embeddings
    """
    global _embedding_model

    if _embedding_model is None:
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        logger.info(f"Loading embedding model: {model_name}")
        _embedding_model = SentenceTransformer(model_name)
        logger.info("✅ Embedding model loaded successfully")

    return _embedding_model


def generate_embedding(text: str) -> List[float]:
    """
    Generate vector embedding for text.

    Args:
        text: Text to embed

    Returns:
        List of floats representing the embedding vector
    """
    model = get_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


# =============================================================================
# Pattern Recognition
# =============================================================================

def recognize_claim_patterns(
    moments: List[DebateMoment],
    workspace_id: str
) -> List[MemoryPattern]:
    """
    Recognize recurring claim structures from debate moments.

    Analyzes analyst moments to identify common claim patterns:
    - Claims that get rejected consistently
    - Claims that need specific types of evidence
    - Claims with similar structure/reasoning

    Args:
        moments: List of DebateMoment objects to analyze
        workspace_id: Workspace identifier for pattern storage

    Returns:
        List of recognized MemoryPattern objects
    """
    patterns = []

    # Filter analyst moments
    analyst_moments = [m for m in moments if m.agent_type == "analyst"]

    if len(analyst_moments) < 3:
        logger.info("Not enough analyst moments for pattern recognition (need ≥3)")
        return patterns

    # Group rejected vs. accepted claims
    rejected_moments = [m for m in analyst_moments if m.is_rejected]
    accepted_moments = [m for m in analyst_moments if not m.is_rejected]

    # Pattern 1: High rejection rate on certain claim types
    if len(rejected_moments) >= 2:
        # Find common characteristics in rejected claims
        rejected_texts = [m.content[:200] for m in rejected_moments]  # First 200 chars

        pattern = MemoryPattern(
            workspace_id=workspace_id,
            pattern_type=MemoryPatternType.CLAIM_STRUCTURE,
            description=f"Pattern: {len(rejected_moments)} claims rejected. Common structure needs stronger empirical support.",
            frequency=len(rejected_moments),
            example_sessions=[m.session_id for m in rejected_moments[:5]],  # Max 5 examples
            metadata={
                "rejection_rate": len(rejected_moments) / len(analyst_moments),
                "sample_claims": rejected_texts[:3]
            }
        )

        # Generate embedding from concatenated rejected claims
        combined_text = " ".join(rejected_texts)
        pattern.embedding = generate_embedding(combined_text)

        patterns.append(pattern)
        logger.info(f"Recognized pattern: High rejection rate ({pattern.metadata['rejection_rate']:.2%})")

    # Pattern 2: Successful claim structures
    if len(accepted_moments) >= 2:
        accepted_texts = [m.content[:200] for m in accepted_moments]

        pattern = MemoryPattern(
            workspace_id=workspace_id,
            pattern_type=MemoryPatternType.CLAIM_STRUCTURE,
            description=f"Pattern: {len(accepted_moments)} claims accepted. Strong empirical evidence pattern.",
            frequency=len(accepted_moments),
            example_sessions=[m.session_id for m in accepted_moments[:5]],
            metadata={
                "acceptance_rate": len(accepted_moments) / len(analyst_moments),
                "sample_claims": accepted_texts[:3]
            }
        )

        combined_text = " ".join(accepted_texts)
        pattern.embedding = generate_embedding(combined_text)

        patterns.append(pattern)
        logger.info(f"Recognized pattern: High acceptance rate ({pattern.metadata['acceptance_rate']:.2%})")

    return patterns


def recognize_skeptic_patterns(
    moments: List[DebateMoment],
    workspace_id: str
) -> List[MemoryPattern]:
    """
    Recognize recurring skeptic objection types.

    Analyzes skeptic moments to identify common objection patterns:
    - Frequent objection types (evidence quality, logical consistency, etc.)
    - Objections that lead to debate resolution
    - Objections that cause impasses

    Args:
        moments: List of DebateMoment objects to analyze
        workspace_id: Workspace identifier for pattern storage

    Returns:
        List of recognized MemoryPattern objects
    """
    patterns = []

    # Filter skeptic moments
    skeptic_moments = [m for m in moments if m.agent_type == "skeptic"]

    if len(skeptic_moments) < 3:
        logger.info("Not enough skeptic moments for pattern recognition (need ≥3)")
        return patterns

    # Analyze objection content
    objection_texts = [m.content[:200] for m in skeptic_moments]

    pattern = MemoryPattern(
        workspace_id=workspace_id,
        pattern_type=MemoryPatternType.SKEPTIC_OBJECTION,
        description=f"Pattern: {len(skeptic_moments)} skeptic objections. Common concerns about evidence quality and logical consistency.",
        frequency=len(skeptic_moments),
        example_sessions=[m.session_id for m in skeptic_moments[:5]],
        metadata={
            "objection_count": len(skeptic_moments),
            "sample_objections": objection_texts[:3]
        }
    )

    # Generate embedding
    combined_text = " ".join(objection_texts)
    pattern.embedding = generate_embedding(combined_text)

    patterns.append(pattern)
    logger.info(f"Recognized pattern: {len(skeptic_moments)} skeptic objections")

    return patterns


def recognize_debate_convergence_patterns(
    moments: List[DebateMoment],
    workspace_id: str
) -> List[MemoryPattern]:
    """
    Recognize patterns in how debates converge or reach impasses.

    Analyzes debate flow to identify:
    - Debates that converged quickly (< 3 rounds)
    - Debates that reached impasses (circular arguments)
    - Debates with high similarity scores (natural termination)

    Args:
        moments: List of DebateMoment objects to analyze
        workspace_id: Workspace identifier for pattern storage

    Returns:
        List of recognized MemoryPattern objects
    """
    patterns = []

    # Group by session
    sessions = {}
    for moment in moments:
        if moment.session_id not in sessions:
            sessions[moment.session_id] = []
        sessions[moment.session_id].append(moment)

    # Analyze convergence patterns
    quick_convergence = []
    circular_impasse = []

    for session_id, session_moments in sessions.items():
        max_round = max(m.round_number for m in session_moments)
        high_similarity = any(m.similarity_score and m.similarity_score > 0.8 for m in session_moments)

        if max_round <= 2:
            quick_convergence.append(session_id)
        elif high_similarity:
            circular_impasse.append(session_id)

    # Create patterns
    if quick_convergence:
        pattern = MemoryPattern(
            workspace_id=workspace_id,
            pattern_type=MemoryPatternType.CONVERGENCE_PATTERN,
            description=f"Pattern: {len(quick_convergence)} debates converged quickly (≤2 rounds). Strong initial evidence led to rapid consensus.",
            frequency=len(quick_convergence),
            example_sessions=quick_convergence[:5],
            metadata={
                "convergence_type": "quick",
                "avg_rounds": 1.5
            }
        )
        patterns.append(pattern)
        logger.info(f"Recognized pattern: {len(quick_convergence)} quick convergences")

    if circular_impasse:
        pattern = MemoryPattern(
            workspace_id=workspace_id,
            pattern_type=MemoryPatternType.CONVERGENCE_PATTERN,
            description=f"Pattern: {len(circular_impasse)} debates reached circular impasses. High similarity scores (>0.8) indicate repeated arguments.",
            frequency=len(circular_impasse),
            example_sessions=circular_impasse[:5],
            metadata={
                "convergence_type": "circular_impasse",
                "similarity_threshold": 0.8
            }
        )
        patterns.append(pattern)
        logger.info(f"Recognized pattern: {len(circular_impasse)} circular impasses")

    return patterns


def recognize_all_patterns(workspace_id: str, lookback_days: int = 30) -> List[MemoryPattern]:
    """
    Recognize all patterns from recent debate history.

    Analyzes all debate moments from the past N days to identify patterns.

    Args:
        workspace_id: Workspace identifier
        lookback_days: Number of days to look back (default: 30)

    Returns:
        List of all recognized MemoryPattern objects
    """
    logger.info(f"Recognizing patterns for workspace {workspace_id} (lookback: {lookback_days} days)")

    # Get all sessions in workspace from the past N days
    cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)

    # Get sessions (we'll need to filter by date)
    sessions, _ = memory_store.list_sessions(
        workspace_id=workspace_id,
        status=None,  # All statuses
        page=1,
        page_size=1000  # Get a large batch
    )

    # Filter by date
    recent_sessions = [s for s in sessions if s.created_at >= cutoff_date]

    if not recent_sessions:
        logger.info("No recent sessions found for pattern recognition")
        return []

    logger.info(f"Analyzing {len(recent_sessions)} recent sessions")

    # Get all moments from these sessions
    all_moments = []
    for session in recent_sessions:
        moments = memory_store.get_session_moments(session.session_id)
        all_moments.extend(moments)

    if not all_moments:
        logger.info("No debate moments found")
        return []

    logger.info(f"Analyzing {len(all_moments)} debate moments")

    # Recognize different pattern types
    patterns = []

    patterns.extend(recognize_claim_patterns(all_moments, workspace_id))
    patterns.extend(recognize_skeptic_patterns(all_moments, workspace_id))
    patterns.extend(recognize_debate_convergence_patterns(all_moments, workspace_id))

    # Store patterns in Neo4j
    for pattern in patterns:
        success = memory_store.add_memory_pattern(pattern)
        if success:
            logger.info(f"Stored pattern: {pattern.pattern_type.value}")

    logger.info(f"✅ Recognized {len(patterns)} patterns total")

    return patterns


# =============================================================================
# Memory Compression
# =============================================================================

def compress_session(session_id: str) -> Dict[str, Any]:
    """
    Compress a session by summarizing key moments.

    Creates a compressed representation of a debate session:
    - Key claims (analyst moments)
    - Key objections (skeptic moments)
    - Final synthesis
    - Metadata (iteration count, papers cited, etc.)

    Args:
        session_id: Session identifier

    Returns:
        Dict with compressed session data
    """
    logger.info(f"Compressing session: {session_id}")

    # Get session metadata
    session = memory_store.get_session(session_id)
    if not session:
        logger.error(f"Session not found: {session_id}")
        return {}

    # Get all moments
    moments = memory_store.get_session_moments(session_id)

    # Extract key information
    analyst_moments = [m for m in moments if m.agent_type == "analyst"]
    skeptic_moments = [m for m in moments if m.agent_type == "skeptic"]
    synthesizer_moments = [m for m in moments if m.agent_type == "synthesizer"]

    # Collect all cited papers
    all_papers = set()
    for moment in moments:
        all_papers.update(moment.paper_urls)

    compressed = {
        "session_id": session_id,
        "title": session.title,
        "query": session.original_query,
        "status": session.status.value,
        "created_at": session.created_at.isoformat(),
        "iteration_count": session.iteration_count,
        "summary": {
            "total_moments": len(moments),
            "analyst_claims": len(analyst_moments),
            "skeptic_objections": len(skeptic_moments),
            "syntheses": len(synthesizer_moments),
            "papers_cited": len(all_papers)
        },
        "key_claims": [m.content[:150] + "..." for m in analyst_moments[:3]],  # Top 3 claims
        "key_objections": [m.content[:150] + "..." for m in skeptic_moments[:3]],  # Top 3 objections
        "final_synthesis": synthesizer_moments[-1].content[:300] + "..." if synthesizer_moments else None,
        "papers_cited": list(all_papers)
    }

    logger.info(f"✅ Session compressed: {len(moments)} moments → {len(str(compressed))} bytes")

    return compressed


# =============================================================================
# Context Injection
# =============================================================================

def find_relevant_context(
    query: str,
    workspace_id: str,
    limit: int = 5,
    similarity_threshold: float = 0.75
) -> List[Dict[str, Any]]:
    """
    Find relevant context from past debates for a new query.

    Uses semantic search to find similar past claims, objections, and insights.

    Args:
        query: New research query
        workspace_id: Workspace to search within
        limit: Maximum results to return
        similarity_threshold: Minimum similarity score (0-1)

    Returns:
        List of context items with session info and similarity scores
    """
    logger.info(f"Finding relevant context for query: '{query[:50]}...'")

    # Generate embedding for query
    query_embedding = generate_embedding(query)

    # Search for similar moments
    results = memory_store.search_similar_moments(
        query_embedding=query_embedding,
        workspace_id=workspace_id,
        limit=limit,
        similarity_threshold=similarity_threshold
    )

    # Format context
    context_items = []
    for moment, similarity in results:
        # Get session info
        session = memory_store.get_session(moment.session_id)

        context_item = {
            "similarity": similarity,
            "agent_type": moment.agent_type,
            "content": moment.content[:200] + "..." if len(moment.content) > 200 else moment.content,
            "session_title": session.title if session else "Unknown",
            "session_query": session.original_query if session else "Unknown",
            "round_number": moment.round_number,
            "papers": moment.paper_urls[:3]  # Top 3 papers
        }
        context_items.append(context_item)

    logger.info(f"✅ Found {len(context_items)} relevant context items")

    return context_items


def inject_memory_context(query: str, workspace_id: str) -> str:
    """
    Inject relevant memory context into agent prompts.

    Retrieves similar past debates and formats them as context for the agent.

    Args:
        query: New research query
        workspace_id: Workspace to search within

    Returns:
        Formatted context string to inject into agent prompts
    """
    context_items = find_relevant_context(query, workspace_id)

    if not context_items:
        return ""

    # Format context for injection
    context_parts = ["\n### Relevant Past Debates ###\n"]

    for i, item in enumerate(context_items, 1):
        context_parts.append(
            f"{i}. From '{item['session_title']}' (similarity: {item['similarity']:.2f}):\n"
            f"   {item['agent_type'].capitalize()}: {item['content']}\n"
        )

    context_parts.append("\nUse these insights to inform your reasoning, but don't repeat rejected arguments.\n")

    return "".join(context_parts)
