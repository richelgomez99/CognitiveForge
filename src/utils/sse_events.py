"""
Server-Sent Events (SSE) utilities for real-time progress streaming.

This module provides functions to emit progress events during dialectical synthesis,
enabling real-time UI updates for keyword extraction, paper discovery, thesis generation,
critique, and final synthesis.

All events follow the SSE format:
    event: event_name
    data: {"field": "value"}
    
See: specs/004-visualization-ux-tier2/contracts/sse-events.yaml for full schema.
"""

import json
import time
from typing import Dict, Any, List, Optional


def _create_event(event_name: str, data: Dict[str, Any], agent: Optional[str] = None, round_number: Optional[int] = None) -> str:
    """
    Create an SSE event string.
    
    Args:
        event_name: Event type (e.g., "keyword_extraction_start")
        data: Event-specific data dictionary
        agent: Optional agent name ("Analyst", "Skeptic", "Synthesizer")
        round_number: Optional round number (1-indexed)
    
    Returns:
        Formatted SSE event string
    """
    event_data = {
        **data,
        "timestamp": time.time()
    }
    
    if agent:
        event_data["agent"] = agent
    if round_number:
        event_data["round_number"] = round_number
    
    # SSE format: event line, data line, blank line
    return f"event: {event_name}\ndata: {json.dumps(event_data)}\n\n"


# ────────────────────────────────────────────────────────────────
# KEYWORD EXTRACTION EVENTS
# ────────────────────────────────────────────────────────────────

def emit_keyword_extraction_start(claim: str, agent: str = "Analyst", round_number: Optional[int] = None) -> str:
    """
    Emit event when keyword extraction starts.
    
    Args:
        claim: Claim to extract keywords from
        agent: Agent name (default: "Analyst")
        round_number: Optional round number
    
    Returns:
        SSE event string
    """
    return _create_event(
        "keyword_extraction_start",
        {"claim": claim},
        agent=agent,
        round_number=round_number
    )


def emit_keyword_extraction_complete(keywords: List[str], agent: str = "Analyst", round_number: Optional[int] = None) -> str:
    """
    Emit event when keyword extraction completes.
    
    Args:
        keywords: Extracted multi-word keywords
        agent: Agent name (default: "Analyst")
        round_number: Optional round number
    
    Returns:
        SSE event string
    """
    return _create_event(
        "keyword_extraction_complete",
        {
            "keywords": keywords,
            "count": len(keywords)
        },
        agent=agent,
        round_number=round_number
    )


# ────────────────────────────────────────────────────────────────
# DISCOVERY EVENTS
# ────────────────────────────────────────────────────────────────

def emit_discovery_start(source: str, query: str, agent: str, round_number: Optional[int] = None) -> str:
    """
    Emit event when paper discovery starts.
    
    Args:
        source: Discovery source ("arxiv" or "semantic_scholar")
        query: Search query (keyword)
        agent: Agent name ("Analyst" or "Skeptic")
        round_number: Optional round number
    
    Returns:
        SSE event string
    """
    return _create_event(
        "discovery_start",
        {
            "source": source,
            "query": query
        },
        agent=agent,
        round_number=round_number
    )


def emit_discovery_paper_found(title: str, url: str, source: str, agent: str, round_number: Optional[int] = None) -> str:
    """
    Emit event when a paper is discovered (real-time, per paper).
    
    Args:
        title: Paper title
        url: Paper URL
        source: Discovery source ("arxiv" or "semantic_scholar")
        agent: Agent name ("Analyst" or "Skeptic")
        round_number: Optional round number
    
    Returns:
        SSE event string
    """
    return _create_event(
        "discovery_paper_found",
        {
            "title": title,
            "url": url,
            "source": source
        },
        agent=agent,
        round_number=round_number
    )


def emit_discovery_complete(total: int, duplicates: int, added: int, agent: str, round_number: Optional[int] = None) -> str:
    """
    Emit event when paper discovery completes.
    
    Args:
        total: Total papers discovered
        duplicates: Duplicate papers (already in KG)
        added: New papers added to KG
        agent: Agent name ("Analyst" or "Skeptic")
        round_number: Optional round number
    
    Returns:
        SSE event string
    """
    return _create_event(
        "discovery_complete",
        {
            "total": total,
            "duplicates": duplicates,
            "added": added
        },
        agent=agent,
        round_number=round_number
    )


# ────────────────────────────────────────────────────────────────
# THESIS GENERATION EVENTS
# ────────────────────────────────────────────────────────────────

def emit_thesis_generation_start(evidence_count: int, round_number: Optional[int] = None) -> str:
    """
    Emit event when thesis generation starts.
    
    Args:
        evidence_count: Number of papers in evidence pool
        round_number: Optional round number
    
    Returns:
        SSE event string
    """
    return _create_event(
        "thesis_generation_start",
        {"evidence_count": evidence_count},
        agent="Analyst",
        round_number=round_number
    )


def emit_thesis_generation_complete(confidence: float, claim: str, round_number: Optional[int] = None) -> str:
    """
    Emit event when thesis generation completes.
    
    Args:
        confidence: Confidence score (0-1)
        claim: Generated thesis claim (truncated to 200 chars for event)
        round_number: Optional round number
    
    Returns:
        SSE event string
    """
    # Truncate claim for event (full claim in state)
    truncated_claim = claim[:200] + "..." if len(claim) > 200 else claim
    
    return _create_event(
        "thesis_generation_complete",
        {
            "confidence": confidence,
            "claim": truncated_claim
        },
        agent="Analyst",
        round_number=round_number
    )


# ────────────────────────────────────────────────────────────────
# CRITIQUE EVENTS
# ────────────────────────────────────────────────────────────────

def emit_critique_start(thesis_claim: str, round_number: Optional[int] = None) -> str:
    """
    Emit event when critique starts.
    
    Args:
        thesis_claim: Thesis being critiqued (truncated to 200 chars)
        round_number: Optional round number
    
    Returns:
        SSE event string
    """
    truncated_claim = thesis_claim[:200] + "..." if len(thesis_claim) > 200 else thesis_claim
    
    return _create_event(
        "critique_start",
        {"thesis_claim": truncated_claim},
        agent="Skeptic",
        round_number=round_number
    )


def emit_critique_complete(
    contradiction_found: bool,
    critique: str,
    counter_evidence_count: int,
    round_number: Optional[int] = None
) -> str:
    """
    Emit event when critique completes.
    
    Args:
        contradiction_found: True if Skeptic found contradictions
        critique: Critique summary (truncated to 200 chars)
        counter_evidence_count: Number of counter-evidence papers found
        round_number: Optional round number
    
    Returns:
        SSE event string
    """
    truncated_critique = critique[:200] + "..." if len(critique) > 200 else critique
    
    return _create_event(
        "critique_complete",
        {
            "contradiction_found": contradiction_found,
            "critique": truncated_critique,
            "counter_evidence_count": counter_evidence_count
        },
        agent="Skeptic",
        round_number=round_number
    )


# ────────────────────────────────────────────────────────────────
# SYNTHESIS EVENTS
# ────────────────────────────────────────────────────────────────

def emit_synthesis_start(rounds_completed: int, total_papers: int) -> str:
    """
    Emit event when synthesis starts.
    
    Args:
        rounds_completed: Number of dialectical rounds completed
        total_papers: Total papers in knowledge pool
    
    Returns:
        SSE event string
    """
    return _create_event(
        "synthesis_start",
        {
            "rounds_completed": rounds_completed,
            "total_papers": total_papers
        },
        agent="Synthesizer"
    )


def emit_synthesis_complete(
    confidence: float,
    novelty: float,
    insight: str,
    word_count: int
) -> str:
    """
    Emit event when synthesis completes.
    
    Args:
        confidence: Confidence score (0-100)
        novelty: Novelty score (0-100)
        insight: Final synthesis insight (truncated to 300 chars)
        word_count: Total word count of comprehensive synthesis
    
    Returns:
        SSE event string
    """
    truncated_insight = insight[:300] + "..." if len(insight) > 300 else insight
    
    return _create_event(
        "synthesis_complete",
        {
            "confidence": confidence,
            "novelty": novelty,
            "insight": truncated_insight,
            "word_count": word_count
        },
        agent="Synthesizer"
    )


# ────────────────────────────────────────────────────────────────
# ERROR EVENT
# ────────────────────────────────────────────────────────────────

def emit_error(
    message: str,
    error_type: str,
    recoverable: bool = True,
    agent: Optional[str] = None,
    round_number: Optional[int] = None
) -> str:
    """
    Emit error event.
    
    Args:
        message: Human-readable error message
        error_type: Error category ("api_error", "validation_error", "discovery_error", "kg_error")
        recoverable: True if system can recover (e.g., retry)
        agent: Optional agent name
        round_number: Optional round number
    
    Returns:
        SSE event string
    """
    return _create_event(
        "error",
        {
            "message": message,
            "error_type": error_type,
            "recoverable": recoverable
        },
        agent=agent,
        round_number=round_number
    )

