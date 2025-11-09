"""
Memory management utilities for Tier 1 iterative debate.

This module provides functions for managing debate memory to prevent circular arguments.
"""

import logging
from typing import List
from src.models import DebateMemory

logger = logging.getLogger(__name__)

# Maximum number of items to store in each memory list (FIFO eviction)
MAX_MEMORY_ITEMS = 10


def update_debate_memory(
    memory: DebateMemory,
    rejected_claim: str = None,
    skeptic_objection: str = None,
    weak_evidence_url: str = None
) -> DebateMemory:
    """
    Update debate memory with new rejected claims, objections, or weak evidence.
    
    Implements FIFO eviction: if any list exceeds MAX_MEMORY_ITEMS (10),
    remove the oldest items to maintain the limit.
    
    Tier 1: T041 - Memory management with FIFO eviction.
    
    Args:
        memory: Current DebateMemory state
        rejected_claim: New claim that was rejected (optional)
        skeptic_objection: New objection from Skeptic (optional)
        weak_evidence_url: URL of paper deemed weak/irrelevant (optional)
    
    Returns:
        Updated DebateMemory with new items added and FIFO eviction applied
    
    Example:
        >>> memory = {"rejected_claims": [], "skeptic_objections": [], "weak_evidence_urls": []}
        >>> memory = update_debate_memory(memory, rejected_claim="AI is sentient")
        >>> len(memory["rejected_claims"])
        1
    """
    # Create mutable copies of the lists
    rejected_claims = list(memory.get("rejected_claims", []))
    skeptic_objections = list(memory.get("skeptic_objections", []))
    weak_evidence_urls = list(memory.get("weak_evidence_urls", []))
    
    # Add new items if provided
    if rejected_claim:
        rejected_claims.append(rejected_claim)
        logger.info(f"📝 Added rejected claim to memory: {rejected_claim[:60]}...")
    
    if skeptic_objection:
        skeptic_objections.append(skeptic_objection)
        logger.info(f"📝 Added skeptic objection to memory: {skeptic_objection[:60]}...")
    
    if weak_evidence_url:
        weak_evidence_urls.append(weak_evidence_url)
        logger.info(f"📝 Added weak evidence URL to memory: {weak_evidence_url}")
    
    # Apply FIFO eviction if any list exceeds MAX_MEMORY_ITEMS
    if len(rejected_claims) > MAX_MEMORY_ITEMS:
        removed = rejected_claims[:len(rejected_claims) - MAX_MEMORY_ITEMS]
        rejected_claims = rejected_claims[-MAX_MEMORY_ITEMS:]
        logger.info(f"🗑️  FIFO eviction: removed {len(removed)} oldest rejected claims")
    
    if len(skeptic_objections) > MAX_MEMORY_ITEMS:
        removed = skeptic_objections[:len(skeptic_objections) - MAX_MEMORY_ITEMS]
        skeptic_objections = skeptic_objections[-MAX_MEMORY_ITEMS:]
        logger.info(f"🗑️  FIFO eviction: removed {len(removed)} oldest skeptic objections")
    
    if len(weak_evidence_urls) > MAX_MEMORY_ITEMS:
        removed = weak_evidence_urls[:len(weak_evidence_urls) - MAX_MEMORY_ITEMS]
        weak_evidence_urls = weak_evidence_urls[-MAX_MEMORY_ITEMS:]
        logger.info(f"🗑️  FIFO eviction: removed {len(removed)} oldest weak evidence URLs")
    
    # Return updated memory
    return DebateMemory(
        rejected_claims=rejected_claims,
        skeptic_objections=skeptic_objections,
        weak_evidence_urls=weak_evidence_urls
    )


def format_memory_for_prompt(memory: DebateMemory) -> str:
    """
    Format debate memory for inclusion in LLM prompts.
    
    Tier 1: Helper function for T043 - Memory-informed prompts.
    
    Args:
        memory: Current DebateMemory state
    
    Returns:
        Formatted string for prompt inclusion
    
    Example:
        >>> memory = {"rejected_claims": ["Claim 1", "Claim 2"], "skeptic_objections": ["Obj 1"], "weak_evidence_urls": []}
        >>> formatted = format_memory_for_prompt(memory)
        >>> "REJECTED" in formatted
        True
    """
    if not memory:
        return ""
    
    rejected_claims = memory.get("rejected_claims", [])
    skeptic_objections = memory.get("skeptic_objections", [])
    weak_evidence_urls = memory.get("weak_evidence_urls", [])
    
    # Only include non-empty sections
    sections = []
    
    if rejected_claims:
        claims_text = "\n".join([f"  - {claim}" for claim in rejected_claims])
        sections.append(f"REJECTED CLAIMS (avoid similar arguments):\n{claims_text}")
    
    if skeptic_objections:
        objections_text = "\n".join([f"  - {obj}" for obj in skeptic_objections])
        sections.append(f"PAST OBJECTIONS (address or avoid):\n{objections_text}")
    
    if weak_evidence_urls:
        urls_text = "\n".join([f"  - {url}" for url in weak_evidence_urls])
        sections.append(f"WEAK EVIDENCE SOURCES (avoid citing):\n{urls_text}")
    
    if not sections:
        return ""
    
    return "\n\n".join(sections)


def get_memory_summary(memory: DebateMemory) -> str:
    """
    Get a brief summary of memory contents for logging.
    
    Tier 1: Helper function for T045 - Memory logging.
    
    Args:
        memory: Current DebateMemory state
    
    Returns:
        Brief summary string
    
    Example:
        >>> memory = {"rejected_claims": ["A", "B"], "skeptic_objections": ["X"], "weak_evidence_urls": []}
        >>> summary = get_memory_summary(memory)
        >>> "2 rejected claims" in summary
        True
    """
    if not memory:
        return "Memory: empty"
    
    rejected_count = len(memory.get("rejected_claims", []))
    objections_count = len(memory.get("skeptic_objections", []))
    weak_urls_count = len(memory.get("weak_evidence_urls", []))
    
    parts = []
    if rejected_count > 0:
        parts.append(f"{rejected_count} rejected claims")
    if objections_count > 0:
        parts.append(f"{objections_count} objections")
    if weak_urls_count > 0:
        parts.append(f"{weak_urls_count} weak URLs")
    
    if not parts:
        return "Memory: empty"
    
    return f"Memory: {', '.join(parts)}"

