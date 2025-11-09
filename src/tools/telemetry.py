"""
Telemetry tracking for discovery strategy analysis (Tier 1: T077).

Logs actual vs. predicted paper counts for later analysis.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

TELEMETRY_DIR = Path(".taskmaster/telemetry")
TELEMETRY_FILE = TELEMETRY_DIR / "discovery_telemetry.jsonl"

# Enable/disable telemetry (default: enabled in dev, disabled in prod)
TELEMETRY_ENABLED = os.getenv("TELEMETRY_ENABLED", "true").lower() == "true"


# =============================================================================
# Telemetry Logging
# =============================================================================

def log_discovery_telemetry(
    claim: str,
    strategy_recommended: int,
    actual_papers_fetched: int,
    avg_relevance: float,
    follow_up_triggered: bool,
    complexity_level: str,
    claim_id: Optional[str] = None
) -> None:
    """
    Log discovery telemetry for strategy analysis.
    
    Tier 1: T077 - Telemetry tracking for discovery strategy validation.
    
    Args:
        claim: The thesis claim
        strategy_recommended: Number of papers recommended by LLM strategy
        actual_papers_fetched: Actual number of papers discovered
        avg_relevance: Average relevance score of discovered papers
        follow_up_triggered: Whether follow-up discovery was triggered
        complexity_level: Inferred complexity level (LOW, MEDIUM, HIGH)
        claim_id: Optional UUID for tracking
    
    Output:
        Appends JSON line to .taskmaster/telemetry/discovery_telemetry.jsonl
    
    Example:
        >>> log_discovery_telemetry(
        ...     claim="Consciousness is computational",
        ...     strategy_recommended=7,
        ...     actual_papers_fetched=6,
        ...     avg_relevance=0.75,
        ...     follow_up_triggered=False,
        ...     complexity_level="HIGH"
        ... )
        # Appends: {"timestamp": "2025-01-01T12:00:00", "claim": "...", ...}
    
    Tier 1: Used to validate adaptive discovery strategy effectiveness.
    """
    if not TELEMETRY_ENABLED:
        logger.debug("Telemetry disabled, skipping log")
        return
    
    try:
        # Ensure telemetry directory exists
        TELEMETRY_DIR.mkdir(parents=True, exist_ok=True)
        
        # Build telemetry record
        record: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "claim": claim[:200],  # Truncate for readability
            "claim_id": claim_id,
            "strategy_recommended": strategy_recommended,
            "actual_papers_fetched": actual_papers_fetched,
            "avg_relevance": round(avg_relevance, 3),
            "follow_up_triggered": follow_up_triggered,
            "complexity_level": complexity_level,
            "delta": actual_papers_fetched - strategy_recommended  # Difference (+ or -)
        }
        
        # Append to JSONL file
        with open(TELEMETRY_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
        
        logger.debug(f"📊 Telemetry logged: {strategy_recommended} rec → {actual_papers_fetched} actual (delta: {record['delta']})")
    
    except Exception as e:
        logger.warning(f"Failed to log telemetry: {e}")
        # Don't fail the main operation if telemetry fails


# =============================================================================
# Telemetry Analysis
# =============================================================================

def get_telemetry_summary() -> Dict[str, Any]:
    """
    Load and summarize discovery telemetry for analysis.
    
    Tier 1: T077 - Helper function for telemetry analysis.
    
    Returns:
        Dictionary with telemetry statistics:
        - total_discoveries: Total number of logged discoveries
        - avg_delta: Average difference between recommended and actual
        - complexity_breakdown: Counts by complexity level
        - follow_up_rate: Percentage of discoveries that triggered follow-up
    
    Example:
        >>> summary = get_telemetry_summary()
        >>> print(f"Avg delta: {summary['avg_delta']:.2f} papers")
    
    Tier 1: Used by validation scripts to assess strategy accuracy.
    """
    if not TELEMETRY_FILE.exists():
        logger.warning("No telemetry data found")
        return {
            "total_discoveries": 0,
            "avg_delta": 0.0,
            "complexity_breakdown": {},
            "follow_up_rate": 0.0
        }
    
    try:
        records = []
        with open(TELEMETRY_FILE, "r") as f:
            for line in f:
                records.append(json.loads(line.strip()))
        
        if not records:
            return {
                "total_discoveries": 0,
                "avg_delta": 0.0,
                "complexity_breakdown": {},
                "follow_up_rate": 0.0
            }
        
        # Calculate statistics
        total_discoveries = len(records)
        avg_delta = sum(r["delta"] for r in records) / total_discoveries
        
        # Complexity breakdown
        complexity_counts = {}
        for r in records:
            level = r.get("complexity_level", "UNKNOWN")
            complexity_counts[level] = complexity_counts.get(level, 0) + 1
        
        # Follow-up rate
        follow_up_count = sum(1 for r in records if r.get("follow_up_triggered", False))
        follow_up_rate = (follow_up_count / total_discoveries) * 100
        
        return {
            "total_discoveries": total_discoveries,
            "avg_delta": round(avg_delta, 2),
            "complexity_breakdown": complexity_counts,
            "follow_up_rate": round(follow_up_rate, 2)
        }
    
    except Exception as e:
        logger.error(f"Failed to load telemetry: {e}")
        return {
            "total_discoveries": 0,
            "avg_delta": 0.0,
            "complexity_breakdown": {},
            "follow_up_rate": 0.0
        }

