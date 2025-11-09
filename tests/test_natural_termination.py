"""
Test suite for Natural Termination System (Option 3).

Tests the intelligent "genuinely stuck" detection that replaces
arbitrary MAX_ITERATIONS with natural termination based on
consecutive high-similarity rejections.

Tier 1 Enhancement: Natural Termination
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from src.graph import route_debate
from src.models import AgentState, Antithesis


# =============================================================================
# Test 1: Consecutive High-Similarity Detection
# =============================================================================

def test_consecutive_high_similarity_triggers_stuck():
    """
    Test that 2 consecutive high-similarity rejections (>0.75) triggers
    "genuinely stuck" detection and routes to synthesizer with impasse mode.
    """
    # Create state with 1 previous high-similarity rejection
    state: AgentState = {
        "messages": [],
        "original_query": "What is consciousness?",
        "current_thesis": None,
        "current_antithesis": Antithesis(
            contradiction_found=True,
            counter_claim="This is too similar",
            conflicting_evidence=[],
            critique="Your claim is very similar to previous attempts."
        ),
        "final_synthesis": None,
        "contradiction_report": "",
        "iteration_count": 3,
        "procedural_memory": "",
        "debate_memory": {
            "rejected_claims": ["Claim A", "Claim B"],
            "skeptic_objections": [],
            "weak_evidence_urls": []
        },
        "current_claim_id": "test-claim-123",
        "synthesis_mode": None,
        "consecutive_high_similarity_count": 1,  # First high-similarity rejection
        "last_similarity_score": 0.78  # Second high-similarity rejection
    }
    
    # Route debate
    next_node = route_debate(state)
    
    # Should route to synthesizer (genuinely stuck)
    assert next_node == "synthesizer"
    
    # Should set impasse mode
    assert state["synthesis_mode"] == "impasse"
    
    # Should increment consecutive count to 2
    assert state["consecutive_high_similarity_count"] == 2


# =============================================================================
# Test 2: Similarity Reset on Low Score
# =============================================================================

@patch.dict(os.environ, {"MAX_ITERATIONS": "10"})
def test_similarity_reset_on_low_score():
    """
    Test that consecutive_high_similarity_count is reset to 0 when
    a rejection has low similarity, indicating progress is being made.
    """
    # Create state with 1 previous high-similarity rejection but now low similarity
    state: AgentState = {
        "messages": [],
        "original_query": "What is consciousness?",
        "current_thesis": None,
        "current_antithesis": Antithesis(
            contradiction_found=True,
            counter_claim="This is different, exploring a new theoretical direction",
            conflicting_evidence=[],
            critique="Your claim explores a new perspective with sufficient differentiation from previous attempts."
        ),
        "final_synthesis": None,
        "contradiction_report": "",
        "iteration_count": 3,  # Well below MAX_ITERATIONS=10
        "procedural_memory": "",
        "debate_memory": {
            "rejected_claims": ["Claim A", "Claim B"],
            "skeptic_objections": [],
            "weak_evidence_urls": []
        },
        "current_claim_id": "test-claim-123",
        "synthesis_mode": None,
        "consecutive_high_similarity_count": 1,  # Had 1 high-similarity
        "last_similarity_score": None  # But now similarity is low (None)
    }
    
    # Route debate
    next_node = route_debate(state)
    
    # Should route to analyst (continue exploring)
    assert next_node == "analyst"
    
    # Should reset consecutive count to 0
    assert state["consecutive_high_similarity_count"] == 0


# =============================================================================
# Test 3: First High-Similarity (Not Stuck Yet)
# =============================================================================

def test_first_high_similarity_not_stuck():
    """
    Test that the first high-similarity rejection increments the counter
    but does NOT trigger "genuinely stuck" (need 2 consecutive).
    """
    # Create state with NO previous high-similarity rejections
    state: AgentState = {
        "messages": [],
        "original_query": "What is consciousness?",
        "current_thesis": None,
        "current_antithesis": Antithesis(
            contradiction_found=True,
            counter_claim="This is somewhat similar",
            conflicting_evidence=[],
            critique="Your claim is similar to a previous attempt."
        ),
        "final_synthesis": None,
        "contradiction_report": "",
        "iteration_count": 2,
        "procedural_memory": "",
        "debate_memory": {
            "rejected_claims": ["Claim A"],
            "skeptic_objections": [],
            "weak_evidence_urls": []
        },
        "current_claim_id": "test-claim-123",
        "synthesis_mode": None,
        "consecutive_high_similarity_count": 0,  # No previous high-similarity
        "last_similarity_score": 0.76  # First high-similarity rejection
    }
    
    # Route debate
    next_node = route_debate(state)
    
    # Should route to analyst (not stuck yet, need 2 consecutive)
    assert next_node == "analyst"
    
    # Should increment consecutive count to 1
    assert state["consecutive_high_similarity_count"] == 1
    
    # Should NOT set impasse mode
    assert state["synthesis_mode"] == "standard"


# =============================================================================
# Test 4: Circular Detection Still Works (Priority)
# =============================================================================

def test_circular_detection_priority():
    """
    Test that circular argument detection (>0.80 similarity with keywords)
    still takes priority and immediately routes to synthesizer.
    """
    # Create state with circular argument detected
    state: AgentState = {
        "messages": [],
        "original_query": "What is consciousness?",
        "current_thesis": None,
        "current_antithesis": Antithesis(
            contradiction_found=True,
            counter_claim="This is circular",
            conflicting_evidence=[],
            critique="This claim is 83% similar to previously rejected thesis: 'Previous claim'. This represents a circular argument."
        ),
        "final_synthesis": None,
        "contradiction_report": "",
        "iteration_count": 2,
        "procedural_memory": "",
        "debate_memory": {
            "rejected_claims": ["Previous claim"],
            "skeptic_objections": [],
            "weak_evidence_urls": []
        },
        "current_claim_id": "test-claim-123",
        "synthesis_mode": None,
        "consecutive_high_similarity_count": 0,
        "last_similarity_score": 0.83  # High similarity from circular detection
    }
    
    # Route debate
    next_node = route_debate(state)
    
    # Should route to synthesizer immediately (circular detected)
    assert next_node == "synthesizer"
    
    # Should set impasse mode
    assert state["synthesis_mode"] == "impasse"


# =============================================================================
# Test 5: MAX_ITERATIONS as Safety Net
# =============================================================================

@patch.dict(os.environ, {"MAX_ITERATIONS": "10"})
def test_max_iterations_safety_net():
    """
    Test that MAX_ITERATIONS (default 10) still functions as a safety net
    when natural termination doesn't trigger.
    """
    # Create state at MAX_ITERATIONS with low similarity
    state: AgentState = {
        "messages": [],
        "original_query": "What is consciousness?",
        "current_thesis": None,
        "current_antithesis": Antithesis(
            contradiction_found=True,
            counter_claim="Yet another perspective that needs deeper exploration",
            conflicting_evidence=[],
            critique="Your claim needs refinement and additional supporting evidence to be convincing."
        ),
        "final_synthesis": None,
        "contradiction_report": "",
        "iteration_count": 10,  # At MAX_ITERATIONS (default)
        "procedural_memory": "",
        "debate_memory": {
            "rejected_claims": ["Claim A", "Claim B", "Claim C"],
            "skeptic_objections": [],
            "weak_evidence_urls": []
        },
        "current_claim_id": "test-claim-123",
        "synthesis_mode": None,
        "consecutive_high_similarity_count": 0,  # No high similarity
        "last_similarity_score": None  # Low similarity
    }
    
    # Route debate
    next_node = route_debate(state)
    
    # Should route to synthesizer (safety net)
    assert next_node == "synthesizer"
    
    # Should set exhausted_attempts mode
    assert state["synthesis_mode"] == "exhausted_attempts"


# =============================================================================
# Test 6: No Contradiction Found (Natural Acceptance)
# =============================================================================

def test_no_contradiction_natural_acceptance():
    """
    Test that when Skeptic finds no contradiction, system proceeds to
    synthesizer with standard mode (natural acceptance).
    """
    # Create state with no contradiction found
    state: AgentState = {
        "messages": [],
        "original_query": "What is 2+2?",
        "current_thesis": None,
        "current_antithesis": Antithesis(
            contradiction_found=False,  # No contradiction
            counter_claim=None,
            conflicting_evidence=[],
            critique="The thesis is well-reasoned and supported."
        ),
        "final_synthesis": None,
        "contradiction_report": "",
        "iteration_count": 1,
        "procedural_memory": "",
        "debate_memory": {
            "rejected_claims": [],
            "skeptic_objections": [],
            "weak_evidence_urls": []
        },
        "current_claim_id": "test-claim-123",
        "synthesis_mode": None,
        "consecutive_high_similarity_count": 0,
        "last_similarity_score": None
    }
    
    # Route debate
    next_node = route_debate(state)
    
    # Should route to synthesizer (natural acceptance)
    assert next_node == "synthesizer"
    
    # Should set standard mode
    assert state["synthesis_mode"] == "standard"


# =============================================================================
# Test 7: High Similarity Threshold (0.75)
# =============================================================================

@patch.dict(os.environ, {"MAX_ITERATIONS": "10"})
def test_high_similarity_threshold():
    """
    Test that similarity scores ABOVE 0.75 are counted as high-similarity,
    and scores BELOW 0.75 are not.
    """
    # Test Case 1: 0.74 similarity (below threshold)
    state1: AgentState = {
        "messages": [],
        "original_query": "Test query about complex topic",
        "current_thesis": None,
        "current_antithesis": Antithesis(
            contradiction_found=True,
            counter_claim="This is a test counter-claim with sufficient length",
            conflicting_evidence=[],
            critique="This is a test critique with at least thirty characters to satisfy validation"
        ),
        "final_synthesis": None,
        "contradiction_report": "",
        "iteration_count": 2,
        "procedural_memory": "",
        "debate_memory": {"rejected_claims": [], "skeptic_objections": [], "weak_evidence_urls": []},
        "current_claim_id": "test",
        "synthesis_mode": None,
        "consecutive_high_similarity_count": 1,  # Had 1 high-similarity
        "last_similarity_score": 0.74  # Below threshold (should reset)
    }
    
    next_node1 = route_debate(state1)
    
    # Should NOT count as high-similarity (below 0.75)
    assert state1["consecutive_high_similarity_count"] == 0  # Reset
    assert next_node1 == "analyst"
    
    # Test Case 2: 0.75 similarity (at threshold)
    state2: AgentState = {
        "messages": [],
        "original_query": "Test query about complex topic",
        "current_thesis": None,
        "current_antithesis": Antithesis(
            contradiction_found=True,
            counter_claim="This is a test counter-claim with sufficient length",
            conflicting_evidence=[],
            critique="This is a test critique with at least thirty characters to satisfy validation"
        ),
        "final_synthesis": None,
        "contradiction_report": "",
        "iteration_count": 2,
        "procedural_memory": "",
        "debate_memory": {"rejected_claims": [], "skeptic_objections": [], "weak_evidence_urls": []},
        "current_claim_id": "test",
        "synthesis_mode": None,
        "consecutive_high_similarity_count": 0,
        "last_similarity_score": 0.75  # At threshold (should count)
    }
    
    next_node2 = route_debate(state2)
    
    # Should count as high-similarity (at or above 0.75)
    assert state2["consecutive_high_similarity_count"] == 1  # Incremented
    assert next_node2 == "analyst"  # Not stuck yet (need 2 consecutive)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

