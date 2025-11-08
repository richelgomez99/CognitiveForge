"""
Unit tests for LangGraph routing logic.

Tests the route_debate conditional function and iteration management.
"""

import pytest
import os
from src.graph import route_debate, increment_iteration
from src.models import AgentState, Antithesis, Evidence


class TestRouteDebate:
    """Test the route_debate conditional routing function."""
    
    def test_route_to_analyst_when_contradiction_found(self):
        """Test routing back to analyst when contradiction found and iterations remain."""
        # Set MAX_ITERATIONS for testing
        os.environ["MAX_ITERATIONS"] = "3"
        
        antithesis = Antithesis(
            contradiction_found=True,
            counter_claim="There are significant computational limitations to consider",
            conflicting_evidence=[Evidence(source_url="https://example.com/paper", snippet="This evidence shows conflicting data about computational costs")],
            critique="The thesis overlooks important computational cost considerations that impact practical deployment"
        )
        
        state: AgentState = {
            "messages": [],
            "original_query": "Test query",
            "current_thesis": None,
            "current_antithesis": antithesis,
            "final_synthesis": None,
            "contradiction_report": "Found contradiction",
            "iteration_count": 0,  # First iteration, can loop
            "procedural_memory": ""
        }
        
        # Should route to analyst for refinement
        result = route_debate(state)
        assert result == "analyst"
    
    def test_route_to_synthesizer_when_no_contradiction(self):
        """Test routing to synthesizer when no contradiction found."""
        antithesis = Antithesis(
            contradiction_found=False,
            counter_claim=None,
            conflicting_evidence=[],
            critique="The thesis is well-supported with comprehensive evidence and sound logical reasoning"
        )
        
        state: AgentState = {
            "messages": [],
            "original_query": "Test query",
            "current_thesis": None,
            "current_antithesis": antithesis,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": ""
        }
        
        # Should proceed to synthesizer
        result = route_debate(state)
        assert result == "synthesizer"
    
    def test_route_to_synthesizer_at_max_iterations(self):
        """Test routing to synthesizer when max iterations reached."""
        os.environ["MAX_ITERATIONS"] = "3"
        
        antithesis = Antithesis(
            contradiction_found=True,  # Contradiction exists
            counter_claim="There are still significant issues that remain unresolved",
            conflicting_evidence=[Evidence(source_url="https://example.com/conflict", snippet="This evidence demonstrates continuing conflicts in the analysis")],
            critique="Significant problems remain unaddressed in the thesis formulation"
        )
        
        state: AgentState = {
            "messages": [],
            "original_query": "Test query",
            "current_thesis": None,
            "current_antithesis": antithesis,
            "final_synthesis": None,
            "contradiction_report": "Issues found",
            "iteration_count": 3,  # At max iterations
            "procedural_memory": ""
        }
        
        # Should proceed to synthesizer despite contradiction
        result = route_debate(state)
        assert result == "synthesizer"
    
    def test_route_with_different_max_iterations(self):
        """Test routing respects MAX_ITERATIONS environment variable."""
        # Set lower max iterations
        os.environ["MAX_ITERATIONS"] = "2"
        
        antithesis = Antithesis(
            contradiction_found=True,
            counter_claim="There are important issues that need to be addressed",
            conflicting_evidence=[Evidence(source_url="https://example.com/conflict", snippet="This evidence shows conflicting information about the approach")],
            critique="The analysis has several problems that require attention and resolution"
        )
        
        # At iteration 1, should still loop (< 2)
        state_1: AgentState = {
            "messages": [],
            "original_query": "Test",
            "current_thesis": None,
            "current_antithesis": antithesis,
            "final_synthesis": None,
            "contradiction_report": "Issues",
            "iteration_count": 1,
            "procedural_memory": ""
        }
        
        result_1 = route_debate(state_1)
        assert result_1 == "analyst"
        
        # At iteration 2, should stop looping (>= 2)
        state_2: AgentState = {
            "messages": [],
            "original_query": "Test",
            "current_thesis": None,
            "current_antithesis": antithesis,
            "final_synthesis": None,
            "contradiction_report": "Issues",
            "iteration_count": 2,
            "procedural_memory": ""
        }
        
        result_2 = route_debate(state_2)
        assert result_2 == "synthesizer"


class TestIncrementIteration:
    """Test the increment_iteration helper node."""
    
    def test_increment_from_zero(self):
        """Test incrementing iteration count from 0."""
        state: AgentState = {
            "messages": [],
            "original_query": "Test",
            "current_thesis": None,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": ""
        }
        
        result = increment_iteration(state)
        
        assert "iteration_count" in result
        assert result["iteration_count"] == 1
    
    def test_increment_from_nonzero(self):
        """Test incrementing iteration count from positive value."""
        state: AgentState = {
            "messages": [],
            "original_query": "Test",
            "current_thesis": None,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 2,
            "procedural_memory": ""
        }
        
        result = increment_iteration(state)
        
        assert result["iteration_count"] == 3
    
    def test_increment_preserves_only_iteration_count(self):
        """Test increment only returns iteration_count update."""
        state: AgentState = {
            "messages": ["msg1", "msg2"],
            "original_query": "Complex query",
            "current_thesis": None,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "Report",
            "iteration_count": 1,
            "procedural_memory": "Memory"
        }
        
        result = increment_iteration(state)
        
        # Should only contain iteration_count
        assert len(result) == 1
        assert "iteration_count" in result
        assert result["iteration_count"] == 2


class TestRoutingLogic:
    """Integration tests for routing logic."""
    
    def test_full_iteration_cycle(self):
        """Test a full iteration cycle: contradiction → increment → analyst."""
        os.environ["MAX_ITERATIONS"] = "3"
        
        # Step 1: Contradiction found, route to analyst
        antithesis = Antithesis(
            contradiction_found=True,
            counter_claim="There are significant counter-arguments to consider",
            conflicting_evidence=[Evidence(source_url="https://example.com/paper", snippet="This evidence provides alternative perspectives")],
            critique="The original thesis requires refinement based on additional considerations"
        )
        
        state_step1: AgentState = {
            "messages": [],
            "original_query": "Test",
            "current_thesis": None,
            "current_antithesis": antithesis,
            "final_synthesis": None,
            "contradiction_report": "Found",
            "iteration_count": 0,
            "procedural_memory": ""
        }
        
        route_result = route_debate(state_step1)
        assert route_result == "analyst"
        
        # Step 2: Increment iteration count
        increment_result = increment_iteration(state_step1)
        assert increment_result["iteration_count"] == 1
        
        # Step 3: Updated state for next iteration
        state_step2: AgentState = {
            **state_step1,
            "iteration_count": increment_result["iteration_count"]
        }
        
        assert state_step2["iteration_count"] == 1
        
        # If contradiction persists, can route to analyst again
        route_result_2 = route_debate(state_step2)
        assert route_result_2 == "analyst"  # Still < max iterations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

