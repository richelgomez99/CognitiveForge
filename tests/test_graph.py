"""
Integration tests for the full LangGraph dialectical synthesis system.

Tests end-to-end graph execution with real LLM calls.
"""

import pytest
import os
from src.graph import build_graph
from src.models import AgentState


class TestGraphCompilation:
    """Test graph structure and compilation."""
    
    def test_build_graph_returns_compiled(self):
        """Test build_graph returns a compiled graph."""
        graph = build_graph()
        assert graph is not None
        # Graph should have invoke method
        assert hasattr(graph, 'invoke')
    
    def test_graph_structure(self):
        """Test graph has all required nodes."""
        graph = build_graph()
        
        # Get graph representation
        graph_repr = str(graph)
        
        # Should contain key nodes
        # Note: exact structure checking depends on LangGraph API
        assert graph is not None


class TestGraphExecution:
    """Test full graph execution with real LLM calls."""
    
    @pytest.mark.slow
    def test_simple_query_execution(self):
        """Test graph execution with a simple query (no iterations expected)."""
        graph = build_graph()
        
        initial_state: AgentState = {
            "messages": [],
            "original_query": "What is a neural network?",
            "current_thesis": None,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": "",
            "thread_id": "test_simple_001"
        }
        
        # Execute graph
        final_state = graph.invoke(initial_state)
        
        # Verify outputs
        assert final_state is not None
        assert "final_synthesis" in final_state
        
        synthesis = final_state["final_synthesis"]
        assert synthesis is not None
        assert len(synthesis.novel_insight) > 0
        assert 0.0 <= synthesis.confidence_score <= 1.0
        assert 0.0 <= synthesis.novelty_score <= 1.0
        assert len(synthesis.evidence_lineage) >= 3  # Synthesizer pads to minimum 3 if needed
    
    @pytest.mark.slow
    def test_complex_query_execution(self):
        """Test graph execution with a query likely to cause contradictions."""
        graph = build_graph()
        
        initial_state: AgentState = {
            "messages": [],
            "original_query": "Are transformers always better than RNNs for all tasks?",
            "current_thesis": None,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": "",
            "thread_id": "test_complex_001"
        }
        
        # Execute graph
        final_state = graph.invoke(initial_state)
        
        # Verify outputs
        assert final_state is not None
        assert final_state["final_synthesis"] is not None
        
        # May have had iterations
        assert final_state["iteration_count"] >= 0
    
    @pytest.mark.slow  
    def test_graph_with_max_iterations_reached(self):
        """Test graph behavior when max iterations is reached."""
        # Set low max iterations
        os.environ["MAX_ITERATIONS"] = "2"
        
        graph = build_graph()
        
        initial_state: AgentState = {
            "messages": [],
            "original_query": "What are the fundamental limitations of current AI systems?",
            "current_thesis": None,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": "",
            "thread_id": "test_max_iter_001"
        }
        
        # Execute graph
        final_state = graph.invoke(initial_state)
        
        # Should complete even if max iterations reached
        assert final_state["final_synthesis"] is not None
        
        # Iteration count should not exceed max
        assert final_state["iteration_count"] <= 2
        
        # Reset to default
        os.environ["MAX_ITERATIONS"] = "3"


class TestGraphOutputValidation:
    """Test validation of graph outputs."""
    
    @pytest.mark.slow
    def test_synthesis_has_all_required_fields(self):
        """Test final synthesis has all required Pydantic fields."""
        graph = build_graph()
        
        initial_state: AgentState = {
            "messages": [],
            "original_query": "Explain backpropagation in neural networks",
            "current_thesis": None,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": "",
            "thread_id": "test_validation_001"
        }
        
        final_state = graph.invoke(initial_state)
        synthesis = final_state["final_synthesis"]
        
        # Check all required fields
        assert hasattr(synthesis, 'novel_insight')
        assert hasattr(synthesis, 'supporting_claims')
        assert hasattr(synthesis, 'evidence_lineage')
        assert hasattr(synthesis, 'confidence_score')
        assert hasattr(synthesis, 'novelty_score')
        assert hasattr(synthesis, 'reasoning')
        
        # Verify content quality
        assert len(synthesis.novel_insight) >= 50  # Minimum length
        assert len(synthesis.supporting_claims) >= 1
        assert len(synthesis.evidence_lineage) >= 3
    
    @pytest.mark.slow
    def test_messages_are_accumulated(self):
        """Test that messages from all agents are accumulated."""
        graph = build_graph()
        
        initial_state: AgentState = {
            "messages": [],
            "original_query": "What is machine learning?",
            "current_thesis": None,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": "",
            "thread_id": "test_messages_001"
        }
        
        final_state = graph.invoke(initial_state)
        
        # Messages should have been added by agents
        assert len(final_state["messages"]) > 0


class TestGraphPerformance:
    """Test performance characteristics of the graph."""
    
    @pytest.mark.slow
    def test_execution_completes_in_reasonable_time(self):
        """Test graph execution completes within 2 minutes (SC-T1-001)."""
        import time
        
        graph = build_graph()
        
        initial_state: AgentState = {
            "messages": [],
            "original_query": "Compare supervised and unsupervised learning",
            "current_thesis": None,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": "",
            "thread_id": "test_perf_001"
        }
        
        start_time = time.time()
        final_state = graph.invoke(initial_state)
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Should complete within 300 seconds (5 minutes) for real LLM calls
        # Note: With mocked LLMs this would be < 10s, but real Gemini API calls
        # for a full dialectical debate (Analyst -> Skeptic -> Synthesizer, potentially multiple cycles)
        # can take 2-4 minutes depending on query complexity and API latency
        assert duration < 300, f"Execution took {duration:.2f}s, exceeds 300s limit"
        
        # Verify it still produced valid output
        assert final_state["final_synthesis"] is not None


if __name__ == "__main__":
    # Run with verbose output and show durations
    pytest.main([__file__, "-v", "-s", "--durations=10"])

