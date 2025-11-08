"""
End-to-end tests with diverse research queries.

Tests the full system with 5+ different query types and validates success metrics.
"""

import pytest
from src.graph import build_graph
from src.models import AgentState


class TestDiverseQueries:
    """Test system with diverse research queries across different domains."""
    
    @pytest.mark.slow
    @pytest.mark.e2e
    def test_technical_architecture_query(self):
        """Test Query 1: Technical architecture comparison."""
        graph = build_graph()
        
        state: AgentState = {
            "messages": [],
            "original_query": "What are the key differences between transformer and LSTM architectures?",
            "current_thesis": None,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": "",
            "thread_id": "e2e_technical_001"
        }
        
        final_state = graph.invoke(state)
        
        # Validate success criteria
        synthesis = final_state["final_synthesis"]
        assert synthesis.confidence_score >= 0.7  # SC-T1-002
        assert synthesis.novelty_score >= 0.3  # SC-T1-003
        assert len(synthesis.evidence_lineage) >= 3  # SC-T1-004
        
        print(f"\n✅ Technical Query - Confidence: {synthesis.confidence_score:.2f}, Novelty: {synthesis.novelty_score:.2f}")
    
    @pytest.mark.slow
    @pytest.mark.e2e
    def test_limitation_analysis_query(self):
        """Test Query 2: Limitation and trade-off analysis."""
        graph = build_graph()
        
        state: AgentState = {
            "messages": [],
            "original_query": "What are the computational limitations of attention mechanisms in transformers?",
            "current_thesis": None,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": "",
            "thread_id": "e2e_limitations_001"
        }
        
        final_state = graph.invoke(state)
        
        synthesis = final_state["final_synthesis"]
        assert synthesis.confidence_score >= 0.7
        assert synthesis.novelty_score >= 0.3
        assert len(synthesis.evidence_lineage) >= 3
        
        # This query should likely trigger contradiction (skeptic finds limitations)
        # So iteration_count might be > 0
        print(f"\n✅ Limitations Query - Iterations: {final_state['iteration_count']}, Confidence: {synthesis.confidence_score:.2f}")
    
    @pytest.mark.slow
    @pytest.mark.e2e
    def test_application_domain_query(self):
        """Test Query 3: Application domain suitability."""
        graph = build_graph()
        
        state: AgentState = {
            "messages": [],
            "original_query": "How effective is transfer learning for low-resource languages in NLP?",
            "current_thesis": None,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": "",
            "thread_id": "e2e_application_001"
        }
        
        final_state = graph.invoke(state)
        
        synthesis = final_state["final_synthesis"]
        assert synthesis.confidence_score >= 0.7
        assert synthesis.novelty_score >= 0.3
        assert len(synthesis.evidence_lineage) >= 3
        
        print(f"\n✅ Application Query - Confidence: {synthesis.confidence_score:.2f}, Novelty: {synthesis.novelty_score:.2f}")
    
    @pytest.mark.slow
    @pytest.mark.e2e
    def test_theoretical_concept_query(self):
        """Test Query 4: Theoretical concept explanation."""
        graph = build_graph()
        
        state: AgentState = {
            "messages": [],
            "original_query": "What is the relationship between self-attention and positional encoding in transformers?",
            "current_thesis": None,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": "",
            "thread_id": "e2e_theory_001"
        }
        
        final_state = graph.invoke(state)
        
        synthesis = final_state["final_synthesis"]
        assert synthesis.confidence_score >= 0.7
        assert synthesis.novelty_score >= 0.3
        assert len(synthesis.evidence_lineage) >= 3
        
        print(f"\n✅ Theoretical Query - Confidence: {synthesis.confidence_score:.2f}, Novelty: {synthesis.novelty_score:.2f}")
    
    @pytest.mark.slow
    @pytest.mark.e2e
    def test_optimization_strategy_query(self):
        """Test Query 5: Optimization and improvement strategies."""
        graph = build_graph()
        
        state: AgentState = {
            "messages": [],
            "original_query": "What are the most effective strategies for reducing inference time in large language models?",
            "current_thesis": None,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": "",
            "thread_id": "e2e_optimization_001"
        }
        
        final_state = graph.invoke(state)
        
        synthesis = final_state["final_synthesis"]
        assert synthesis.confidence_score >= 0.7
        assert synthesis.novelty_score >= 0.3
        assert len(synthesis.evidence_lineage) >= 3
        
        print(f"\n✅ Optimization Query - Confidence: {synthesis.confidence_score:.2f}, Novelty: {synthesis.novelty_score:.2f}")
    
    @pytest.mark.slow
    @pytest.mark.e2e
    def test_emerging_trend_query(self):
        """Test Query 6: Emerging trends and future directions."""
        graph = build_graph()
        
        state: AgentState = {
            "messages": [],
            "original_query": "How does retrieval-augmented generation (RAG) improve LLM performance?",
            "current_thesis": None,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": "",
            "thread_id": "e2e_trend_001"
        }
        
        final_state = graph.invoke(state)
        
        synthesis = final_state["final_synthesis"]
        assert synthesis.confidence_score >= 0.7
        assert synthesis.novelty_score >= 0.3
        assert len(synthesis.evidence_lineage) >= 3
        
        print(f"\n✅ Trend Query - Confidence: {synthesis.confidence_score:.2f}, Novelty: {synthesis.novelty_score:.2f}")


class TestSuccessMetrics:
    """Test all success criteria metrics (SC-T1-001 through SC-T1-004)."""
    
    @pytest.mark.slow
    @pytest.mark.e2e
    def test_all_success_criteria(self):
        """Test all success criteria with a comprehensive query."""
        import time
        
        graph = build_graph()
        
        state: AgentState = {
            "messages": [],
            "original_query": "What are the trade-offs between model size and performance in modern neural networks?",
            "current_thesis": None,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": "",
            "thread_id": "e2e_metrics_001"
        }
        
        # Measure execution time
        start_time = time.time()
        final_state = graph.invoke(state)
        duration = time.time() - start_time
        
        synthesis = final_state["final_synthesis"]
        
        # SC-T1-001: Execution time < 2 minutes
        assert duration < 120, f"Duration {duration:.2f}s exceeds 120s"
        print(f"\n✅ SC-T1-001: Execution time: {duration:.2f}s < 120s")
        
        # SC-T1-002: Confidence score >= 0.7
        assert synthesis.confidence_score >= 0.7, \
            f"Confidence {synthesis.confidence_score} < 0.7"
        print(f"✅ SC-T1-002: Confidence: {synthesis.confidence_score:.2f} >= 0.7")
        
        # SC-T1-003: Novelty score >= 0.3
        assert synthesis.novelty_score >= 0.3, \
            f"Novelty {synthesis.novelty_score} < 0.3"
        print(f"✅ SC-T1-003: Novelty: {synthesis.novelty_score:.2f} >= 0.3")
        
        # SC-T1-004: Evidence lineage >= 3 sources
        assert len(synthesis.evidence_lineage) >= 3, \
            f"Evidence count {len(synthesis.evidence_lineage)} < 3"
        print(f"✅ SC-T1-004: Evidence sources: {len(synthesis.evidence_lineage)} >= 3")
        
        print(f"\n🎉 All success criteria passed!")
        print(f"   Novel Insight: {synthesis.novel_insight[:100]}...")


class TestCyclicDebateFlow:
    """Test the cyclic debate mechanism with contradictions."""
    
    @pytest.mark.slow
    @pytest.mark.e2e
    def test_multiple_iteration_flow(self):
        """Test system handles multiple iterations when contradictions persist."""
        import os
        os.environ["MAX_ITERATIONS"] = "3"
        
        graph = build_graph()
        
        # Use a controversial query likely to trigger contradictions
        state: AgentState = {
            "messages": [],
            "original_query": "Are neural networks truly capable of reasoning like humans?",
            "current_thesis": None,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": "",
            "thread_id": "e2e_cyclic_001"
        }
        
        final_state = graph.invoke(state)
        
        # Verify system completed
        assert final_state["final_synthesis"] is not None
        
        # Check if iterations occurred
        iterations = final_state["iteration_count"]
        print(f"\n✅ Cyclic Debate - Iterations: {iterations}")
        
        # If no iterations, that's okay (no contradiction found)
        # If iterations occurred, verify within max
        assert 0 <= iterations <= 3
        
        # Verify quality maintained through iterations
        synthesis = final_state["final_synthesis"]
        assert synthesis.confidence_score >= 0.7
        assert synthesis.novelty_score >= 0.3


if __name__ == "__main__":
    # Run E2E tests with detailed output
    pytest.main([__file__, "-v", "-s", "-m", "e2e", "--durations=10"])

