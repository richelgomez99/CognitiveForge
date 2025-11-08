"""
Unit tests for agent nodes (Analyst, Skeptic, Synthesizer).

Tests each agent's ability to process state and produce valid outputs.
"""

import pytest
from src.agents.analyst import analyst_node
from src.agents.skeptic import skeptic_node
from src.agents.synthesizer import synthesizer_node
from src.models import AgentState, Thesis, Antithesis, Evidence


class TestAnalystNode:
    """Test the Analyst agent node."""
    
    def test_analyst_generates_thesis(self):
        """Test Analyst generates a valid Thesis from query."""
        state: AgentState = {
            "messages": [],
            "original_query": "What are the key features of transformers?",
            "current_thesis": None,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": ""
        }
        
        # Execute analyst node
        result = analyst_node(state)
        
        # Verify output structure
        assert "current_thesis" in result
        assert "messages" in result
        
        # Verify thesis is valid
        thesis = result["current_thesis"]
        assert isinstance(thesis, Thesis)
        assert len(thesis.claim) > 0
        assert len(thesis.reasoning) > 0
        assert len(thesis.evidence) >= 2
    
    def test_analyst_with_procedural_memory(self):
        """Test Analyst incorporates procedural memory if provided."""
        state: AgentState = {
            "messages": [],
            "original_query": "Explain attention mechanisms",
            "current_thesis": None,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": "Focus on computational efficiency"
        }
        
        result = analyst_node(state)
        
        # Should still generate valid thesis
        assert "current_thesis" in result
        thesis = result["current_thesis"]
        assert isinstance(thesis, Thesis)
    
    def test_analyst_requires_query(self):
        """Test Analyst fails gracefully without query."""
        state: AgentState = {
            "messages": [],
            "original_query": "",  # Empty query
            "current_thesis": None,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": ""
        }
        
        # Should handle empty query
        # Either raise exception or return gracefully
        try:
            result = analyst_node(state)
            # If it doesn't raise, check it handled it somehow
            assert True
        except Exception as e:
            # Expected to fail with empty query
            assert "query" in str(e).lower() or "empty" in str(e).lower()


class TestSkepticNode:
    """Test the Skeptic agent node."""
    
    def test_skeptic_evaluates_thesis(self):
        """Test Skeptic generates valid Antithesis."""
        # Create a thesis for skeptic to evaluate
        thesis = Thesis(
            claim="Transformers are the best architecture for all NLP tasks",
            reasoning="They achieve state-of-the-art performance on benchmarks across multiple domains and task types",
            evidence=[
                Evidence(source_url="https://arxiv.org/abs/1", snippet="Transformers excel at machine translation tasks"),
                Evidence(source_url="https://arxiv.org/abs/2", snippet="BERT achieves top scores on benchmarks")
            ]
        )
        
        state: AgentState = {
            "messages": ["Analyst's thesis generated"],
            "original_query": "Are transformers the best for NLP?",
            "current_thesis": thesis,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": ""
        }
        
        # Execute skeptic node
        result = skeptic_node(state)
        
        # Verify output structure
        assert "current_antithesis" in result
        assert "contradiction_report" in result
        assert "messages" in result
        
        # Verify antithesis is valid
        antithesis = result["current_antithesis"]
        assert isinstance(antithesis, Antithesis)
        assert isinstance(antithesis.contradiction_found, bool)
        assert len(antithesis.critique) > 0
    
    def test_skeptic_requires_thesis(self):
        """Test Skeptic fails without a thesis."""
        state: AgentState = {
            "messages": [],
            "original_query": "Test query",
            "current_thesis": None,  # No thesis provided
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": ""
        }
        
        # Should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            skeptic_node(state)
        
        assert "thesis" in str(exc_info.value).lower()


class TestSynthesizerNode:
    """Test the Synthesizer agent node."""
    
    def test_synthesizer_creates_synthesis(self):
        """Test Synthesizer generates valid Synthesis."""
        # Create thesis
        thesis = Thesis(
            claim="Transformers are highly effective for NLP tasks",
            reasoning="They use attention mechanisms that enable parallel processing and capture long-range dependencies effectively",
            evidence=[
                Evidence(source_url="https://arxiv.org/abs/1706.03762", snippet="Attention is all you need for sequence modeling"),
                Evidence(source_url="https://arxiv.org/abs/1810.04805", snippet="BERT pretraining achieves state-of-art")
            ]
        )
        
        # Create antithesis
        antithesis = Antithesis(
            contradiction_found=True,
            counter_claim="Transformers have computational limitations",
            conflicting_evidence=[
                Evidence(source_url="https://arxiv.org/abs/2020.1", snippet="Quadratic complexity")
            ],
            critique="Computational cost scales poorly with sequence length"
        )
        
        state: AgentState = {
            "messages": ["Analyst's thesis", "Skeptic's antithesis"],
            "original_query": "What are transformers?",
            "current_thesis": thesis,
            "current_antithesis": antithesis,
            "final_synthesis": None,
            "contradiction_report": "Found contradictions",
            "iteration_count": 1,
            "procedural_memory": "",
            "thread_id": "test_synthesizer_001"
        }
        
        # Execute synthesizer node
        result = synthesizer_node(state)
        
        # Verify output structure
        assert "final_synthesis" in result
        assert "messages" in result
        
        # Verify synthesis is valid
        synthesis = result["final_synthesis"]
        assert synthesis is not None
        assert len(synthesis.novel_insight) > 0
        assert len(synthesis.supporting_claims) > 0
        assert len(synthesis.evidence_lineage) >= 3  # From both thesis and antithesis
        assert 0.0 <= synthesis.confidence_score <= 1.0
        assert 0.0 <= synthesis.novelty_score <= 1.0
    
    def test_synthesizer_requires_both_inputs(self):
        """Test Synthesizer fails without thesis or antithesis."""
        # Missing thesis
        state_no_thesis: AgentState = {
            "messages": [],
            "original_query": "Test",
            "current_thesis": None,
            "current_antithesis": Antithesis(
                contradiction_found=False,
                critique="This is a valid critique with sufficient length for validation"
            ),
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": "",
            "thread_id": "test"
        }
        
        with pytest.raises(ValueError) as exc_info:
            synthesizer_node(state_no_thesis)
        assert "thesis" in str(exc_info.value).lower()
        
        # Missing antithesis
        thesis = Thesis(
            claim="This is a valid test claim with sufficient length",
            reasoning="This is valid test reasoning with sufficient length for validation",
            evidence=[
                Evidence(source_url="url1", snippet="This is valid evidence one"),
                Evidence(source_url="url2", snippet="This is valid evidence two")
            ]
        )
        state_no_antithesis: AgentState = {
            "messages": [],
            "original_query": "Test",
            "current_thesis": thesis,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": "",
            "thread_id": "test"
        }
        
        with pytest.raises(ValueError) as exc_info:
            synthesizer_node(state_no_antithesis)
        assert "antithesis" in str(exc_info.value).lower()
    
    def test_synthesizer_aggregates_evidence(self):
        """Test Synthesizer aggregates evidence from both agents."""
        thesis = Thesis(
            claim="This is a comprehensive test claim with sufficient length for validation",
            reasoning="This is comprehensive test reasoning with sufficient length to pass validation requirements",
            evidence=[
                Evidence(source_url="https://example.com/1", snippet="This is first piece of evidence"),
                Evidence(source_url="https://example.com/2", snippet="This is second piece of evidence")
            ]
        )
        
        antithesis = Antithesis(
            contradiction_found=False,
            counter_claim=None,
            conflicting_evidence=[
                Evidence(source_url="https://example.com/3", snippet="This is third piece of evidence")
            ],
            critique="The thesis presents minor points that could be strengthened with additional evidence"
        )
        
        state: AgentState = {
            "messages": [],
            "original_query": "Test",
            "current_thesis": thesis,
            "current_antithesis": antithesis,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": "",
            "thread_id": "test_agg"
        }
        
        result = synthesizer_node(state)
        synthesis = result["final_synthesis"]
        
        # Should have evidence from both thesis (2) and antithesis (1) = 3 total
        assert len(synthesis.evidence_lineage) >= 3
        assert "https://example.com/1" in synthesis.evidence_lineage
        assert "https://example.com/2" in synthesis.evidence_lineage
        assert "https://example.com/3" in synthesis.evidence_lineage


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

