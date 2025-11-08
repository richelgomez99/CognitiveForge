"""
Unit tests for Pydantic models and AgentState.

Tests validation rules, required fields, and schema correctness.
"""

import pytest
from pydantic import ValidationError
from src.models import Evidence, Thesis, Antithesis, Synthesis, AgentState


class TestEvidence:
    """Test Evidence Pydantic model."""
    
    def test_valid_evidence(self):
        """Test creating valid Evidence instance."""
        evidence = Evidence(
            source_url="https://arxiv.org/abs/1706.03762",
            snippet="Attention is all you need",
            relevance_score=0.9
        )
        assert evidence.source_url == "https://arxiv.org/abs/1706.03762"
        assert evidence.snippet == "Attention is all you need"
        assert evidence.relevance_score == 0.9
    
    def test_evidence_optional_relevance(self):
        """Test Evidence without optional relevance_score."""
        evidence = Evidence(
            source_url="https://example.com",
            snippet="Test snippet"
        )
        assert evidence.relevance_score is None
    
    def test_evidence_missing_required(self):
        """Test Evidence fails without required fields."""
        with pytest.raises(ValidationError):
            Evidence(source_url="https://example.com")  # Missing snippet


class TestThesis:
    """Test Thesis Pydantic model."""
    
    def test_valid_thesis(self):
        """Test creating valid Thesis with minimum evidence."""
        evidence = [
            Evidence(source_url="https://arxiv.org/abs/1", snippet="This is valid evidence snippet with sufficient length"),
            Evidence(source_url="https://arxiv.org/abs/2", snippet="This is another valid evidence snippet with sufficient length")
        ]
        thesis = Thesis(
            claim="Transformers are highly effective neural network architectures",
            reasoning="They use self-attention mechanisms which allow for better capture of long-range dependencies in sequential data",
            evidence=evidence
        )
        assert thesis.claim == "Transformers are highly effective neural network architectures"
        assert len(thesis.evidence) == 2
    
    def test_thesis_min_evidence_validation(self):
        """Test Thesis requires at least 2 evidence items."""
        with pytest.raises(ValidationError) as exc_info:
            Thesis(
                claim="This is a valid claim with sufficient length for testing",
                reasoning="This is valid reasoning with sufficient length to pass validation requirements",
                evidence=[Evidence(source_url="https://example.com", snippet="Only one piece of evidence here")]
            )
        assert "at least 2" in str(exc_info.value).lower()
    
    def test_thesis_missing_required(self):
        """Test Thesis fails without required fields."""
        with pytest.raises(ValidationError):
            Thesis(claim="Test", reasoning="Test")  # Missing evidence


class TestAntithesis:
    """Test Antithesis Pydantic model."""
    
    def test_antithesis_with_contradiction(self):
        """Test Antithesis when contradiction is found."""
        evidence = Evidence(source_url="https://example.com", snippet="Conflicting data")
        antithesis = Antithesis(
            contradiction_found=True,
            counter_claim="Actually, transformers have limitations",
            conflicting_evidence=[evidence],
            critique="The thesis overlooks computational cost"
        )
        assert antithesis.contradiction_found is True
        assert antithesis.counter_claim is not None
        assert len(antithesis.conflicting_evidence) == 1
    
    def test_antithesis_no_contradiction(self):
        """Test Antithesis when no contradiction found."""
        antithesis = Antithesis(
            contradiction_found=False,
            counter_claim=None,
            conflicting_evidence=[],
            critique="The thesis is well-supported with sufficient evidence and sound reasoning throughout"
        )
        assert antithesis.contradiction_found is False
        assert antithesis.counter_claim is None
    
    def test_antithesis_requires_critique(self):
        """Test Antithesis always requires critique."""
        with pytest.raises(ValidationError):
            Antithesis(
                contradiction_found=False,
                counter_claim=None,
                conflicting_evidence=None
                # Missing critique
            )


class TestSynthesis:
    """Test Synthesis Pydantic model."""
    
    def test_valid_synthesis(self):
        """Test creating valid Synthesis with all fields."""
        synthesis = Synthesis(
            novel_insight="Transformers represent a balanced approach that combines effectiveness with computational considerations for practical deployment",
            supporting_claims=["Claim 1", "Claim 2"],
            evidence_lineage=[
                "https://arxiv.org/abs/1",
                "https://arxiv.org/abs/2",
                "https://arxiv.org/abs/3"
            ],
            confidence_score=0.85,
            novelty_score=0.70,
            reasoning="This synthesis integrates insights from both the thesis and antithesis to arrive at a nuanced understanding"
        )
        assert len(synthesis.novel_insight) >= 50
        assert len(synthesis.evidence_lineage) >= 3
        assert 0.0 <= synthesis.confidence_score <= 1.0
        assert 0.0 <= synthesis.novelty_score <= 1.0
    
    def test_synthesis_min_evidence_lineage(self):
        """Test Synthesis requires at least 3 evidence URLs."""
        with pytest.raises(ValidationError):
            Synthesis(
                novel_insight="Test",
                supporting_claims=["Claim"],
                evidence_lineage=["url1", "url2"],  # Only 2, need 3
                confidence_score=0.8,
                novelty_score=0.7,
                reasoning="Test"
            )
    
    def test_synthesis_score_bounds(self):
        """Test Synthesis scores must be between 0.0 and 1.0."""
        # Test confidence_score > 1.0
        with pytest.raises(ValidationError):
            Synthesis(
                novel_insight="Test",
                supporting_claims=["Claim"],
                evidence_lineage=["url1", "url2", "url3"],
                confidence_score=1.5,  # Invalid
                novelty_score=0.7,
                reasoning="Test"
            )
        
        # Test novelty_score < 0.0
        with pytest.raises(ValidationError):
            Synthesis(
                novel_insight="Test",
                supporting_claims=["Claim"],
                evidence_lineage=["url1", "url2", "url3"],
                confidence_score=0.8,
                novelty_score=-0.1,  # Invalid
                reasoning="Test"
            )


class TestAgentState:
    """Test AgentState TypedDict structure."""
    
    def test_agent_state_creation(self):
        """Test creating a valid AgentState."""
        state: AgentState = {
            "messages": ["Message 1", "Message 2"],
            "original_query": "What are transformers?",
            "current_thesis": None,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 0,
            "procedural_memory": ""
        }
        assert state["original_query"] == "What are transformers?"
        assert state["iteration_count"] == 0
        assert len(state["messages"]) == 2
    
    def test_agent_state_with_thesis(self):
        """Test AgentState with populated thesis."""
        evidence = [
            Evidence(source_url="https://arxiv.org/abs/1", snippet="This is valid evidence with sufficient length"),
            Evidence(source_url="https://arxiv.org/abs/2", snippet="This is another valid evidence with sufficient length")
        ]
        thesis = Thesis(
            claim="This is a valid test claim with sufficient length for validation",
            reasoning="This is valid test reasoning with sufficient length to pass all validation requirements",
            evidence=evidence
        )
        
        state: AgentState = {
            "messages": [],
            "original_query": "Test query",
            "current_thesis": thesis,
            "current_antithesis": None,
            "final_synthesis": None,
            "contradiction_report": "",
            "iteration_count": 1,
            "procedural_memory": ""
        }
        
        assert state["current_thesis"] is not None
        assert len(state["current_thesis"].claim) >= 20
        assert state["iteration_count"] == 1


class TestModelSerialization:
    """Test JSON serialization/deserialization of models."""
    
    def test_evidence_json_roundtrip(self):
        """Test Evidence can be serialized and deserialized."""
        original = Evidence(
            source_url="https://example.com",
            snippet="Test snippet",
            relevance_score=0.9
        )
        
        # Serialize
        json_str = original.model_dump_json()
        
        # Deserialize
        reconstructed = Evidence.model_validate_json(json_str)
        
        assert reconstructed.source_url == original.source_url
        assert reconstructed.snippet == original.snippet
        assert reconstructed.relevance_score == original.relevance_score
    
    def test_synthesis_json_roundtrip(self):
        """Test Synthesis can be serialized and deserialized."""
        original = Synthesis(
            novel_insight="This is a comprehensive test insight with sufficient length to pass validation requirements",
            supporting_claims=["Claim 1", "Claim 2"],
            evidence_lineage=["url1", "url2", "url3"],
            confidence_score=0.85,
            novelty_score=0.70,
            reasoning="This is comprehensive test reasoning with sufficient length to pass all validation requirements"
        )
        
        # Serialize
        json_str = original.model_dump_json()
        
        # Deserialize
        reconstructed = Synthesis.model_validate_json(json_str)
        
        assert reconstructed.novel_insight == original.novel_insight
        assert reconstructed.confidence_score == original.confidence_score
        assert reconstructed.novelty_score == original.novelty_score
        assert len(reconstructed.evidence_lineage) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

