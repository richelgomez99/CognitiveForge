"""
Tests for final Epic 5 agents:
- Counter-Perspective Agent (Task 6)
- Novelty Assessor Agent (Task 7)
- Synthesis Reviewer Agent (Task 8)
"""

import pytest


# =============================================================================
# Counter-Perspective Agent Tests
# =============================================================================

def test_counter_perspective_creation():
    """Test creating Counter-Perspective agent."""
    from src.agents.counter_perspective import CounterPerspectiveAgent
    from src.agents.base_agent import AgentRole

    agent = CounterPerspectiveAgent()
    assert agent.get_role() == AgentRole.COUNTER_PERSPECTIVE


def test_counter_perspective_metadata():
    """Test Counter-Perspective metadata."""
    from src.agents.counter_perspective import CounterPerspectiveAgent
    from src.agents.base_agent import AgentRole, AgentPriority

    agent = CounterPerspectiveAgent()
    metadata = agent.get_metadata()

    assert metadata.role == AgentRole.COUNTER_PERSPECTIVE
    assert metadata.priority == AgentPriority.MEDIUM
    assert AgentRole.ANALYST in metadata.dependencies


def test_counter_perspective_generation():
    """Test counter-perspective generation."""
    from src.agents.counter_perspective import CounterPerspectiveAgent
    from src.models import Thesis, Evidence

    agent = CounterPerspectiveAgent()

    thesis = Thesis(
        claim="Neural networks ALWAYS achieve perfect accuracy on ALL datasets.",
        reasoning="Therefore, because neural networks are perfect, they always work.",
        evidence=[
            Evidence(source_url="https://example.com/p1", snippet="Evidence 1"),
            Evidence(source_url="https://example.com/p2", snippet="Evidence 2")
        ]
    )

    state = {"current_thesis": thesis}
    result = agent.execute(state)

    # Should generate counter-perspectives
    assert "counter_perspectives" in result
    perspectives = result["counter_perspectives"]
    assert len(perspectives) > 0
    assert all(hasattr(p, "perspective") for p in perspectives)


# =============================================================================
# Novelty Assessor Agent Tests
# =============================================================================

def test_novelty_assessor_creation():
    """Test creating Novelty Assessor agent."""
    from src.agents.novelty_assessor import NoveltyAssessorAgent
    from src.agents.base_agent import AgentRole

    agent = NoveltyAssessorAgent()
    assert agent.get_role() == AgentRole.NOVELTY_ASSESSOR


def test_novelty_assessor_metadata():
    """Test Novelty Assessor metadata."""
    from src.agents.novelty_assessor import NoveltyAssessorAgent
    from src.agents.base_agent import AgentRole, AgentPriority

    agent = NoveltyAssessorAgent()
    metadata = agent.get_metadata()

    assert metadata.role == AgentRole.NOVELTY_ASSESSOR
    assert metadata.priority == AgentPriority.LOW
    assert AgentRole.ANALYST in metadata.dependencies


def test_novelty_assessment():
    """Test novelty assessment."""
    from src.agents.novelty_assessor import NoveltyAssessorAgent
    from src.models import Thesis, Evidence

    agent = NoveltyAssessorAgent()

    thesis = Thesis(
        claim="This research proposes a novel approach combining transfer learning with active learning strategies.",
        reasoning="The combination of transfer learning and active learning creates a unique methodology that addresses data efficiency.",
        evidence=[
            Evidence(source_url="https://example.com/p1", snippet="Transfer learning research"),
            Evidence(source_url="https://example.com/p2", snippet="Active learning paper"),
            Evidence(source_url="https://example.com/p3", snippet="Combined approach study"),
            Evidence(source_url="https://example.com/p4", snippet="Novel methodology"),
            Evidence(source_url="https://example.com/p5", snippet="Empirical results")
        ]
    )

    curated_papers = [e.source_url for e in thesis.evidence]

    state = {
        "current_thesis": thesis,
        "curated_papers": curated_papers
    }

    result = agent.execute(state)

    # Should return novelty assessment
    assert "novelty_assessment" in result
    assessment = result["novelty_assessment"]
    assert 0.0 <= assessment.novelty_score <= 100.0
    assert len(assessment.innovation_factors) > 0
    assert 0.0 <= assessment.assessment_confidence <= 1.0


# =============================================================================
# Synthesis Reviewer Agent Tests
# =============================================================================

def test_synthesis_reviewer_creation():
    """Test creating Synthesis Reviewer agent."""
    from src.agents.synthesis_reviewer import SynthesisReviewerAgent
    from src.agents.base_agent import AgentRole

    agent = SynthesisReviewerAgent()
    assert agent.get_role() == AgentRole.SYNTHESIS_REVIEWER


def test_synthesis_reviewer_metadata():
    """Test Synthesis Reviewer metadata."""
    from src.agents.synthesis_reviewer import SynthesisReviewerAgent
    from src.agents.base_agent import AgentRole, AgentPriority

    agent = SynthesisReviewerAgent()
    metadata = agent.get_metadata()

    assert metadata.role == AgentRole.SYNTHESIS_REVIEWER
    assert metadata.priority == AgentPriority.CRITICAL
    assert AgentRole.SYNTHESIZER in metadata.dependencies
    assert metadata.can_run_parallel is False  # Must run last


def test_synthesis_review_pass():
    """Test synthesis review with passing quality."""
    from src.agents.synthesis_reviewer import SynthesisReviewerAgent
    from src.models import ValidationReport, ConsistencyCheck, BiasReport
    from unittest.mock import Mock

    agent = SynthesisReviewerAgent()

    # Use mock with required insight field
    synthesis = Mock()
    synthesis.insight = "Neural networks demonstrate strong performance on image classification tasks through hierarchical feature learning. The evidence supports this claim with high confidence."

    # High quality reports
    evidence_report = ValidationReport(
        alignments=[],
        overall_strength=0.8,
        strong_count=3,
        weak_count=0,
        recommendation="Strong support"
    )

    consistency_report = ConsistencyCheck(
        similar_past_claims=[],
        contradiction_found=False,
        consistency_score=0.9,
        recommendation="Consistent"
    )

    bias_report = BiasReport(
        detected_biases=[],
        bias_score=0.2,
        recommendation="Low bias"
    )

    state = {
        "final_synthesis": synthesis,
        "evidence_validation_report": evidence_report,
        "consistency_check_report": consistency_report,
        "bias_detection_report": bias_report
    }

    result = agent.execute(state)

    # Should pass QA
    review = result["synthesis_review"]
    assert review.qa_passed is True
    assert review.quality_score >= 70.0
    assert review.consistency_check is True
    assert review.evidence_check is True


def test_synthesis_review_fail():
    """Test synthesis review with failing quality."""
    from src.agents.synthesis_reviewer import SynthesisReviewerAgent
    from src.models import ValidationReport, ConsistencyCheck, BiasReport
    from unittest.mock import Mock

    agent = SynthesisReviewerAgent()

    # Use mock with short insight
    synthesis = Mock()
    synthesis.insight = "Short"

    # Low quality reports
    evidence_report = ValidationReport(
        alignments=[],
        overall_strength=0.3,  # Weak
        strong_count=0,
        weak_count=3,
        recommendation="Weak support"
    )

    consistency_report = ConsistencyCheck(
        similar_past_claims=[],
        contradiction_found=True,
        consistency_score=0.4,  # Low
        recommendation="Inconsistent"
    )

    bias_report = BiasReport(
        detected_biases=[],
        bias_score=0.8,  # High
        recommendation="High bias"
    )

    state = {
        "final_synthesis": synthesis,
        "evidence_validation_report": evidence_report,
        "consistency_check_report": consistency_report,
        "bias_detection_report": bias_report
    }

    result = agent.execute(state)

    # Should fail QA
    review = result["synthesis_review"]
    assert review.qa_passed is False
    assert len(review.issues_found) > 0


def test_synthesis_review_no_synthesis():
    """Test synthesis review with no synthesis."""
    from src.agents.synthesis_reviewer import SynthesisReviewerAgent

    agent = SynthesisReviewerAgent()
    state = {"final_synthesis": None}
    result = agent.execute(state)

    review = result["synthesis_review"]
    assert review.qa_passed is False
    assert review.quality_score == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
