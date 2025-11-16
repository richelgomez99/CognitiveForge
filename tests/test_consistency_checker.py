"""
Tests for Consistency Checker Agent (Epic 5 Task 5)
"""

import pytest


def test_consistency_checker_creation():
    """Test creating a Consistency Checker agent."""
    from src.agents.consistency_checker import ConsistencyCheckerAgent
    from src.agents.base_agent import AgentRole

    agent = ConsistencyCheckerAgent()
    assert agent.get_role() == AgentRole.CONSISTENCY_CHECKER


def test_consistency_checker_metadata():
    """Test Consistency Checker metadata."""
    from src.agents.consistency_checker import ConsistencyCheckerAgent
    from src.agents.base_agent import AgentRole, AgentPriority

    agent = ConsistencyCheckerAgent()
    metadata = agent.get_metadata()

    assert metadata.role == AgentRole.CONSISTENCY_CHECKER
    assert metadata.priority == AgentPriority.MEDIUM
    assert AgentRole.ANALYST in metadata.dependencies
    assert AgentRole.EVIDENCE_VALIDATOR in metadata.dependencies


def test_consistency_checker_no_thesis():
    """Test Consistency Checker with no thesis."""
    from src.agents.consistency_checker import ConsistencyCheckerAgent

    agent = ConsistencyCheckerAgent()
    state = {"current_thesis": None}
    result = agent.execute(state)

    assert "consistency_check_report" in result
    report = result["consistency_check_report"]
    assert report.contradiction_found is False
    assert report.consistency_score >= 0.0


def test_consistency_checker_consistent_thesis():
    """Test with consistent thesis (no issues)."""
    from src.agents.consistency_checker import ConsistencyCheckerAgent
    from src.models import Thesis, Evidence, ValidationReport

    agent = ConsistencyCheckerAgent()

    thesis = Thesis(
        claim="Neural networks achieve high accuracy on image classification because of hierarchical feature learning.",
        reasoning="Therefore, neural networks learn hierarchical features from images, which leads to high accuracy on classification tasks. This is because the network progressively learns more abstract representations.",
        evidence=[
            Evidence(source_url="https://example.com/p1", snippet="Neural networks learn features hierarchically"),
            Evidence(source_url="https://example.com/p2", snippet="High accuracy achieved on ImageNet classification")
        ]
    )

    validation_report = ValidationReport(
        alignments=[],
        overall_strength=0.8,
        strong_count=2,
        weak_count=0,
        recommendation="Strong support"
    )

    state = {
        "current_thesis": thesis,
        "evidence_validation_report": validation_report
    }

    result = agent.execute(state)
    report = result["consistency_check_report"]

    # Should be consistent
    assert report.contradiction_found is False
    assert report.consistency_score > 0.6


def test_consistency_checker_internal_inconsistency():
    """Test detection of internal inconsistency."""
    from src.agents.consistency_checker import ConsistencyCheckerAgent
    from src.models import Thesis, Evidence

    agent = ConsistencyCheckerAgent()

    # Claim about neural networks, but reasoning talks about cooking
    thesis = Thesis(
        claim="Neural networks achieve high accuracy on image classification tasks.",
        reasoning="Cooking pasta requires boiling water. The temperature must be high enough. Salt enhances flavor.",
        evidence=[
            Evidence(source_url="https://example.com/p1", snippet="Relevant evidence 1"),
            Evidence(source_url="https://example.com/p2", snippet="Relevant evidence 2")
        ]
    )

    state = {"current_thesis": thesis}
    result = agent.execute(state)
    report = result["consistency_check_report"]

    # Should detect inconsistency (low score)
    assert report.consistency_score < 1.0


def test_consistency_checker_evidence_inconsistency():
    """Test detection of evidence-claim inconsistency."""
    from src.agents.consistency_checker import ConsistencyCheckerAgent
    from src.models import Thesis, Evidence, ValidationReport, EvidenceAlignment

    agent = ConsistencyCheckerAgent()

    thesis = Thesis(
        claim="Neural networks always achieve perfect accuracy on all datasets.",
        reasoning="Therefore neural networks consistently achieve perfect results because they are flawless.",
        evidence=[
            Evidence(source_url="https://example.com/p1", snippet="Some evidence"),
            Evidence(source_url="https://example.com/p2", snippet="More evidence")
        ]
    )

    # Weak validation report
    validation_report = ValidationReport(
        alignments=[
            EvidenceAlignment(
                evidence_url="https://example.com/p1",
                claim_excerpt="perfect accuracy",
                alignment_score=0.3,
                alignment_category="weak",
                reasoning="Weak support"
            )
        ],
        overall_strength=0.3,
        strong_count=0,
        weak_count=1,
        recommendation="Weak support"
    )

    state = {
        "current_thesis": thesis,
        "evidence_validation_report": validation_report
    }

    result = agent.execute(state)
    report = result["consistency_check_report"]

    # Should detect evidence inconsistency (low score)
    assert report.consistency_score <= 0.8


def test_consistency_checker_reasoning_coherence():
    """Test detection of poor reasoning coherence."""
    from src.agents.consistency_checker import ConsistencyCheckerAgent
    from src.models import Thesis, Evidence

    agent = ConsistencyCheckerAgent()

    # Reasoning without logical connectors
    thesis = Thesis(
        claim="Neural networks are effective for classification tasks.",
        reasoning="Neural networks have layers. Deep learning uses gradients. Models need data. Training takes time. Results vary across datasets. Architectures differ.",
        evidence=[
            Evidence(source_url="https://example.com/p1", snippet="Neural network evidence"),
            Evidence(source_url="https://example.com/p2", snippet="Classification results")
        ]
    )

    state = {"current_thesis": thesis}
    result = agent.execute(state)
    report = result["consistency_check_report"]

    # Coherence check runs and returns a score
    assert isinstance(report.consistency_score, float)
    assert 0.0 <= report.consistency_score <= 1.0


def test_consistency_checker_logical_contradictions():
    """Test detection of logical contradictions."""
    from src.agents.consistency_checker import ConsistencyCheckerAgent
    from src.models import Thesis, Evidence

    agent = ConsistencyCheckerAgent()

    # Contradictory statements
    thesis = Thesis(
        claim="Neural networks always achieve perfect results but never work correctly.",
        reasoning="The model increases accuracy while simultaneously decreases performance. It improves results but also worsens outcomes.",
        evidence=[
            Evidence(source_url="https://example.com/p1", snippet="Contradictory evidence 1"),
            Evidence(source_url="https://example.com/p2", snippet="Contradictory evidence 2")
        ]
    )

    state = {"current_thesis": thesis}
    result = agent.execute(state)
    report = result["consistency_check_report"]

    # Should detect contradictions
    assert report.contradiction_found is True


def test_consistency_checker_score_calculation():
    """Test consistency score calculation."""
    from src.agents.consistency_checker import ConsistencyCheckerAgent

    agent = ConsistencyCheckerAgent()

    # Test with different issue counts
    report_0_issues = agent._create_report([])
    assert report_0_issues.consistency_score == 1.0
    assert report_0_issues.contradiction_found is False

    report_2_issues = agent._create_report(["Issue 1", "Issue 2"])
    assert 0.0 < report_2_issues.consistency_score < 1.0

    report_5_issues = agent._create_report(["I1", "I2", "I3", "I4", "I5"])
    assert report_5_issues.consistency_score == 0.0

    # Test contradiction detection
    report_with_contradiction = agent._create_report(["Issue with contradiction in text"])
    assert report_with_contradiction.contradiction_found is True


def test_consistency_checker_run_method():
    """Test running Consistency Checker with hooks."""
    from src.agents.consistency_checker import ConsistencyCheckerAgent
    from src.models import Thesis, Evidence

    agent = ConsistencyCheckerAgent()

    thesis = Thesis(
        claim="Neural networks show promise for various machine learning applications.",
        reasoning="Therefore, given that neural networks learn from data, they can be effective for tasks like classification and regression.",
        evidence=[
            Evidence(source_url="https://example.com/p1", snippet="Neural networks effective for classification"),
            Evidence(source_url="https://example.com/p2", snippet="Regression tasks benefit from neural networks")
        ]
    )

    state = {"current_thesis": thesis}
    result = agent.run(state)

    assert "consistency_check_report" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
