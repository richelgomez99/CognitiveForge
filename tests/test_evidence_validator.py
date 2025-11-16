"""
Tests for Evidence Validator Agent (Epic 5 Task 3)
"""

import pytest
from typing import Dict, Any


def test_evidence_validator_creation():
    """Test creating an Evidence Validator agent."""
    from src.agents.evidence_validator import EvidenceValidatorAgent
    from src.agents.base_agent import AgentRole

    agent = EvidenceValidatorAgent()

    assert agent.get_role() == AgentRole.EVIDENCE_VALIDATOR
    assert agent.alignment_threshold == 0.6


def test_evidence_validator_metadata():
    """Test Evidence Validator metadata."""
    from src.agents.evidence_validator import EvidenceValidatorAgent
    from src.agents.base_agent import AgentRole, AgentPriority

    agent = EvidenceValidatorAgent()
    metadata = agent.get_metadata()

    assert metadata.role == AgentRole.EVIDENCE_VALIDATOR
    assert metadata.priority == AgentPriority.HIGH
    assert AgentRole.ANALYST in metadata.dependencies
    assert AgentRole.PAPER_CURATOR in metadata.dependencies
    assert metadata.can_run_parallel is True  # Can run parallel with Bias Detector


def test_evidence_validator_no_thesis():
    """Test Evidence Validator with no thesis."""
    from src.agents.evidence_validator import EvidenceValidatorAgent

    agent = EvidenceValidatorAgent()

    state = {
        "current_thesis": None,
        "curated_papers": [],
        "discovered_papers": []
    }

    result = agent.execute(state)

    assert "evidence_validation_report" in result
    report = result["evidence_validation_report"]
    assert len(report.alignments) == 0


def test_evidence_validator_no_papers():
    """Test Evidence Validator with thesis but no papers."""
    from src.agents.evidence_validator import EvidenceValidatorAgent
    from src.models import Thesis, Evidence

    agent = EvidenceValidatorAgent()

    thesis = Thesis(
        claim="Neural networks can generalize to unseen data effectively.",
        reasoning="Deep learning models learn feature representations that transfer across different domains and datasets.",
        evidence=[
            Evidence(source_url="https://example.com/paper1", snippet="Neural networks show good generalization"),
            Evidence(source_url="https://example.com/paper2", snippet="Transfer learning enables generalization")
        ]
    )

    state = {
        "current_thesis": thesis,
        "curated_papers": [],
        "discovered_papers": []
    }

    result = agent.execute(state)

    report = result["evidence_validation_report"]
    assert report.overall_strength == 0.0
    assert "No evidence" in report.recommendation


def test_evidence_validator_strong_alignment():
    """Test Evidence Validator with strong evidence alignment."""
    from src.agents.evidence_validator import EvidenceValidatorAgent
    from src.models import Thesis, Evidence

    agent = EvidenceValidatorAgent()

    thesis = Thesis(
        claim="Neural networks achieve high accuracy on image classification tasks.",
        reasoning="Deep convolutional networks learn hierarchical features from images through multiple layers of processing.",
        evidence=[
            Evidence(source_url="https://example.com/cnn", snippet="CNNs achieve 95% accuracy on ImageNet classification"),
            Evidence(source_url="https://example.com/resnet", snippet="ResNet architectures improve image recognition performance")
        ]
    )

    papers = [
        {
            "title": "Deep Neural Networks for Image Classification",
            "abstract": "We demonstrate that neural networks achieve state-of-the-art accuracy on ImageNet classification tasks using convolutional architectures.",
            "url": "https://arxiv.org/abs/1234.5678",
            "citationCount": 500,
            "year": 2023
        }
    ]

    state = {
        "current_thesis": thesis,
        "curated_papers": ["https://arxiv.org/abs/1234.5678"],
        "discovered_papers": papers
    }

    result = agent.execute(state)

    report = result["evidence_validation_report"]

    # Should have alignments
    assert len(report.alignments) > 0

    # Should have strong alignment
    assert report.strong_count > 0
    assert report.overall_strength > 0.5


def test_evidence_validator_weak_alignment():
    """Test Evidence Validator with weak evidence alignment."""
    from src.agents.evidence_validator import EvidenceValidatorAgent
    from src.models import Thesis, Evidence

    agent = EvidenceValidatorAgent()

    thesis = Thesis(
        claim="Neural networks are the best machine learning models available.",
        reasoning="They outperform all other approaches in most machine learning tasks across different domains and datasets.",
        evidence=[
            Evidence(source_url="https://example.com/nn1", snippet="Neural networks show strong performance on benchmarks"),
            Evidence(source_url="https://example.com/nn2", snippet="Deep learning achieves state-of-the-art results")
        ]
    )

    # Unrelated paper
    papers = [
        {
            "title": "Support Vector Machines for Classification",
            "abstract": "We study SVM algorithms for binary classification tasks.",
            "url": "https://example.com/svm",
            "citationCount": 100,
            "year": 2020
        }
    ]

    state = {
        "current_thesis": thesis,
        "curated_papers": ["https://example.com/svm"],
        "discovered_papers": papers
    }

    result = agent.execute(state)

    report = result["evidence_validation_report"]

    # Should have low overall strength
    assert report.overall_strength < 0.5


def test_evidence_validator_extract_claims():
    """Test claim extraction from thesis."""
    from src.agents.evidence_validator import EvidenceValidatorAgent
    from src.models import Thesis, Evidence

    agent = EvidenceValidatorAgent()

    thesis = Thesis(
        claim="Main claim about neural networks and their effectiveness",
        reasoning="Research shows that deep learning works well in practice. Studies demonstrate effectiveness across multiple domains.",
        evidence=[
            Evidence(source_url="https://example.com/research1", snippet="Deep learning shows promising results"),
            Evidence(source_url="https://example.com/research2", snippet="Studies confirm neural network effectiveness")
        ]
    )

    claims = agent._extract_claims(thesis)

    # Should extract main claim + assertions from reasoning
    assert len(claims) >= 1
    assert "Main claim about neural networks and their effectiveness" in claims


def test_evidence_validator_keyword_extraction():
    """Test keyword extraction."""
    from src.agents.evidence_validator import EvidenceValidatorAgent

    agent = EvidenceValidatorAgent()

    keywords = agent._extract_keywords(
        "neural networks and deep learning for image classification"
    )

    assert "neural" in keywords
    assert "networks" in keywords
    assert "deep" in keywords
    assert "learning" in keywords
    assert "image" in keywords
    assert "classification" in keywords

    # Stopwords should be filtered
    assert "and" not in keywords
    assert "for" not in keywords


def test_evidence_validator_alignment_calculation():
    """Test alignment score calculation."""
    from src.agents.evidence_validator import EvidenceValidatorAgent

    agent = EvidenceValidatorAgent(alignment_threshold=0.6)

    # Strong alignment
    score, category, reasoning = agent._calculate_alignment(
        claim="Neural networks learn representations from data",
        paper_title="Deep Learning and Neural Network Representations",
        paper_abstract="We study how neural networks learn feature representations from training data."
    )

    assert score > 0.5
    assert category in ["strong", "moderate"]

    # Weak alignment
    score_weak, category_weak, reasoning_weak = agent._calculate_alignment(
        claim="Neural networks are used for classification",
        paper_title="Random Forests for Regression Tasks",
        paper_abstract="Random forest algorithms are effective for regression problems."
    )

    assert score_weak < score  # Should be weaker


def test_evidence_validator_contradictory_detection():
    """Test detection of contradictory evidence."""
    from src.agents.evidence_validator import EvidenceValidatorAgent

    agent = EvidenceValidatorAgent()

    # Paper with contradictory language
    score, category, reasoning = agent._calculate_alignment(
        claim="Neural networks always work well",
        paper_title="Neural Network Limitations",
        paper_abstract="However, neural networks often fail on small datasets. Contrary to popular belief, they refute the assumption of universal applicability."
    )

    # Should detect contradictory evidence
    assert "contradictory" in category.lower() or "contradictory" in reasoning.lower()


def test_evidence_validator_report_creation():
    """Test validation report creation."""
    from src.agents.evidence_validator import EvidenceValidatorAgent
    from src.models import EvidenceAlignment

    agent = EvidenceValidatorAgent()

    alignments = [
        EvidenceAlignment(
            evidence_url="https://example.com/paper1",
            claim_excerpt="Neural networks work well",
            alignment_score=0.85,
            alignment_category="strong",
            reasoning="Strong support"
        ),
        EvidenceAlignment(
            evidence_url="https://example.com/paper2",
            claim_excerpt="Deep learning is effective",
            alignment_score=0.75,
            alignment_category="strong",
            reasoning="Strong support"
        ),
        EvidenceAlignment(
            evidence_url="https://example.com/paper3",
            claim_excerpt="Models generalize",
            alignment_score=0.35,
            alignment_category="weak",
            reasoning="Weak support"
        )
    ]

    report = agent._create_report(alignments)

    assert report.strong_count == 2
    assert report.weak_count == 1
    assert report.overall_strength > 0.6
    assert "strong" in report.recommendation.lower()


def test_evidence_validator_run_method():
    """Test running Evidence Validator with hooks."""
    from src.agents.evidence_validator import EvidenceValidatorAgent
    from src.models import Thesis, Evidence

    agent = EvidenceValidatorAgent()

    thesis = Thesis(
        claim="Test claim about neural network performance and effectiveness",
        reasoning="Test reasoning that explains how neural networks perform well on various tasks and datasets.",
        evidence=[
            Evidence(source_url="https://example.com/test1", snippet="Neural networks perform effectively on test data"),
            Evidence(source_url="https://example.com/test2", snippet="Empirical results show strong performance metrics")
        ]
    )

    papers = [
        {
            "title": "Test Paper",
            "abstract": "Test abstract with claim and reasoning keywords.",
            "url": "https://example.com/test",
            "citationCount": 10,
            "year": 2023
        }
    ]

    state = {
        "current_thesis": thesis,
        "curated_papers": ["https://example.com/test"],
        "discovered_papers": papers
    }

    # Use run() to test hooks
    result = agent.run(state)

    assert "evidence_validation_report" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
