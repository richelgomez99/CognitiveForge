"""
Tests for Bias Detector Agent (Epic 5 Task 4)
"""

import pytest
from typing import Dict, Any


def test_bias_detector_creation():
    """Test creating a Bias Detector agent."""
    from src.agents.bias_detector import BiasDetectorAgent
    from src.agents.base_agent import AgentRole

    agent = BiasDetectorAgent()

    assert agent.get_role() == AgentRole.BIAS_DETECTOR
    assert agent.bias_threshold == 0.6


def test_bias_detector_metadata():
    """Test Bias Detector metadata."""
    from src.agents.bias_detector import BiasDetectorAgent
    from src.agents.base_agent import AgentRole, AgentPriority

    agent = BiasDetectorAgent()
    metadata = agent.get_metadata()

    assert metadata.role == AgentRole.BIAS_DETECTOR
    assert metadata.priority == AgentPriority.HIGH
    assert AgentRole.ANALYST in metadata.dependencies
    assert AgentRole.PAPER_CURATOR in metadata.dependencies
    assert metadata.can_run_parallel is True


def test_bias_detector_no_thesis():
    """Test Bias Detector with no thesis."""
    from src.agents.bias_detector import BiasDetectorAgent

    agent = BiasDetectorAgent()

    state = {
        "current_thesis": None,
        "curated_papers": [],
        "discovered_papers": []
    }

    result = agent.execute(state)

    assert "bias_detection_report" in result
    report = result["bias_detection_report"]
    assert len(report.detected_biases) == 0


def test_bias_detector_selection_bias():
    """Test detection of selection bias (high filtering rate)."""
    from src.agents.bias_detector import BiasDetectorAgent
    from src.models import Thesis, Evidence

    agent = BiasDetectorAgent()

    thesis = Thesis(
        claim="Neural networks are highly effective for all tasks.",
        reasoning="Studies show that neural networks outperform other methods in various applications consistently.",
        evidence=[
            Evidence(source_url="https://example.com/paper1", snippet="Neural networks show strong performance"),
            Evidence(source_url="https://example.com/paper2", snippet="Deep learning achieves good results")
        ]
    )

    # Simulate high filtering: 3 selected from 15 discovered (80% filtered)
    discovered_papers = [
        {"title": f"Paper {i}", "abstract": f"Abstract {i}", "url": f"https://example.com/paper{i}"}
        for i in range(1, 16)
    ]

    curated_papers = ["https://example.com/paper1", "https://example.com/paper2", "https://example.com/paper3"]

    state = {
        "current_thesis": thesis,
        "curated_papers": curated_papers,
        "discovered_papers": discovered_papers,
        "curation_metadata": None
    }

    result = agent.execute(state)
    report = result["bias_detection_report"]

    # Should detect selection bias
    bias_types = [b.bias_type for b in report.detected_biases]
    assert "selection" in bias_types


def test_bias_detector_confirmation_bias():
    """Test detection of confirmation bias (no counter-evidence)."""
    from src.agents.bias_detector import BiasDetectorAgent
    from src.models import Thesis, Evidence

    agent = BiasDetectorAgent()

    thesis = Thesis(
        claim="Neural networks always work perfectly on all datasets.",
        reasoning="Research consistently demonstrates that neural networks achieve excellent performance across all domains.",
        evidence=[
            Evidence(source_url="https://example.com/paper1", snippet="Neural networks perform well"),
            Evidence(source_url="https://example.com/paper2", snippet="Excellent results achieved")
        ]
    )

    # All papers are purely positive (no critical terms)
    papers = [
        {
            "title": f"Neural Networks Are Great {i}",
            "abstract": f"Our study shows neural networks perform excellently on dataset {i}.",
            "url": f"https://example.com/paper{i}"
        }
        for i in range(1, 6)
    ]

    state = {
        "current_thesis": thesis,
        "curated_papers": [p["url"] for p in papers],
        "discovered_papers": papers
    }

    result = agent.execute(state)
    report = result["bias_detection_report"]

    # Should detect confirmation bias
    bias_types = [b.bias_type for b in report.detected_biases]
    assert "confirmation" in bias_types


def test_bias_detector_overgeneralization():
    """Test detection of overgeneralization bias."""
    from src.agents.bias_detector import BiasDetectorAgent
    from src.models import Thesis, Evidence

    agent = BiasDetectorAgent()

    # Thesis with absolute language
    thesis = Thesis(
        claim="Neural networks ALWAYS achieve perfect accuracy on ALL datasets.",
        reasoning="Studies universally demonstrate that neural networks NEVER fail and invariably outperform other methods.",
        evidence=[
            Evidence(source_url="https://example.com/paper1", snippet="Neural networks work well"),
            Evidence(source_url="https://example.com/paper2", snippet="Good performance observed")
        ]
    )

    # Limited evidence (only 2 papers)
    papers = [
        {"title": "Paper 1", "abstract": "Neural networks work well", "url": "https://example.com/paper1"},
        {"title": "Paper 2", "abstract": "Good results", "url": "https://example.com/paper2"}
    ]

    state = {
        "current_thesis": thesis,
        "curated_papers": [p["url"] for p in papers],
        "discovered_papers": papers
    }

    result = agent.execute(state)
    report = result["bias_detection_report"]

    # Should detect overgeneralization
    bias_types = [b.bias_type for b in report.detected_biases]
    assert "overgeneralization" in bias_types


def test_bias_detector_cherry_picking():
    """Test detection of cherry-picking bias (all papers from same source)."""
    from src.agents.bias_detector import BiasDetectorAgent
    from src.models import Thesis, Evidence

    agent = BiasDetectorAgent()

    thesis = Thesis(
        claim="Neural networks demonstrate strong performance capabilities.",
        reasoning="Multiple studies from arXiv show that neural networks achieve high accuracy on benchmark tasks.",
        evidence=[
            Evidence(source_url="https://arxiv.org/abs/1111.1111", snippet="Neural networks perform well"),
            Evidence(source_url="https://arxiv.org/abs/2222.2222", snippet="Strong results achieved")
        ]
    )

    # All papers from arXiv
    papers = [
        {"title": f"Paper {i}", "abstract": f"Abstract {i}", "url": f"https://arxiv.org/abs/{i}{i}{i}{i}.{i}{i}{i}{i}"}
        for i in range(1, 4)
    ]

    state = {
        "current_thesis": thesis,
        "curated_papers": [p["url"] for p in papers],
        "discovered_papers": papers
    }

    result = agent.execute(state)
    report = result["bias_detection_report"]

    # Should detect cherry-picking
    bias_types = [b.bias_type for b in report.detected_biases]
    assert "cherry_picking" in bias_types


def test_bias_detector_no_biases():
    """Test that no biases are detected for balanced analysis."""
    from src.agents.bias_detector import BiasDetectorAgent
    from src.models import Thesis, Evidence

    agent = BiasDetectorAgent()

    thesis = Thesis(
        claim="Neural networks can be effective for certain types of tasks.",
        reasoning="Studies suggest that neural networks show promise in specific domains, though limitations exist in others.",
        evidence=[
            Evidence(source_url="https://example.com/paper1", snippet="Neural networks show promise"),
            Evidence(source_url="https://nature.com/paper2", snippet="Some limitations observed")
        ]
    )

    # Diverse papers with critical language
    papers = [
        {
            "title": "Neural Networks: Pros and Cons",
            "abstract": "Neural networks show promise. However, there are limitations in certain scenarios.",
            "url": "https://example.com/paper1"
        },
        {
            "title": "Critical Analysis of Deep Learning",
            "abstract": "While neural networks achieve good results, contrary to popular belief, they have challenges.",
            "url": "https://nature.com/paper2"
        },
        {
            "title": "Alternative Approaches",
            "abstract": "Neural networks are effective but alternative methods may be better in some cases.",
            "url": "https://ieee.org/paper3"
        },
        {
            "title": "Limitations of Neural Networks",
            "abstract": "This study questions the universal applicability of neural networks.",
            "url": "https://acm.org/paper4"
        },
        {
            "title": "Balanced View",
            "abstract": "Neural networks have strengths and weaknesses that should be considered.",
            "url": "https://springer.com/paper5"
        }
    ]

    state = {
        "current_thesis": thesis,
        "curated_papers": [p["url"] for p in papers],
        "discovered_papers": papers[:6]  # Low filtering rate
    }

    result = agent.execute(state)
    report = result["bias_detection_report"]

    # Should have low bias score
    assert report.bias_score < 0.6
    assert "balanced" in report.recommendation.lower() or "no significant" in report.recommendation.lower()


def test_bias_detector_high_bias_score():
    """Test high bias score calculation."""
    from src.agents.bias_detector import BiasDetectorAgent
    from src.models import Thesis, Evidence

    agent = BiasDetectorAgent()

    # Thesis with multiple bias indicators
    thesis = Thesis(
        claim="Neural networks ALWAYS achieve perfect results on ALL datasets without exception.",
        reasoning="Studies universally and invariably demonstrate that neural networks NEVER fail under any circumstances.",
        evidence=[
            Evidence(source_url="https://arxiv.org/abs/1111.1111", snippet="Perfect results"),
            Evidence(source_url="https://arxiv.org/abs/2222.2222", snippet="Flawless performance")
        ]
    )

    # High filtering + all positive + same source
    discovered_papers = [
        {"title": f"Paper {i}", "abstract": f"Positive results {i}", "url": f"https://arxiv.org/abs/{i}{i}{i}{i}.{i}{i}{i}{i}"}
        for i in range(1, 16)
    ]

    curated_papers = discovered_papers[:2]

    state = {
        "current_thesis": thesis,
        "curated_papers": [p["url"] for p in curated_papers],
        "discovered_papers": discovered_papers
    }

    result = agent.execute(state)
    report = result["bias_detection_report"]

    # Should have high bias score
    assert report.bias_score > 0.6
    assert len(report.detected_biases) >= 2


def test_bias_detector_run_method():
    """Test running Bias Detector with hooks."""
    from src.agents.bias_detector import BiasDetectorAgent
    from src.models import Thesis, Evidence

    agent = BiasDetectorAgent()

    thesis = Thesis(
        claim="Neural networks show potential for various applications.",
        reasoning="Research indicates that neural networks can be effective in certain contexts and domains.",
        evidence=[
            Evidence(source_url="https://example.com/paper1", snippet="Neural networks show effectiveness"),
            Evidence(source_url="https://example.com/paper2", snippet="Positive results in experiments")
        ]
    )

    papers = [
        {"title": "Paper 1", "abstract": "Study of neural networks", "url": "https://example.com/paper1"}
    ]

    state = {
        "current_thesis": thesis,
        "curated_papers": [p["url"] for p in papers],
        "discovered_papers": papers
    }

    # Use run() to test hooks
    result = agent.run(state)

    assert "bias_detection_report" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
