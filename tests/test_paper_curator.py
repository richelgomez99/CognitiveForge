"""
Tests for Paper Curator Agent (Epic 5 Task 2)
"""

import pytest
from typing import Dict, Any


def test_paper_curator_agent_creation():
    """Test creating a Paper Curator agent."""
    from src.agents.paper_curator import PaperCuratorAgent
    from src.agents.base_agent import AgentRole

    agent = PaperCuratorAgent()

    assert agent.get_role() == AgentRole.PAPER_CURATOR
    assert agent.min_quality_threshold == 0.4
    assert agent.top_k == 10


def test_paper_curator_metadata():
    """Test Paper Curator agent metadata."""
    from src.agents.paper_curator import PaperCuratorAgent
    from src.agents.base_agent import AgentRole, AgentPriority

    agent = PaperCuratorAgent()
    metadata = agent.get_metadata()

    assert metadata.role == AgentRole.PAPER_CURATOR
    assert metadata.priority == AgentPriority.HIGH
    assert AgentRole.ANALYST in metadata.dependencies
    assert metadata.can_run_parallel is False


def test_paper_curator_empty_papers():
    """Test Paper Curator with no discovered papers."""
    from src.agents.paper_curator import PaperCuratorAgent

    agent = PaperCuratorAgent()

    state = {
        "original_query": "How do neural networks learn?",
        "discovered_papers": []
    }

    result = agent.execute(state)

    assert result["curated_papers"] == []
    assert result["curation_metadata"].total_discovered == 0


def test_paper_curator_scoring():
    """Test paper scoring logic."""
    from src.agents.paper_curator import PaperCuratorAgent

    agent = PaperCuratorAgent()

    # Sample papers with different qualities
    papers = [
        {
            "title": "Deep Learning for Neural Network Generalization",
            "abstract": "We study how neural networks learn to generalize from training data.",
            "url": "https://arxiv.org/abs/1234.5678",
            "citationCount": 150,
            "year": 2023
        },
        {
            "title": "Unrelated Topic: Cooking Recipes",
            "abstract": "Best recipes for pasta dishes.",
            "url": "https://example.com/cooking",
            "citationCount": 5,
            "year": 2010
        },
        {
            "title": "Neural Networks and Machine Learning",
            "abstract": "Comprehensive overview of neural network architectures and learning algorithms.",
            "url": "https://arxiv.org/abs/2345.6789",
            "citationCount": 75,
            "year": 2022
        }
    ]

    query = "How do neural networks learn and generalize?"

    result = agent.execute({
        "original_query": query,
        "discovered_papers": papers
    })

    # Should have curated papers
    assert len(result["curated_papers"]) > 0

    # First paper should be highly relevant
    curated_urls = result["curated_papers"]
    assert "https://arxiv.org/abs/1234.5678" in curated_urls

    # Cooking paper should likely be filtered out or ranked low
    report = result["curation_metadata"]
    assert report.total_discovered == 3


def test_paper_curator_filtering():
    """Test quality threshold filtering."""
    from src.agents.paper_curator import PaperCuratorAgent

    # Set high quality threshold
    agent = PaperCuratorAgent(min_quality_threshold=0.7, top_k=5)

    papers = [
        {
            "title": "Low Quality Paper",
            "abstract": "Minimal abstract.",
            "url": "https://example.com/low",
            "citationCount": 0,
            "year": 2000
        },
        {
            "title": "High Quality Neural Network Paper",
            "abstract": "Detailed study of neural network generalization with extensive experiments.",
            "url": "https://arxiv.org/abs/1111.2222",
            "citationCount": 200,
            "year": 2024
        }
    ]

    result = agent.execute({
        "original_query": "neural networks",
        "discovered_papers": papers
    })

    # Low quality paper should be filtered out
    report = result["curation_metadata"]
    assert report.filtered_count > 0


def test_paper_curator_ranking():
    """Test paper ranking by combined score."""
    from src.agents.paper_curator import PaperCuratorAgent
    from src.models import PaperScore

    agent = PaperCuratorAgent()

    papers = [
        {
            "title": "Highly Cited but Less Relevant",
            "abstract": "Machine learning applications in robotics and computer vision.",
            "url": "https://example.com/paper1",
            "citationCount": 500,
            "year": 2023
        },
        {
            "title": "Highly Relevant: Neural Network Generalization",
            "abstract": "How do neural networks learn to generalize from limited training data?",
            "url": "https://example.com/paper2",
            "citationCount": 100,
            "year": 2024
        }
    ]

    query = "How do neural networks generalize?"

    result = agent.execute({
        "original_query": query,
        "discovered_papers": papers
    })

    # Paper 2 should rank higher due to relevance
    curated = result["curated_papers"]
    assert len(curated) == 2

    # The highly relevant paper should appear first
    assert "https://example.com/paper2" == curated[0]


def test_paper_curator_top_k_limit():
    """Test that top_k limits the number of returned papers."""
    from src.agents.paper_curator import PaperCuratorAgent

    agent = PaperCuratorAgent(top_k=3)

    # Create 10 papers
    papers = [
        {
            "title": f"Paper {i} on Neural Networks",
            "abstract": f"Study {i} of neural network learning and generalization.",
            "url": f"https://example.com/paper{i}",
            "citationCount": 50 + i,
            "year": 2023
        }
        for i in range(10)
    ]

    result = agent.execute({
        "original_query": "neural networks",
        "discovered_papers": papers
    })

    # Should only return top 3
    assert len(result["curated_papers"]) == 3


def test_paper_curator_relevance_calculation():
    """Test relevance score calculation."""
    from src.agents.paper_curator import PaperCuratorAgent

    agent = PaperCuratorAgent()

    # Test keyword extraction
    keywords = agent._extract_keywords("neural networks and deep learning")
    assert "neural" in keywords
    assert "networks" in keywords
    assert "deep" in keywords
    assert "learning" in keywords
    assert "and" not in keywords  # stopword

    # Test relevance calculation
    relevance = agent._calculate_relevance(
        title="Deep Neural Networks for Image Classification",
        abstract="We use neural networks to classify images using deep learning.",
        query="How do neural networks work?"
    )

    # Should have some relevance due to keyword overlap
    assert relevance > 0.3


def test_paper_curator_quality_calculation():
    """Test quality score calculation."""
    from src.agents.paper_curator import PaperCuratorAgent

    agent = PaperCuratorAgent()

    # High quality: many citations, recent, has abstract
    quality_high = agent._calculate_quality(
        citation_count=150,
        year=2024,
        has_abstract=True
    )

    # Low quality: no citations, old, no abstract
    quality_low = agent._calculate_quality(
        citation_count=0,
        year=2000,
        has_abstract=False
    )

    assert quality_high > quality_low
    assert quality_high > 0.5
    assert quality_low < 0.5


def test_paper_curator_run_method():
    """Test running Paper Curator with pre/post hooks."""
    from src.agents.paper_curator import PaperCuratorAgent

    agent = PaperCuratorAgent()

    papers = [
        {
            "title": "Neural Networks",
            "abstract": "Study of neural networks.",
            "url": "https://example.com/paper",
            "citationCount": 100,
            "year": 2023
        }
    ]

    state = {
        "original_query": "neural networks",
        "discovered_papers": papers
    }

    # Use run() instead of execute() to test hooks
    result = agent.run(state)

    assert "curated_papers" in result
    assert "curation_metadata" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
