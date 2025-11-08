"""
Pytest configuration and shared fixtures for tests.

Provides reusable test data that meets Pydantic validation requirements.
"""

import pytest
from src.models import Evidence, Thesis, Antithesis, Synthesis


@pytest.fixture
def valid_evidence():
    """Factory for creating valid Evidence instances."""
    def _make_evidence(url="https://arxiv.org/abs/1706.03762", 
                       snippet="This is a valid snippet with enough characters to pass validation",
                       score=0.9):
        return Evidence(
            source_url=url,
            snippet=snippet,
            relevance_score=score
        )
    return _make_evidence


@pytest.fixture
def valid_thesis():
    """Factory for creating valid Thesis instances."""
    def _make_thesis(
        claim="Transformers represent a significant advancement in neural network architectures",
        reasoning="The self-attention mechanism allows transformers to capture long-range dependencies more effectively than previous architectures"
    ):
        return Thesis(
            claim=claim,
            reasoning=reasoning,
            evidence=[
                Evidence(
                    source_url="https://arxiv.org/abs/1706.03762",
                    snippet="Attention is all you need for sequence transduction",
                    relevance_score=0.95
                ),
                Evidence(
                    source_url="https://arxiv.org/abs/1810.04805",
                    snippet="BERT achieves state-of-the-art on multiple benchmarks",
                    relevance_score=0.90
                )
            ]
        )
    return _make_thesis


@pytest.fixture
def valid_antithesis_with_contradiction():
    """Factory for creating valid Antithesis with contradiction."""
    def _make_antithesis():
        return Antithesis(
            contradiction_found=True,
            counter_claim="However, transformers have computational limitations for long sequences",
            conflicting_evidence=[
                Evidence(
                    source_url="https://arxiv.org/abs/2020.12345",
                    snippet="Quadratic complexity limits transformer scalability",
                    relevance_score=0.85
                )
            ],
            critique="While transformers excel at many tasks, their quadratic computational complexity makes them impractical for very long sequences without significant modifications"
        )
    return _make_antithesis


@pytest.fixture
def valid_antithesis_no_contradiction():
    """Factory for creating valid Antithesis without contradiction."""
    def _make_antithesis():
        return Antithesis(
            contradiction_found=False,
            counter_claim=None,
            conflicting_evidence=[],
            critique="The thesis is well-supported by the evidence and reasoning provided is sound"
        )
    return _make_antithesis


@pytest.fixture
def valid_synthesis():
    """Factory for creating valid Synthesis instances."""
    def _make_synthesis():
        return Synthesis(
            novel_insight="Transformers represent a powerful architecture that balances effectiveness with computational considerations, requiring careful optimization for production use",
            supporting_claims=[
                "Self-attention enables effective sequence modeling",
                "Computational complexity requires optimization strategies"
            ],
            evidence_lineage=[
                "https://arxiv.org/abs/1706.03762",
                "https://arxiv.org/abs/1810.04805",
                "https://arxiv.org/abs/2020.12345"
            ],
            confidence_score=0.85,
            novelty_score=0.70,
            reasoning="By synthesizing the strengths of attention mechanisms with an understanding of their computational limitations, we arrive at a balanced perspective that informs practical implementation"
        )
    return _make_synthesis

