"""
Tests for Epic 5: 10-Agent Architecture - Foundation (Task 1)

Tests:
- Base agent framework
- Agent registry
- Epic 5 models
- Agent metadata and priority system
"""

import pytest
from typing import Dict, Any


def test_agent_roles_enum():
    """Test AgentRole enum has all 10 agents."""
    from src.agents.base_agent import AgentRole

    # Original 3 agents
    assert AgentRole.ANALYST.value == "analyst"
    assert AgentRole.SKEPTIC.value == "skeptic"
    assert AgentRole.SYNTHESIZER.value == "synthesizer"

    # New 7 agents (Epic 5)
    assert AgentRole.PAPER_CURATOR.value == "paper_curator"
    assert AgentRole.EVIDENCE_VALIDATOR.value == "evidence_validator"
    assert AgentRole.BIAS_DETECTOR.value == "bias_detector"
    assert AgentRole.CONSISTENCY_CHECKER.value == "consistency_checker"
    assert AgentRole.COUNTER_PERSPECTIVE.value == "counter_perspective"
    assert AgentRole.NOVELTY_ASSESSOR.value == "novelty_assessor"
    assert AgentRole.SYNTHESIS_REVIEWER.value == "synthesis_reviewer"

    # Total: 10 agents
    assert len(AgentRole) == 10


def test_agent_priority_enum():
    """Test AgentPriority enum."""
    from src.agents.base_agent import AgentPriority

    assert AgentPriority.CRITICAL.value == 1
    assert AgentPriority.HIGH.value == 2
    assert AgentPriority.MEDIUM.value == 3
    assert AgentPriority.LOW.value == 4


def test_agent_metadata_creation():
    """Test creating AgentMetadata."""
    from src.agents.base_agent import AgentMetadata, AgentRole, AgentPriority

    metadata = AgentMetadata(
        role=AgentRole.PAPER_CURATOR,
        priority=AgentPriority.HIGH,
        dependencies=[],
        can_run_parallel=True,
        timeout_seconds=30,
        description="Filter and rank papers"
    )

    assert metadata.role == AgentRole.PAPER_CURATOR
    assert metadata.priority == AgentPriority.HIGH
    assert metadata.dependencies == []
    assert metadata.can_run_parallel is True
    assert metadata.timeout_seconds == 30


def test_base_agent_interface():
    """Test BaseAgent abstract interface."""
    from src.agents.base_agent import BaseAgent, AgentRole, AgentMetadata, AgentPriority

    class TestAgent(BaseAgent):
        def get_role(self) -> AgentRole:
            return AgentRole.PAPER_CURATOR

        def get_metadata(self) -> AgentMetadata:
            return AgentMetadata(
                role=AgentRole.PAPER_CURATOR,
                priority=AgentPriority.HIGH,
                dependencies=[],
                can_run_parallel=True,
                timeout_seconds=30,
                description="Test agent"
            )

        def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
            return {"test_field": "test_value"}

    agent = TestAgent()
    assert agent.get_role() == AgentRole.PAPER_CURATOR

    # Test run() with hooks
    state = {"original_query": "test"}
    result = agent.run(state)
    assert result == {"test_field": "test_value"}


def test_agent_registry_registration():
    """Test agent registration in AgentRegistry."""
    from src.agents.base_agent import AgentRegistry, BaseAgent, AgentRole, AgentMetadata, AgentPriority

    class TestAgent(BaseAgent):
        def get_role(self) -> AgentRole:
            return AgentRole.PAPER_CURATOR

        def get_metadata(self) -> AgentMetadata:
            return AgentMetadata(
                role=AgentRole.PAPER_CURATOR,
                priority=AgentPriority.HIGH,
                dependencies=[],
                can_run_parallel=True,
                timeout_seconds=30,
                description="Test agent"
            )

        def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
            return {}

    registry = AgentRegistry()
    agent = TestAgent()
    registry.register(agent)

    # Test retrieval
    retrieved = registry.get_agent(AgentRole.PAPER_CURATOR)
    assert retrieved is agent

    metadata = registry.get_metadata(AgentRole.PAPER_CURATOR)
    assert metadata.role == AgentRole.PAPER_CURATOR


def test_agent_execution_order_simple():
    """Test execution order calculation with simple dependency chain."""
    from src.agents.base_agent import AgentRegistry, BaseAgent, AgentRole, AgentMetadata, AgentPriority

    class Agent1(BaseAgent):
        def get_role(self): return AgentRole.ANALYST
        def get_metadata(self):
            return AgentMetadata(
                role=AgentRole.ANALYST,
                priority=AgentPriority.CRITICAL,
                dependencies=[],
                can_run_parallel=False,
                timeout_seconds=60,
                description="First agent"
            )
        def execute(self, state): return {}

    class Agent2(BaseAgent):
        def get_role(self): return AgentRole.PAPER_CURATOR
        def get_metadata(self):
            return AgentMetadata(
                role=AgentRole.PAPER_CURATOR,
                priority=AgentPriority.HIGH,
                dependencies=[AgentRole.ANALYST],
                can_run_parallel=True,
                timeout_seconds=30,
                description="Depends on Analyst"
            )
        def execute(self, state): return {}

    registry = AgentRegistry()
    registry.register(Agent1())
    registry.register(Agent2())

    stages = registry.get_execution_order()

    # Analyst should run first, then Paper Curator
    assert len(stages) >= 2
    assert AgentRole.ANALYST in stages[0]
    assert AgentRole.PAPER_CURATOR in stages[1]


def test_agent_parallel_execution_grouping():
    """Test that parallelizable agents are grouped together."""
    from src.agents.base_agent import AgentRegistry, BaseAgent, AgentRole, AgentMetadata, AgentPriority

    class Agent1(BaseAgent):
        def get_role(self): return AgentRole.EVIDENCE_VALIDATOR
        def get_metadata(self):
            return AgentMetadata(
                role=AgentRole.EVIDENCE_VALIDATOR,
                priority=AgentPriority.HIGH,
                dependencies=[],
                can_run_parallel=True,
                timeout_seconds=30,
                description="Parallel agent 1"
            )
        def execute(self, state): return {}

    class Agent2(BaseAgent):
        def get_role(self): return AgentRole.BIAS_DETECTOR
        def get_metadata(self):
            return AgentMetadata(
                role=AgentRole.BIAS_DETECTOR,
                priority=AgentPriority.HIGH,
                dependencies=[],
                can_run_parallel=True,
                timeout_seconds=30,
                description="Parallel agent 2"
            )
        def execute(self, state): return {}

    registry = AgentRegistry()
    registry.register(Agent1())
    registry.register(Agent2())

    stages = registry.get_execution_order()

    # Both should be in same stage since they can run in parallel
    assert len(stages) == 1
    assert len(stages[0]) == 2


def test_epic5_models_import():
    """Test that all Epic 5 models can be imported."""
    from src.models import (
        PaperScore, CurationReport,
        EvidenceAlignment, ValidationReport,
        DetectedBias, BiasReport,
        ConsistencyCheck,
        CounterPerspectiveItem,
        NoveltyAssessment,
        SynthesisReview
    )

    # Just verify they exist
    assert PaperScore is not None
    assert CurationReport is not None
    assert ValidationReport is not None
    assert BiasReport is not None


def test_paper_score_model():
    """Test PaperScore model creation."""
    from src.models import PaperScore

    score = PaperScore(
        url="https://arxiv.org/abs/1234.5678",
        title="Test Paper",
        relevance_score=0.9,
        quality_score=0.8,
        citation_count=42,
        rank=1
    )

    assert score.url == "https://arxiv.org/abs/1234.5678"
    assert score.relevance_score == 0.9
    assert score.quality_score == 0.8
    assert score.citation_count == 42
    assert score.rank == 1


def test_curation_report_model():
    """Test CurationReport model creation."""
    from src.models import CurationReport, PaperScore

    paper = PaperScore(
        url="https://arxiv.org/abs/1234.5678",
        title="Test Paper",
        relevance_score=0.9,
        quality_score=0.8,
        citation_count=42,
        rank=1
    )

    report = CurationReport(
        curated_papers=[paper],
        total_discovered=10,
        filtered_count=9,
        avg_quality=0.8
    )

    assert len(report.curated_papers) == 1
    assert report.total_discovered == 10
    assert report.filtered_count == 9
    assert report.avg_quality == 0.8


def test_validation_report_model():
    """Test ValidationReport model creation."""
    from src.models import ValidationReport, EvidenceAlignment

    alignment = EvidenceAlignment(
        evidence_url="https://example.com/paper",
        claim_excerpt="Neural networks learn representations",
        alignment_score=0.85,
        alignment_category="strong",
        reasoning="Evidence directly supports the claim"
    )

    report = ValidationReport(
        alignments=[alignment],
        overall_strength=0.85,
        strong_count=1,
        weak_count=0,
        recommendation="Thesis has strong evidence support"
    )

    assert len(report.alignments) == 1
    assert report.overall_strength == 0.85
    assert report.strong_count == 1


def test_bias_report_model():
    """Test BiasReport model creation."""
    from src.models import BiasReport, DetectedBias

    bias = DetectedBias(
        bias_type="selection",
        confidence=0.7,
        description="Only pro-argument papers selected",
        affected_papers=["https://example.com/paper1"]
    )

    report = BiasReport(
        detected_biases=[bias],
        bias_score=0.6,
        recommendation="Include counter-evidence for balance"
    )

    assert len(report.detected_biases) == 1
    assert report.bias_score == 0.6


def test_novelty_assessment_model():
    """Test NoveltyAssessment model creation."""
    from src.models import NoveltyAssessment

    assessment = NoveltyAssessment(
        novelty_score=75.0,
        innovation_factors=["New perspective", "Novel combination"],
        comparison_to_baseline="Extends prior work significantly",
        assessment_confidence=0.8
    )

    assert assessment.novelty_score == 75.0
    assert len(assessment.innovation_factors) == 2
    assert assessment.assessment_confidence == 0.8


def test_synthesis_review_model():
    """Test SynthesisReview model creation."""
    from src.models import SynthesisReview

    review = SynthesisReview(
        qa_passed=True,
        quality_score=85.0,
        consistency_check=True,
        evidence_check=True,
        issues_found=[],
        recommendation="Accept"
    )

    assert review.qa_passed is True
    assert review.quality_score == 85.0
    assert review.recommendation == "Accept"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
