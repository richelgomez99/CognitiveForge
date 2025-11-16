"""
Epic 5: Task 7 - Novelty Assessor Agent

The Novelty Assessor agent evaluates the novelty and innovation of ideas:
- Compares to existing work
- Identifies novel contributions
- Assesses innovation factors
- Scores novelty level

This ensures research builds meaningfully on prior work.
"""

import logging
from typing import Dict, Any, List

from src.agents.base_agent import BaseAgent, AgentRole, AgentMetadata, AgentPriority
from src.models import NoveltyAssessment

logger = logging.getLogger(__name__)


class NoveltyAssessorAgent(BaseAgent):
    """
    Novelty Assessor Agent - Evaluates novelty and innovation of ideas.

    Responsibilities:
    1. Compare thesis to existing work
    2. Identify novel contributions
    3. Assess innovation factors
    4. Score novelty level
    5. Return novelty assessment
    """

    def __init__(self):
        """Initialize Novelty Assessor agent."""
        super().__init__()

    def get_role(self) -> AgentRole:
        """Return agent role."""
        return AgentRole.NOVELTY_ASSESSOR

    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata."""
        return AgentMetadata(
            role=AgentRole.NOVELTY_ASSESSOR,
            priority=AgentPriority.LOW,
            dependencies=[AgentRole.ANALYST, AgentRole.PAPER_CURATOR],
            can_run_parallel=True,
            timeout_seconds=45,
            description="Assess novelty and innovation of ideas"
        )

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute novelty assessment.

        Args:
            state: AgentState containing:
                - current_thesis: Current thesis
                - curated_papers: List of paper URLs
                - discovered_papers: All discovered papers

        Returns:
            Dict with:
                - novelty_assessment: NoveltyAssessment report
        """
        thesis = state.get("current_thesis")
        curated_papers = state.get("curated_papers", [])

        if not thesis:
            self._logger.warning("No thesis - skipping novelty assessment")
            return {"novelty_assessment": self._create_empty_assessment()}

        self._logger.info(f"Assessing novelty for: '{thesis.claim[:60]}...'")

        # Assess novelty
        novelty_score = self._calculate_novelty_score(thesis, curated_papers)
        innovation_factors = self._identify_innovation_factors(thesis)
        comparison = self._compare_to_baseline(thesis, curated_papers)
        confidence = self._assess_confidence(curated_papers)

        assessment = NoveltyAssessment(
            novelty_score=novelty_score,
            innovation_factors=innovation_factors,
            comparison_to_baseline=comparison,
            assessment_confidence=confidence
        )

        self._logger.info(f"✅ Novelty assessment complete: {novelty_score:.1f}/100")

        return {"novelty_assessment": assessment}

    def _calculate_novelty_score(self, thesis: Any, curated_papers: List[str]) -> float:
        """
        Calculate novelty score (0-100).

        Args:
            thesis: Current thesis
            curated_papers: List of curated papers

        Returns:
            Novelty score (0-100)
        """
        score = 50.0  # Base score

        # Boost score if claim is specific and nuanced
        claim_lower = thesis.claim.lower()

        # Reduce score for common/generic claims
        generic_terms = ["neural network", "machine learning", "deep learning", "ai", "model"]
        generic_count = sum(1 for term in generic_terms if term in claim_lower)

        if generic_count > 2:
            score -= 15

        # Boost score for specificity
        if len(claim_lower.split()) > 15:  # Longer, more specific claims
            score += 10

        # Adjust based on evidence count
        if len(thesis.evidence) >= 5:
            score += 15  # Strong evidence base suggests novel work
        elif len(thesis.evidence) < 3:
            score -= 10  # Weak evidence may indicate derivative work

        # Adjust based on paper recency (if we had year data)
        # For now, assume recent papers suggest novel work
        if len(curated_papers) >= 5:
            score += 10

        # Clamp to 0-100
        return max(0.0, min(100.0, score))

    def _identify_innovation_factors(self, thesis: Any) -> List[str]:
        """
        Identify innovation factors in the thesis.

        Args:
            thesis: Current thesis

        Returns:
            List of innovation factors
        """
        factors = []

        claim_lower = thesis.claim.lower()
        reasoning_lower = thesis.reasoning.lower()

        # Check for novel combination
        if "combination" in reasoning_lower or "combines" in reasoning_lower:
            factors.append("Novel combination of existing approaches")

        # Check for new perspective
        if "perspective" in reasoning_lower or "viewpoint" in reasoning_lower:
            factors.append("New perspective on existing problem")

        # Check for methodological innovation
        if "method" in reasoning_lower or "approach" in reasoning_lower:
            factors.append("Methodological innovation")

        # Check for empirical contribution
        if "empirical" in reasoning_lower or "experiment" in reasoning_lower:
            factors.append("Empirical contribution")

        # Default if none found
        if not factors:
            factors.append("Incremental advancement")

        return factors[:5]  # Max 5 factors

    def _compare_to_baseline(self, thesis: Any, curated_papers: List[str]) -> str:
        """
        Compare thesis to baseline (existing work).

        Args:
            thesis: Current thesis
            curated_papers: List of curated papers

        Returns:
            Comparison description
        """
        if len(curated_papers) >= 5:
            return "Extends prior work with additional evidence and nuanced perspective."
        elif len(curated_papers) >= 3:
            return "Builds moderately on existing research."
        else:
            return "Limited comparison to existing work due to sparse evidence base."

    def _assess_confidence(self, curated_papers: List[str]) -> float:
        """
        Assess confidence in novelty assessment.

        Args:
            curated_papers: List of curated papers

        Returns:
            Confidence score (0-1)
        """
        # More papers = higher confidence in assessment
        if len(curated_papers) >= 10:
            return 0.9
        elif len(curated_papers) >= 5:
            return 0.7
        elif len(curated_papers) >= 3:
            return 0.5
        else:
            return 0.3

    def _create_empty_assessment(self) -> NoveltyAssessment:
        """Create empty novelty assessment."""
        return NoveltyAssessment(
            novelty_score=0.0,
            innovation_factors=[],
            comparison_to_baseline="No assessment performed.",
            assessment_confidence=0.0
        )
