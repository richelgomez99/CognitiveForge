"""
Epic 5: Task 8 - Synthesis Reviewer Agent

The Synthesis Reviewer agent performs final QA on the synthesis:
- Validates synthesis quality
- Checks consistency with evidence
- Verifies all claims are addressed
- Reviews for completeness
- Provides go/no-go recommendation

This is the final quality gate before presenting results to the user.
"""

import logging
from typing import Dict, Any

from src.agents.base_agent import BaseAgent, AgentRole, AgentMetadata, AgentPriority
from src.models import SynthesisReview

logger = logging.getLogger(__name__)


class SynthesisReviewerAgent(BaseAgent):
    """
    Synthesis Reviewer Agent - Final QA on synthesis output.

    Responsibilities:
    1. Validate synthesis quality
    2. Check evidence consistency
    3. Verify completeness
    4. Review for issues
    5. Provide go/no-go recommendation
    """

    def __init__(self, passing_threshold: float = 70.0):
        """
        Initialize Synthesis Reviewer agent.

        Args:
            passing_threshold: Minimum quality score to pass (0-100)
        """
        super().__init__()
        self.passing_threshold = passing_threshold

    def get_role(self) -> AgentRole:
        """Return agent role."""
        return AgentRole.SYNTHESIS_REVIEWER

    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata."""
        return AgentMetadata(
            role=AgentRole.SYNTHESIS_REVIEWER,
            priority=AgentPriority.CRITICAL,
            dependencies=[AgentRole.SYNTHESIZER, AgentRole.EVIDENCE_VALIDATOR, AgentRole.CONSISTENCY_CHECKER],
            can_run_parallel=False,  # Must run last
            timeout_seconds=30,
            description="Final QA review of synthesis output"
        )

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute synthesis review.

        Args:
            state: AgentState containing:
                - final_synthesis: Final synthesis output
                - evidence_validation_report: Evidence validation results
                - consistency_check_report: Consistency check results
                - bias_detection_report: Bias detection results

        Returns:
            Dict with:
                - synthesis_review: SynthesisReview report
        """
        synthesis = state.get("final_synthesis")
        evidence_report = state.get("evidence_validation_report")
        consistency_report = state.get("consistency_check_report")
        bias_report = state.get("bias_detection_report")

        if not synthesis:
            self._logger.warning("No synthesis to review")
            return {"synthesis_review": self._create_failed_review("No synthesis provided")}

        self._logger.info("Reviewing final synthesis...")

        # Run QA checks
        issues = []

        # 1. Evidence check
        if evidence_report and evidence_report.overall_strength < 0.5:
            issues.append(f"Weak evidence support ({evidence_report.overall_strength:.2f})")

        # 2. Consistency check
        if consistency_report and consistency_report.consistency_score < 0.6:
            issues.append(f"Low consistency score ({consistency_report.consistency_score:.2f})")

        # 3. Bias check
        if bias_report and bias_report.bias_score > 0.7:
            issues.append(f"High bias detected ({bias_report.bias_score:.2f})")

        # 4. Completeness check
        if not synthesis.insight or len(synthesis.insight) < 50:
            issues.append("Synthesis insight is too short or missing")

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            evidence_report, consistency_report, bias_report
        )

        # Determine pass/fail
        qa_passed = quality_score >= self.passing_threshold and len(issues) == 0

        # Generate recommendation
        if qa_passed:
            recommendation = "Accept - synthesis meets quality standards"
        elif quality_score >= self.passing_threshold - 10:
            recommendation = "Conditional Accept - minor issues, review recommended"
        else:
            recommendation = f"Reject - quality score {quality_score:.1f} below threshold"

        review = SynthesisReview(
            qa_passed=qa_passed,
            quality_score=quality_score,
            consistency_check=consistency_report.consistency_score >= 0.6 if consistency_report else True,
            evidence_check=evidence_report.overall_strength >= 0.5 if evidence_report else True,
            issues_found=issues,
            recommendation=recommendation
        )

        self._logger.info(f"✅ Synthesis review complete: {'PASS' if qa_passed else 'FAIL'} ({quality_score:.1f}/100)")

        return {"synthesis_review": review}

    def _calculate_quality_score(
        self,
        evidence_report: Any,
        consistency_report: Any,
        bias_report: Any
    ) -> float:
        """
        Calculate overall quality score (0-100).

        Args:
            evidence_report: Evidence validation report
            consistency_report: Consistency check report
            bias_report: Bias detection report

        Returns:
            Quality score (0-100)
        """
        score = 0.0

        # Evidence component (40% weight)
        if evidence_report:
            evidence_score = evidence_report.overall_strength * 100
            score += 0.4 * evidence_score
        else:
            score += 0.4 * 50  # Neutral if missing

        # Consistency component (30% weight)
        if consistency_report:
            consistency_score = consistency_report.consistency_score * 100
            score += 0.3 * consistency_score
        else:
            score += 0.3 * 50

        # Bias component (30% weight, inverted - lower bias = higher score)
        if bias_report:
            bias_score = (1.0 - bias_report.bias_score) * 100
            score += 0.3 * bias_score
        else:
            score += 0.3 * 50

        return round(score, 1)

    def _create_failed_review(self, reason: str) -> SynthesisReview:
        """Create failed review."""
        return SynthesisReview(
            qa_passed=False,
            quality_score=0.0,
            consistency_check=False,
            evidence_check=False,
            issues_found=[reason],
            recommendation="Reject - " + reason
        )
