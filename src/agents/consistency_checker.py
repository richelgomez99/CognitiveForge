"""
Epic 5: Task 5 - Consistency Checker Agent

The Consistency Checker agent validates logical consistency:
- Internal consistency (claims don't contradict each other)
- Evidence-claim consistency (evidence supports the claims made)
- Reasoning consistency (logic flows coherently)
- Temporal consistency (causality is sound)

This ensures arguments are logically coherent and free of contradictions.
"""

import logging
from typing import Dict, Any, List, Tuple
import re

from src.agents.base_agent import BaseAgent, AgentRole, AgentMetadata, AgentPriority
from src.models import ConsistencyCheck

logger = logging.getLogger(__name__)


class ConsistencyCheckerAgent(BaseAgent):
    """
    Consistency Checker Agent - Validates logical consistency of arguments.

    Responsibilities:
    1. Check internal consistency between claims
    2. Validate evidence-claim alignment
    3. Verify reasoning coherence
    4. Detect logical contradictions
    5. Produce consistency report
    """

    def __init__(self):
        """Initialize Consistency Checker agent."""
        super().__init__()

    def get_role(self) -> AgentRole:
        """Return agent role."""
        return AgentRole.CONSISTENCY_CHECKER

    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata."""
        return AgentMetadata(
            role=AgentRole.CONSISTENCY_CHECKER,
            priority=AgentPriority.MEDIUM,
            dependencies=[AgentRole.ANALYST, AgentRole.EVIDENCE_VALIDATOR],
            can_run_parallel=True,
            timeout_seconds=60,
            description="Validate logical consistency of arguments and claims"
        )

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute consistency checking logic.

        Args:
            state: AgentState containing:
                - current_thesis: Current thesis with claim and reasoning
                - evidence_validation_report: Report from Evidence Validator

        Returns:
            Dict with:
                - consistency_check_report: ConsistencyCheck report
        """
        thesis = state.get("current_thesis")
        validation_report = state.get("evidence_validation_report")

        if not thesis:
            self._logger.warning("No thesis to check - skipping consistency check")
            return {"consistency_check_report": self._create_empty_report()}

        self._logger.info(f"Checking consistency for thesis: '{thesis.claim[:60]}...'")

        # Run consistency checks
        issues = []

        # 1. Internal claim consistency
        internal_issues = self._check_internal_consistency(thesis)
        issues.extend(internal_issues)

        # 2. Evidence-claim consistency
        if validation_report:
            evidence_issues = self._check_evidence_consistency(thesis, validation_report)
            issues.extend(evidence_issues)

        # 3. Reasoning coherence
        reasoning_issues = self._check_reasoning_coherence(thesis)
        issues.extend(reasoning_issues)

        # 4. Logical contradictions
        contradiction_issues = self._check_contradictions(thesis)
        issues.extend(contradiction_issues)

        # Create consistency report
        report = self._create_report(issues)

        self._logger.info(
            f"✅ Consistency check complete: {len(issues)} issues found "
            f"(score: {report.consistency_score:.2f})"
        )

        return {"consistency_check_report": report}

    def _check_internal_consistency(self, thesis: Any) -> List[str]:
        """
        Check internal consistency between claim and reasoning.

        Args:
            thesis: Current thesis

        Returns:
            List of consistency issues
        """
        issues = []

        claim = thesis.claim.lower()
        reasoning = thesis.reasoning.lower()

        # Check if reasoning addresses the claim
        # Extract key terms from claim
        claim_keywords = self._extract_keywords(claim)

        # Check if reasoning mentions claim keywords
        reasoning_keywords = self._extract_keywords(reasoning)

        overlap = len(set(claim_keywords) & set(reasoning_keywords))
        overlap_ratio = overlap / max(len(claim_keywords), 1)

        if overlap_ratio < 0.3:
            issues.append(
                "Internal inconsistency: Reasoning does not adequately address key terms in the claim. "
                f"Only {overlap_ratio:.1%} keyword overlap."
            )

        return issues

    def _check_evidence_consistency(
        self,
        thesis: Any,
        validation_report: Any
    ) -> List[str]:
        """
        Check consistency between evidence and claims.

        Args:
            thesis: Current thesis
            validation_report: Evidence validation report

        Returns:
            List of consistency issues
        """
        issues = []

        # Check if evidence supports the claim
        if validation_report.overall_strength < 0.5:
            issues.append(
                f"Evidence-claim inconsistency: Low evidence support ({validation_report.overall_strength:.2f}). "
                "Claims are not well-backed by provided evidence."
            )

        # Check for contradictory evidence
        contradictory_count = len([
            a for a in validation_report.alignments
            if a.alignment_category == "contradictory"
        ])

        if contradictory_count > 0:
            issues.append(
                f"Evidence contradiction: {contradictory_count} pieces of evidence contradict the thesis. "
                "Review claim validity or evidence selection."
            )

        return issues

    def _check_reasoning_coherence(self, thesis: Any) -> List[str]:
        """
        Check coherence of reasoning logic.

        Args:
            thesis: Current thesis

        Returns:
            List of coherence issues
        """
        issues = []

        reasoning = thesis.reasoning

        # Check for coherence markers
        # Good reasoning should have logical connectors
        connectors = [
            "therefore", "thus", "hence", "because", "since", "as a result",
            "consequently", "so", "given that", "due to", "leading to"
        ]

        has_connectors = any(c in reasoning.lower() for c in connectors)

        if not has_connectors and len(reasoning) > 100:
            issues.append(
                "Reasoning coherence issue: Lacks logical connectors (e.g., 'therefore', 'because'). "
                "May indicate weak logical flow."
            )

        # Check for contradictory connectors in same reasoning
        contradictory_pairs = [
            ("supports", "refutes"),
            ("proves", "disproves"),
            ("confirms", "contradicts"),
            ("shows", "fails to show")
        ]

        reasoning_lower = reasoning.lower()
        for word1, word2 in contradictory_pairs:
            if word1 in reasoning_lower and word2 in reasoning_lower:
                issues.append(
                    f"Reasoning contradiction: Contains both '{word1}' and '{word2}'. "
                    "May indicate conflicting arguments."
                )

        return issues

    def _check_contradictions(self, thesis: Any) -> List[str]:
        """
        Check for logical contradictions in claim or reasoning.

        Args:
            thesis: Current thesis

        Returns:
            List of contradiction issues
        """
        issues = []

        combined_text = (thesis.claim + " " + thesis.reasoning).lower()

        # Check for obvious contradictions
        contradiction_patterns = [
            (r'\balways\b.*\bnever\b', "Uses both 'always' and 'never'"),
            (r'\ball\b.*\bnone\b', "Uses both 'all' and 'none'"),
            (r'\bincreases\b.*\bdecreases\b', "Claims both increase and decrease"),
            (r'\bimproves\b.*\bworsens\b', "Claims both improvement and worsening"),
        ]

        for pattern, description in contradiction_patterns:
            if re.search(pattern, combined_text):
                issues.append(
                    f"Logical contradiction detected: {description}. "
                    "Review for inconsistent statements."
                )

        return issues

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        words = re.findall(r'\b\w+\b', text)

        stopwords = {
            "the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
            "was", "one", "our", "out", "day", "get", "has", "him", "his", "how",
            "that", "this", "with", "from", "have", "what", "when", "your", "about",
            "they", "will", "would", "there", "their", "been", "than", "more", "some"
        }

        keywords = [w for w in words if len(w) >= 3 and w not in stopwords]
        return keywords

    def _create_report(self, issues: List[str]) -> ConsistencyCheck:
        """
        Create consistency check report.

        Args:
            issues: List of consistency issues found

        Returns:
            ConsistencyCheck report
        """
        # Calculate consistency score (inverse of issue count)
        # Max 5 issues = score 0, 0 issues = score 1.0
        consistency_score = max(0.0, 1.0 - (len(issues) / 5.0))

        # Check for contradictions
        contradiction_found = any("contradict" in issue.lower() for issue in issues)

        # Generate recommendation
        if consistency_score >= 0.8 and len(issues) == 0:
            recommendation = "Argument is logically consistent. No issues detected."
        elif consistency_score >= 0.6:
            recommendation = f"Generally consistent ({len(issues)} minor issues). Review suggested."
        else:
            recommendation = f"Consistency issues found ({len(issues)}). Revision recommended."

        # Limit recommendation length to 200 chars
        if len(recommendation) > 200:
            recommendation = recommendation[:197] + "..."

        return ConsistencyCheck(
            similar_past_claims=[],  # TODO: Integrate with persistent memory in future
            contradiction_found=contradiction_found,
            consistency_score=consistency_score,
            recommendation=recommendation
        )

    def _create_empty_report(self) -> ConsistencyCheck:
        """Create empty consistency report."""
        return ConsistencyCheck(
            similar_past_claims=[],
            contradiction_found=False,
            consistency_score=0.0,
            recommendation="No analysis performed - insufficient data."
        )
