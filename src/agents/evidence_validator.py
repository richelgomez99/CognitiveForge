"""
Epic 5: Task 3 - Evidence Validator Agent

The Evidence Validator agent validates that evidence from papers aligns
with the claims in the current thesis:
- Extracts key claims from the thesis
- Matches evidence from curated papers to each claim
- Scores alignment quality (strong/weak/contradictory)
- Identifies gaps in evidence support

This ensures claims are backed by empirical evidence and flags weak arguments.
"""

import logging
from typing import Dict, Any, List, Tuple
import re

from src.agents.base_agent import BaseAgent, AgentRole, AgentMetadata, AgentPriority
from src.models import EvidenceAlignment, ValidationReport

logger = logging.getLogger(__name__)


class EvidenceValidatorAgent(BaseAgent):
    """
    Evidence Validator Agent - Validates evidence alignment with claims.

    Responsibilities:
    1. Extract claims from thesis
    2. Match evidence from papers to claims
    3. Score alignment quality
    4. Identify evidence gaps
    5. Produce validation report
    """

    def __init__(self, alignment_threshold: float = 0.6):
        """
        Initialize Evidence Validator agent.

        Args:
            alignment_threshold: Minimum score to consider evidence "strong" (0-1)
        """
        super().__init__()
        self.alignment_threshold = alignment_threshold

    def get_role(self) -> AgentRole:
        """Return agent role."""
        return AgentRole.EVIDENCE_VALIDATOR

    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata."""
        return AgentMetadata(
            role=AgentRole.EVIDENCE_VALIDATOR,
            priority=AgentPriority.HIGH,
            dependencies=[AgentRole.ANALYST, AgentRole.PAPER_CURATOR],
            can_run_parallel=True,  # Can run in parallel with Bias Detector
            timeout_seconds=90,
            description="Validate evidence alignment with thesis claims"
        )

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute evidence validation logic.

        Args:
            state: AgentState containing:
                - current_thesis: Current thesis with claim and reasoning
                - curated_papers: List of paper URLs (from Paper Curator)
                - discovered_papers: Full paper data

        Returns:
            Dict with:
                - evidence_validation_report: ValidationReport with alignments
        """
        thesis = state.get("current_thesis")
        curated_paper_urls = state.get("curated_papers", [])
        discovered_papers = state.get("discovered_papers", [])

        if not thesis:
            self._logger.warning("No thesis to validate - skipping evidence validation")
            return {"evidence_validation_report": self._create_empty_report()}

        if not curated_paper_urls:
            self._logger.warning("No curated papers - evidence validation has no sources")
            return {"evidence_validation_report": self._create_empty_report()}

        self._logger.info(f"Validating evidence for thesis: '{thesis.claim[:60]}...'")

        # Get full paper data for curated papers
        curated_papers = self._get_curated_paper_data(curated_paper_urls, discovered_papers)

        # Extract claims from thesis
        claims = self._extract_claims(thesis)
        self._logger.info(f"Extracted {len(claims)} claims from thesis")

        # Validate each claim against evidence
        alignments = []
        for claim in claims:
            claim_alignments = self._validate_claim(claim, curated_papers)
            alignments.extend(claim_alignments)

        # Create validation report
        report = self._create_report(alignments)

        self._logger.info(
            f"✅ Evidence validation complete: {report.strong_count} strong, "
            f"{report.weak_count} weak alignments (overall: {report.overall_strength:.2f})"
        )

        return {"evidence_validation_report": report}

    def _get_curated_paper_data(
        self,
        curated_urls: List[str],
        all_papers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Get full paper data for curated URLs.

        Args:
            curated_urls: List of paper URLs
            all_papers: All discovered papers

        Returns:
            List of paper dicts with full data
        """
        url_to_paper = {p.get("url"): p for p in all_papers}
        return [url_to_paper[url] for url in curated_urls if url in url_to_paper]

    def _extract_claims(self, thesis) -> List[str]:
        """
        Extract key claims from thesis.

        For now, we extract:
        1. The main claim
        2. Sentences from reasoning that make specific assertions

        Args:
            thesis: Thesis object with claim and reasoning

        Returns:
            List of claim strings
        """
        claims = []

        # Add main claim
        if thesis.claim:
            claims.append(thesis.claim)

        # Extract assertions from reasoning
        if thesis.reasoning:
            # Split into sentences
            sentences = re.split(r'[.!?]+', thesis.reasoning)

            # Keep sentences that make assertions (heuristic: contains key verbs)
            assertion_verbs = ["shows", "demonstrates", "proves", "indicates", "suggests", "reveals"]

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # Minimum length
                    if any(verb in sentence.lower() for verb in assertion_verbs):
                        claims.append(sentence)

        return claims

    def _validate_claim(
        self,
        claim: str,
        papers: List[Dict[str, Any]]
    ) -> List[EvidenceAlignment]:
        """
        Validate a claim against evidence from papers.

        Args:
            claim: Claim to validate
            papers: Curated papers

        Returns:
            List of EvidenceAlignment objects
        """
        alignments = []

        for paper in papers:
            url = paper.get("url", "")
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")

            # Calculate alignment score
            score, category, reasoning = self._calculate_alignment(
                claim=claim,
                paper_title=title,
                paper_abstract=abstract
            )

            # Only include alignments above minimum threshold
            if score > 0.2:  # Minimum relevance threshold
                alignments.append(EvidenceAlignment(
                    evidence_url=url,
                    claim_excerpt=claim[:100],  # First 100 chars
                    alignment_score=score,
                    alignment_category=category,
                    reasoning=reasoning
                ))

        return alignments

    def _calculate_alignment(
        self,
        claim: str,
        paper_title: str,
        paper_abstract: str
    ) -> Tuple[float, str, str]:
        """
        Calculate alignment score between claim and paper evidence.

        Args:
            claim: Claim text
            paper_title: Paper title
            paper_abstract: Paper abstract

        Returns:
            Tuple of (score, category, reasoning)
        """
        # Extract keywords from claim
        claim_keywords = set(self._extract_keywords(claim.lower()))

        # Extract keywords from paper
        title_keywords = set(self._extract_keywords(paper_title.lower()))
        abstract_keywords = set(self._extract_keywords(paper_abstract.lower()))

        # Calculate keyword overlap
        title_overlap = len(claim_keywords & title_keywords) / max(len(claim_keywords), 1)
        abstract_overlap = len(claim_keywords & abstract_keywords) / max(len(claim_keywords), 1)

        # Combined score (weight abstract higher than title)
        alignment_score = (0.3 * title_overlap) + (0.7 * abstract_overlap)

        # Categorize alignment
        if alignment_score >= self.alignment_threshold:
            category = "strong"
            reasoning = f"Strong keyword overlap ({alignment_score:.2f}). Paper directly addresses claim."
        elif alignment_score >= 0.4:
            category = "moderate"
            reasoning = f"Moderate keyword overlap ({alignment_score:.2f}). Paper provides some support."
        elif alignment_score >= 0.2:
            category = "weak"
            reasoning = f"Weak keyword overlap ({alignment_score:.2f}). Limited relevance to claim."
        else:
            category = "insufficient"
            reasoning = "Insufficient evidence. Paper does not address claim."

        # Check for contradictory signals (heuristic)
        contradictory_terms = ["however", "but", "contradiction", "contrary", "refutes", "disproves"]
        if any(term in paper_abstract.lower() for term in contradictory_terms):
            # Reduce score if contradictory language detected
            alignment_score *= 0.8
            category = "contradictory"
            reasoning += " Potential contradictory evidence detected."

        return alignment_score, category, reasoning

    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract meaningful keywords from text.

        Args:
            text: Input text

        Returns:
            List of keywords
        """
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', text)

        # Filter: length >= 3, not common stopwords
        stopwords = {
            "the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
            "was", "one", "our", "out", "day", "get", "has", "him", "his", "how",
            "that", "this", "with", "from", "have", "what", "when", "your", "about",
            "they", "will", "would", "there", "their", "been", "than", "more", "some"
        }

        keywords = [
            w for w in words
            if len(w) >= 3 and w not in stopwords
        ]

        return keywords

    def _create_report(self, alignments: List[EvidenceAlignment]) -> ValidationReport:
        """
        Create validation report from alignments.

        Args:
            alignments: List of evidence alignments

        Returns:
            ValidationReport
        """
        if not alignments:
            return self._create_empty_report()

        # Count by category
        strong_count = len([a for a in alignments if a.alignment_category == "strong"])
        moderate_count = len([a for a in alignments if a.alignment_category == "moderate"])
        weak_count = len([a for a in alignments if a.alignment_category in ["weak", "insufficient"]])
        contradictory_count = len([a for a in alignments if a.alignment_category == "contradictory"])

        # Calculate overall strength (weighted average)
        total_score = sum(a.alignment_score for a in alignments)
        overall_strength = total_score / len(alignments)

        # Generate recommendation
        if overall_strength >= 0.7 and strong_count >= 2:
            recommendation = "Thesis has strong evidence support. Claims are well-backed by empirical research."
        elif overall_strength >= 0.5 and strong_count >= 1:
            recommendation = "Thesis has moderate evidence support. Some claims need stronger backing."
        elif contradictory_count > 0:
            recommendation = f"Warning: {contradictory_count} contradictory evidence found. Review claim validity."
        else:
            recommendation = "Thesis has weak evidence support. Consider refining claims or finding better sources."

        return ValidationReport(
            alignments=alignments,
            overall_strength=overall_strength,
            strong_count=strong_count,
            weak_count=weak_count,
            recommendation=recommendation
        )

    def _create_empty_report(self) -> ValidationReport:
        """Create empty validation report."""
        return ValidationReport(
            alignments=[],
            overall_strength=0.0,
            strong_count=0,
            weak_count=0,
            recommendation="No evidence available for validation."
        )
