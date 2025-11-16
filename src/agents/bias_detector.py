"""
Epic 5: Task 4 - Bias Detector Agent

The Bias Detector agent identifies potential biases in arguments:
- Selection bias (only pro-argument papers selected)
- Confirmation bias (ignoring counter-evidence)
- Cherry-picking (selective evidence presentation)
- Overgeneralization (claims exceed evidence scope)
- Publication bias (reliance on positive results only)

This ensures balanced, objective research synthesis.
"""

import logging
from typing import Dict, Any, List
import re

from src.agents.base_agent import BaseAgent, AgentRole, AgentMetadata, AgentPriority
from src.models import DetectedBias, BiasReport

logger = logging.getLogger(__name__)


class BiasDetectorAgent(BaseAgent):
    """
    Bias Detector Agent - Identifies biases in arguments and evidence selection.

    Responsibilities:
    1. Detect selection bias in paper curation
    2. Identify confirmation bias in evidence usage
    3. Flag cherry-picking of results
    4. Detect overgeneralization
    5. Produce bias report with recommendations
    """

    def __init__(self, bias_threshold: float = 0.6):
        """
        Initialize Bias Detector agent.

        Args:
            bias_threshold: Minimum confidence to report a bias (0-1)
        """
        super().__init__()
        self.bias_threshold = bias_threshold

    def get_role(self) -> AgentRole:
        """Return agent role."""
        return AgentRole.BIAS_DETECTOR

    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata."""
        return AgentMetadata(
            role=AgentRole.BIAS_DETECTOR,
            priority=AgentPriority.HIGH,
            dependencies=[AgentRole.ANALYST, AgentRole.PAPER_CURATOR],
            can_run_parallel=True,  # Can run in parallel with Evidence Validator
            timeout_seconds=90,
            description="Detect biases in arguments and evidence selection"
        )

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute bias detection logic.

        Args:
            state: AgentState containing:
                - current_thesis: Current thesis with claim and reasoning
                - curated_papers: List of paper URLs
                - discovered_papers: All discovered papers (for selection bias)
                - curation_metadata: Metadata from Paper Curator

        Returns:
            Dict with:
                - bias_detection_report: BiasReport with detected biases
        """
        thesis = state.get("current_thesis")
        curated_paper_urls = state.get("curated_papers", [])
        discovered_papers = state.get("discovered_papers", [])
        curation_metadata = state.get("curation_metadata")

        if not thesis:
            self._logger.warning("No thesis to analyze - skipping bias detection")
            return {"bias_detection_report": self._create_empty_report()}

        self._logger.info(f"Detecting biases in thesis: '{thesis.claim[:60]}...'")

        # Get full paper data
        curated_papers = self._get_curated_paper_data(curated_paper_urls, discovered_papers)

        # Run bias detection checks
        detected_biases = []

        # 1. Selection bias
        selection_bias = self._detect_selection_bias(
            curated_papers=curated_papers,
            all_papers=discovered_papers,
            curation_metadata=curation_metadata
        )
        if selection_bias:
            detected_biases.append(selection_bias)

        # 2. Confirmation bias
        confirmation_bias = self._detect_confirmation_bias(
            thesis=thesis,
            curated_papers=curated_papers
        )
        if confirmation_bias:
            detected_biases.append(confirmation_bias)

        # 3. Overgeneralization
        overgeneralization_bias = self._detect_overgeneralization(
            thesis=thesis,
            curated_papers=curated_papers
        )
        if overgeneralization_bias:
            detected_biases.append(overgeneralization_bias)

        # 4. Cherry-picking
        cherry_picking_bias = self._detect_cherry_picking(
            thesis=thesis,
            curated_papers=curated_papers
        )
        if cherry_picking_bias:
            detected_biases.append(cherry_picking_bias)

        # Create bias report
        report = self._create_report(detected_biases)

        self._logger.info(
            f"✅ Bias detection complete: {len(detected_biases)} biases found "
            f"(bias score: {report.bias_score:.2f})"
        )

        return {"bias_detection_report": report}

    def _get_curated_paper_data(
        self,
        curated_urls: List[str],
        all_papers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get full paper data for curated URLs."""
        url_to_paper = {p.get("url"): p for p in all_papers}
        return [url_to_paper[url] for url in curated_urls if url in url_to_paper]

    def _detect_selection_bias(
        self,
        curated_papers: List[Dict[str, Any]],
        all_papers: List[Dict[str, Any]],
        curation_metadata: Any
    ) -> DetectedBias:
        """
        Detect selection bias in paper curation.

        Selection bias occurs when:
        - Very high filtering rate (> 70% papers filtered)
        - Only positive/supporting papers selected

        Args:
            curated_papers: Papers that were selected
            all_papers: All discovered papers
            curation_metadata: Metadata from curation

        Returns:
            DetectedBias if bias found, None otherwise
        """
        if not all_papers or not curated_papers:
            return None

        # Calculate filtering rate
        filtering_rate = (len(all_papers) - len(curated_papers)) / max(len(all_papers), 1)

        # High filtering rate suggests possible selection bias
        if filtering_rate > 0.7:
            confidence = min(filtering_rate, 0.95)  # Cap at 0.95

            return DetectedBias(
                bias_type="selection",
                confidence=confidence,
                description=f"High filtering rate ({filtering_rate:.1%}): {len(all_papers) - len(curated_papers)} of {len(all_papers)} papers excluded. May indicate selective evidence gathering.",
                affected_papers=[p.get("url", "") for p in curated_papers[:3]]  # Show sample
            )

        return None

    def _detect_confirmation_bias(
        self,
        thesis: Any,
        curated_papers: List[Dict[str, Any]]
    ) -> DetectedBias:
        """
        Detect confirmation bias.

        Confirmation bias occurs when:
        - All papers support the claim (no counter-evidence)
        - Papers only present one side of the argument

        Args:
            thesis: Current thesis
            curated_papers: Selected papers

        Returns:
            DetectedBias if bias found, None otherwise
        """
        if not curated_papers or len(curated_papers) < 3:
            return None

        # Look for contradictory or critical language in abstracts
        critical_terms = [
            "however", "but", "limitation", "contrary", "refute", "disprove",
            "challenge", "contradict", "question", "criticize", "alternative"
        ]

        papers_with_critical_terms = 0
        for paper in curated_papers:
            abstract = paper.get("abstract", "").lower()
            if any(term in abstract for term in critical_terms):
                papers_with_critical_terms += 1

        # If very few papers have critical language, may indicate confirmation bias
        critical_rate = papers_with_critical_terms / len(curated_papers)

        if critical_rate < 0.2 and len(curated_papers) >= 5:  # < 20% have critical terms
            confidence = 0.7

            return DetectedBias(
                bias_type="confirmation",
                confidence=confidence,
                description=f"Only {papers_with_critical_terms}/{len(curated_papers)} papers contain critical or alternative perspectives. Consider including counter-evidence for balance.",
                affected_papers=[p.get("url", "") for p in curated_papers]
            )

        return None

    def _detect_overgeneralization(
        self,
        thesis: Any,
        curated_papers: List[Dict[str, Any]]
    ) -> DetectedBias:
        """
        Detect overgeneralization bias.

        Overgeneralization occurs when:
        - Claims use absolute language ("always", "never", "all")
        - Limited evidence (< 3 papers) for broad claims

        Args:
            thesis: Current thesis
            curated_papers: Selected papers

        Returns:
            DetectedBias if bias found, None otherwise
        """
        if not thesis:
            return None

        claim = thesis.claim.lower()
        reasoning = thesis.reasoning.lower()

        # Absolute language indicators
        absolute_terms = [
            "always", "never", "all", "every", "none", "impossible",
            "certain", "definite", "absolute", "universal", "invariably"
        ]

        has_absolute_language = any(term in claim or term in reasoning for term in absolute_terms)

        # Check evidence count
        limited_evidence = len(curated_papers) < 3

        if has_absolute_language and limited_evidence:
            confidence = 0.75

            return DetectedBias(
                bias_type="overgeneralization",
                confidence=confidence,
                description=f"Claim uses absolute language but is supported by only {len(curated_papers)} papers. Consider qualifying claims or adding more evidence.",
                affected_papers=[p.get("url", "") for p in curated_papers]
            )
        elif has_absolute_language:
            confidence = 0.55

            return DetectedBias(
                bias_type="overgeneralization",
                confidence=confidence,
                description="Claim uses absolute language (e.g., 'always', 'never'). Consider more nuanced phrasing.",
                affected_papers=[]
            )

        return None

    def _detect_cherry_picking(
        self,
        thesis: Any,
        curated_papers: List[Dict[str, Any]]
    ) -> DetectedBias:
        """
        Detect cherry-picking bias.

        Cherry-picking occurs when:
        - Very narrow selection of evidence (< 5 papers when many available)
        - All papers from same source or author

        Args:
            thesis: Current thesis
            curated_papers: Selected papers

        Returns:
            DetectedBias if bias found, None otherwise
        """
        if not curated_papers:
            return None

        # Check for narrow selection
        if len(curated_papers) < 5:
            # Extract domains from URLs
            domains = []
            for paper in curated_papers:
                url = paper.get("url", "")
                # Extract domain (e.g., "arxiv.org" from "https://arxiv.org/abs/1234")
                match = re.search(r'https?://([^/]+)', url)
                if match:
                    domains.append(match.group(1))

            # Check if all from same source
            unique_domains = set(domains)

            if len(unique_domains) == 1 and len(curated_papers) >= 3:
                confidence = 0.65

                return DetectedBias(
                    bias_type="cherry_picking",
                    confidence=confidence,
                    description=f"All {len(curated_papers)} papers are from the same source ({list(unique_domains)[0]}). Consider diversifying sources.",
                    affected_papers=[p.get("url", "") for p in curated_papers]
                )

        return None

    def _create_report(self, detected_biases: List[DetectedBias]) -> BiasReport:
        """
        Create bias report from detected biases.

        Args:
            detected_biases: List of detected biases

        Returns:
            BiasReport
        """
        # Filter biases by threshold
        significant_biases = [
            b for b in detected_biases
            if b.confidence >= self.bias_threshold
        ]

        # Calculate overall bias score (average confidence)
        if significant_biases:
            bias_score = sum(b.confidence for b in significant_biases) / len(significant_biases)
        else:
            bias_score = 0.0

        # Generate recommendation
        if not significant_biases:
            recommendation = "No significant biases detected. Analysis appears balanced."
        elif bias_score >= 0.8:
            recommendation = "High bias risk. Strongly recommend revising claims and evidence selection."
        elif bias_score >= 0.6:
            recommendation = "Moderate bias detected. Consider addressing identified issues for more balanced analysis."
        else:
            recommendation = "Low bias detected. Minor improvements recommended."

        return BiasReport(
            detected_biases=significant_biases,
            bias_score=bias_score,
            recommendation=recommendation
        )

    def _create_empty_report(self) -> BiasReport:
        """Create empty bias report."""
        return BiasReport(
            detected_biases=[],
            bias_score=0.0,
            recommendation="No analysis performed - insufficient data."
        )
