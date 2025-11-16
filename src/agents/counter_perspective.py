"""
Epic 5: Task 6 - Counter-Perspective Agent

The Counter-Perspective agent generates alternative viewpoints and counter-arguments:
- Identifies assumptions in the thesis
- Generates alternative interpretations
- Provides counter-evidence perspectives
- Challenges conclusions

This ensures balanced analysis by considering opposing views.
"""

import logging
from typing import Dict, Any, List

from src.agents.base_agent import BaseAgent, AgentRole, AgentMetadata, AgentPriority
from src.models import CounterPerspectiveItem

logger = logging.getLogger(__name__)


class CounterPerspectiveAgent(BaseAgent):
    """
    Counter-Perspective Agent - Generates alternative viewpoints and counter-arguments.

    Responsibilities:
    1. Identify key assumptions in thesis
    2. Generate alternative interpretations
    3. Provide counter-arguments
    4. Challenge conclusions
    5. Return list of counter-perspectives
    """

    def __init__(self, max_perspectives: int = 5):
        """
        Initialize Counter-Perspective agent.

        Args:
            max_perspectives: Maximum number of counter-perspectives to generate
        """
        super().__init__()
        self.max_perspectives = max_perspectives

    def get_role(self) -> AgentRole:
        """Return agent role."""
        return AgentRole.COUNTER_PERSPECTIVE

    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata."""
        return AgentMetadata(
            role=AgentRole.COUNTER_PERSPECTIVE,
            priority=AgentPriority.MEDIUM,
            dependencies=[AgentRole.ANALYST, AgentRole.SKEPTIC],
            can_run_parallel=True,
            timeout_seconds=60,
            description="Generate alternative viewpoints and counter-arguments"
        )

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute counter-perspective generation.

        Args:
            state: AgentState containing:
                - current_thesis: Current thesis with claim and reasoning
                - skeptic_critique: Critique from skeptic (optional)

        Returns:
            Dict with:
                - counter_perspectives: List of CounterPerspectiveItem objects
        """
        thesis = state.get("current_thesis")

        if not thesis:
            self._logger.warning("No thesis - skipping counter-perspective generation")
            return {"counter_perspectives": []}

        self._logger.info(f"Generating counter-perspectives for: '{thesis.claim[:60]}...'")

        perspectives = []

        # 1. Challenge assumptions
        assumption_perspectives = self._challenge_assumptions(thesis)
        perspectives.extend(assumption_perspectives)

        # 2. Generate alternative interpretations
        alternative_perspectives = self._generate_alternatives(thesis)
        perspectives.extend(alternative_perspectives)

        # 3. Provide counter-arguments
        counter_arguments = self._generate_counter_arguments(thesis)
        perspectives.extend(counter_arguments)

        # Limit to max_perspectives
        perspectives = perspectives[:self.max_perspectives]

        self._logger.info(f"✅ Generated {len(perspectives)} counter-perspectives")

        return {"counter_perspectives": perspectives}

    def _challenge_assumptions(self, thesis: Any) -> List[CounterPerspectiveItem]:
        """
        Challenge assumptions in the thesis.

        Args:
            thesis: Current thesis

        Returns:
            List of counter-perspectives challenging assumptions
        """
        perspectives = []

        claim = thesis.claim.lower()

        # Identify common assumption patterns
        if "always" in claim or "never" in claim:
            perspectives.append(CounterPerspectiveItem(
                perspective="Challenge: The claim uses absolute language. Consider scenarios where exceptions exist.",
                supporting_evidence=[],
                strength=0.8
            ))

        if "all" in claim or "every" in claim:
            perspectives.append(CounterPerspectiveItem(
                perspective="Challenge: Universal claims may oversimplify. Consider specific cases or contexts where this doesn't hold.",
                supporting_evidence=[],
                strength=0.7
            ))

        return perspectives

    def _generate_alternatives(self, thesis: Any) -> List[CounterPerspectiveItem]:
        """
        Generate alternative interpretations.

        Args:
            thesis: Current thesis

        Returns:
            List of alternative interpretations
        """
        perspectives = []

        # Generate generic alternative perspective
        perspectives.append(CounterPerspectiveItem(
            perspective="Alternative view: The evidence could be interpreted differently depending on methodological assumptions.",
            supporting_evidence=[],
            strength=0.6
        ))

        return perspectives

    def _generate_counter_arguments(self, thesis: Any) -> List[CounterPerspectiveItem]:
        """
        Generate counter-arguments to the thesis.

        Args:
            thesis: Current thesis

        Returns:
            List of counter-arguments
        """
        perspectives = []

        # Check for causal claims
        if "because" in thesis.reasoning.lower() or "causes" in thesis.reasoning.lower():
            perspectives.append(CounterPerspectiveItem(
                perspective="Counter-argument: Correlation does not imply causation. Alternative explanations may exist.",
                supporting_evidence=[],
                strength=0.9
            ))

        # Challenge evidence sufficiency
        if len(thesis.evidence) < 5:
            perspectives.append(CounterPerspectiveItem(
                perspective=f"Counter-argument: Evidence base is limited ({len(thesis.evidence)} sources). More evidence needed for strong claims.",
                supporting_evidence=[],
                strength=0.7
            ))

        return perspectives
