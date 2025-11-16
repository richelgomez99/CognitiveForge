"""
Epic 5: Task 1 - Base Agent Framework

Provides abstract base class and registry for the 10-agent architecture.
Establishes common interface, agent metadata, and orchestration patterns.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Agent Metadata
# =============================================================================

class AgentRole(str, Enum):
    """Roles in the 10-agent architecture."""
    # Original 3 agents (Tier 1-3)
    ANALYST = "analyst"
    SKEPTIC = "skeptic"
    SYNTHESIZER = "synthesizer"

    # New 7 agents (Epic 5)
    PAPER_CURATOR = "paper_curator"
    EVIDENCE_VALIDATOR = "evidence_validator"
    BIAS_DETECTOR = "bias_detector"
    CONSISTENCY_CHECKER = "consistency_checker"
    COUNTER_PERSPECTIVE = "counter_perspective"
    NOVELTY_ASSESSOR = "novelty_assessor"
    SYNTHESIS_REVIEWER = "synthesis_reviewer"


class AgentPriority(int, Enum):
    """Execution priority for agent orchestration."""
    CRITICAL = 1  # Must complete before others
    HIGH = 2      # Important for quality
    MEDIUM = 3    # Adds value but not blocking
    LOW = 4       # Nice-to-have enhancements


@dataclass
class AgentMetadata:
    """
    Metadata describing an agent's capabilities and requirements.

    Attributes:
        role: Agent's role in the architecture
        priority: Execution priority
        dependencies: List of agent roles this agent depends on
        can_run_parallel: Whether this agent can run in parallel with others
        timeout_seconds: Maximum execution time
        description: Human-readable description
    """
    role: AgentRole
    priority: AgentPriority
    dependencies: List[AgentRole]
    can_run_parallel: bool
    timeout_seconds: int
    description: str


# =============================================================================
# Base Agent Interface
# =============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the 10-agent architecture.

    Provides common interface for:
    - Agent execution
    - State management
    - Error handling
    - Telemetry/logging
    """

    def __init__(self):
        """Initialize base agent."""
        self._metadata: Optional[AgentMetadata] = None
        self._logger = logging.getLogger(f"agent.{self.get_role()}")

    @abstractmethod
    def get_role(self) -> AgentRole:
        """Return the agent's role."""
        pass

    @abstractmethod
    def get_metadata(self) -> AgentMetadata:
        """Return agent metadata."""
        pass

    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's logic.

        Args:
            state: Current agent state (AgentState TypedDict)

        Returns:
            Updated state fields (partial state update)
        """
        pass

    def pre_execute_hook(self, state: Dict[str, Any]) -> None:
        """
        Hook called before execution.

        Can be overridden for validation, logging, etc.

        Args:
            state: Current agent state
        """
        self._logger.info(f"{self.get_role().value}: Starting execution")

    def post_execute_hook(self, state: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Hook called after execution.

        Can be overridden for telemetry, cleanup, etc.

        Args:
            state: Current agent state
            result: Result from execute()
        """
        self._logger.info(f"{self.get_role().value}: Completed execution")

    def handle_error(self, state: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        """
        Handle execution errors.

        Default behavior: log error and return empty state update.
        Override for custom error handling.

        Args:
            state: Current agent state
            error: Exception that occurred

        Returns:
            Partial state update (may be empty)
        """
        self._logger.error(f"{self.get_role().value}: Error during execution: {error}")
        return {}

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the agent with pre/post hooks and error handling.

        This is the main entry point for agent execution.

        Args:
            state: Current agent state

        Returns:
            Updated state fields
        """
        try:
            self.pre_execute_hook(state)
            result = self.execute(state)
            self.post_execute_hook(state, result)
            return result
        except Exception as e:
            return self.handle_error(state, e)


# =============================================================================
# Agent Registry
# =============================================================================

class AgentRegistry:
    """
    Registry for managing all agents in the 10-agent architecture.

    Provides:
    - Agent registration and lookup
    - Dependency resolution
    - Execution ordering
    - Parallel execution groups
    """

    def __init__(self):
        """Initialize empty registry."""
        self._agents: Dict[AgentRole, BaseAgent] = {}
        self._metadata: Dict[AgentRole, AgentMetadata] = {}

    def register(self, agent: BaseAgent) -> None:
        """
        Register an agent.

        Args:
            agent: Agent instance to register
        """
        role = agent.get_role()
        metadata = agent.get_metadata()

        self._agents[role] = agent
        self._metadata[role] = metadata

        logger.info(f"Registered agent: {role.value}")

    def get_agent(self, role: AgentRole) -> Optional[BaseAgent]:
        """
        Get an agent by role.

        Args:
            role: Agent role

        Returns:
            Agent instance or None if not found
        """
        return self._agents.get(role)

    def get_metadata(self, role: AgentRole) -> Optional[AgentMetadata]:
        """
        Get metadata for an agent.

        Args:
            role: Agent role

        Returns:
            Agent metadata or None if not found
        """
        return self._metadata.get(role)

    def get_all_agents(self) -> List[BaseAgent]:
        """
        Get all registered agents.

        Returns:
            List of all agents
        """
        return list(self._agents.values())

    def get_execution_order(self, required_roles: Optional[List[AgentRole]] = None) -> List[List[AgentRole]]:
        """
        Compute execution order based on dependencies and parallelization.

        Returns list of "stages" where each stage contains agents that can run in parallel.

        Args:
            required_roles: Optional list of roles to include (default: all)

        Returns:
            List of stages, where each stage is a list of roles that can run in parallel

        Example:
            [
                [AgentRole.ANALYST],  # Stage 1: Analyst runs alone
                [AgentRole.PAPER_CURATOR],  # Stage 2: Curator processes papers
                [AgentRole.EVIDENCE_VALIDATOR, AgentRole.BIAS_DETECTOR],  # Stage 3: Parallel validation
                [AgentRole.SKEPTIC],  # Stage 4: Skeptic aggregates
                [AgentRole.SYNTHESIZER]  # Stage 5: Final synthesis
            ]
        """
        if required_roles is None:
            required_roles = list(self._metadata.keys())

        # Build dependency graph
        remaining = set(required_roles)
        completed = set()
        stages = []

        while remaining:
            # Find agents whose dependencies are satisfied
            ready = []
            for role in remaining:
                metadata = self._metadata.get(role)
                if metadata is None:
                    continue

                # Check if all dependencies are completed
                if all(dep in completed for dep in metadata.dependencies):
                    ready.append(role)

            if not ready:
                # Circular dependency or missing agent
                logger.error(f"Cannot resolve dependencies for: {remaining}")
                break

            # Sort ready agents by priority
            ready.sort(key=lambda r: self._metadata[r].priority.value)

            # Group agents that can run in parallel
            parallel_group = []
            sequential_only = []

            for role in ready:
                metadata = self._metadata[role]
                if metadata.can_run_parallel and len(parallel_group) > 0:
                    # Can join parallel group
                    parallel_group.append(role)
                elif metadata.can_run_parallel:
                    # Start new parallel group
                    parallel_group.append(role)
                else:
                    # Must run sequentially
                    sequential_only.append(role)

            # Add parallel group as one stage
            if parallel_group:
                stages.append(parallel_group)
                for role in parallel_group:
                    remaining.remove(role)
                    completed.add(role)

            # Add sequential agents as individual stages
            for role in sequential_only:
                stages.append([role])
                remaining.remove(role)
                completed.add(role)

        return stages

    def validate_dependencies(self) -> List[str]:
        """
        Validate that all agent dependencies are satisfied.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        for role, metadata in self._metadata.items():
            for dep in metadata.dependencies:
                if dep not in self._metadata:
                    errors.append(f"{role.value} depends on {dep.value}, but {dep.value} is not registered")

        # Check for circular dependencies
        for role in self._metadata.keys():
            visited = set()
            stack = [role]

            while stack:
                current = stack.pop()
                if current in visited:
                    errors.append(f"Circular dependency detected involving {role.value}")
                    break
                visited.add(current)

                metadata = self._metadata.get(current)
                if metadata:
                    stack.extend(metadata.dependencies)

        return errors


# =============================================================================
# Global Registry Instance
# =============================================================================

# Global registry for all agents
_global_registry = AgentRegistry()


def get_global_registry() -> AgentRegistry:
    """Get the global agent registry."""
    return _global_registry


def register_agent(agent: BaseAgent) -> None:
    """Register an agent in the global registry."""
    _global_registry.register(agent)
