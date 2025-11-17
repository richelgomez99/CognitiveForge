"""
LangGraph StateGraph for the dialectical synthesis system.

This module builds the core LangGraph workflow that orchestrates the
Analyst, Skeptic, and Synthesizer agents through a cyclic debate process.
"""

import os
import sqlite3
import aiosqlite
import logging
from typing import Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Import AgentState and agent nodes
from src.models import AgentState
from src.agents.analyst import analyst_node
from src.agents.skeptic import skeptic_node
from src.agents.synthesizer import synthesizer_node

# Epic 5: Import new quality check agents
from src.agents.paper_curator import PaperCuratorAgent
from src.agents.evidence_validator import EvidenceValidatorAgent
from src.agents.bias_detector import BiasDetectorAgent
from src.agents.consistency_checker import ConsistencyCheckerAgent
from src.agents.counter_perspective import CounterPerspectiveAgent
from src.agents.novelty_assessor import NoveltyAssessorAgent
from src.agents.synthesis_reviewer import SynthesisReviewerAgent
from src.agents.base_agent import AgentRegistry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def route_debate(state: AgentState) -> Literal["analyst", "quality_check"]:
    """
    Conditional routing function for the dialectical debate loop with natural termination.

    Epic 5 Update: Routes to quality_check instead of directly to synthesizer.

    Implements FR-T1-006: Conditional edge from Skeptic based on contradiction detection
    Implements T089: Circular argument detection and impasse synthesis
    Implements Natural Termination: Stop when genuinely stuck (Option 3)

    Decision Logic (in order):
    1. If circular argument detected: Set impasse mode and route to Quality Check
    2. If genuinely stuck (consecutive high-similarity rejections): Set stuck mode and route to Quality Check
    3. If contradiction found AND iterations remain: Loop back to Analyst for refinement
    4. Otherwise: Proceed to Quality Check for validation before synthesis

    Natural Termination Criteria:
    - 2+ consecutive rejections with similarity >0.75 = genuinely stuck
    - Indicates exhaustion of theoretical approaches
    - More intelligent than arbitrary MAX_ITERATIONS cutoff

    Args:
        state: Current AgentState with current_antithesis and iteration_count

    Returns:
        "analyst": Route back to analyst for another iteration
        "quality_check": Proceed to quality check orchestration then synthesis
    """
    # Get max iterations from environment (default: 10, increased from 3 as safety net)
    max_iterations = int(os.getenv("MAX_ITERATIONS", "10"))
    
    # Extract state
    antithesis = state.get("current_antithesis")
    iteration_count = state.get("iteration_count", 0)
    last_similarity_score = state.get("last_similarity_score")
    consecutive_high_similarity_count = state.get("consecutive_high_similarity_count", 0)
    
    # NATURAL TERMINATION: Track consecutive high-similarity rejections
    HIGH_SIMILARITY_THRESHOLD = 0.75  # Looser than circular detection (0.80)
    STUCK_THRESHOLD = 2  # 2 consecutive high-similarity rejections = stuck
    
    if last_similarity_score is not None and last_similarity_score >= HIGH_SIMILARITY_THRESHOLD:
        # High similarity detected
        consecutive_high_similarity_count += 1
        state["consecutive_high_similarity_count"] = consecutive_high_similarity_count
        
        logger.info(f"📊 ROUTING: High similarity detected: {last_similarity_score:.2f}")
        logger.info(f"   Consecutive high-similarity count: {consecutive_high_similarity_count}/{STUCK_THRESHOLD}")
        
        # Check if genuinely stuck
        if consecutive_high_similarity_count >= STUCK_THRESHOLD:
            logger.warning(f"⛔ ROUTING: GENUINELY STUCK detected ({consecutive_high_similarity_count} consecutive high-similarity rejections)")
            logger.warning(f"   System has exhausted theoretical approaches. Proceeding to quality check.")
            state["synthesis_mode"] = "impasse"
            return "quality_check"
    else:
        # Reset counter if similarity is low
        if consecutive_high_similarity_count > 0:
            logger.info(f"📊 ROUTING: Similarity reset (was {consecutive_high_similarity_count} consecutive)")
        state["consecutive_high_similarity_count"] = 0
    
    # T089: Check if circular argument was detected (Tier 1 bug fix)
    if antithesis and antithesis.critique:
        critique_lower = antithesis.critique.lower()
        circular_keywords = ["circular", "identical", "similarity", "redundant", "same as"]
        is_circular = any(keyword in critique_lower for keyword in circular_keywords)
        
        if is_circular:
            logger.info("🔄 ROUTING: Circular argument detected. Setting impasse mode.")
            logger.info(f"   Critique: {antithesis.critique[:150]}...")
            # Set synthesis mode to impasse (will be picked up by synthesizer)
            state["synthesis_mode"] = "impasse"
            return "quality_check"
    
    # Tier 2 (T038): Populate conversation_history before routing
    thesis = state.get("current_thesis")
    papers_analyst = state.get("current_round_papers_analyst", [])
    papers_skeptic = state.get("current_round_papers_skeptic", [])
    
    if thesis and antithesis:
        # Create conversation round
        from src.models import ConversationRound
        conversation_history = state.get("conversation_history", [])
        round_num = len(conversation_history) + 1
        
        new_round: ConversationRound = {
            "round_number": round_num,
            "thesis": thesis,
            "antithesis": antithesis,
            "papers_analyst": papers_analyst,
            "papers_skeptic": papers_skeptic
        }
        
        conversation_history.append(new_round)
        state["conversation_history"] = conversation_history
        logger.info(f"📝 Added Round {round_num} to conversation history")
    
    # Check if we should loop back
    if antithesis and antithesis.contradiction_found and iteration_count < max_iterations:
        # Increment iteration count (will be incremented in state update)
        next_iteration = iteration_count + 1
        logger.info(
            f"🔄 ROUTING: Contradiction found. Looping back to Analyst "
            f"(iteration {next_iteration}/{max_iterations})"
        )
        logger.info(f"   Contradiction: {antithesis.critique[:100]}...")
        state["synthesis_mode"] = "standard"  # Standard synthesis if we reach it
        return "analyst"
    else:
        # Proceed to quality check and synthesis
        if iteration_count >= max_iterations:
            logger.info(f"⏭️  ROUTING: Max iterations ({max_iterations}) reached (safety net). Proceeding to Quality Check.")
            state["synthesis_mode"] = "exhausted_attempts"
        else:
            logger.info(f"✅ ROUTING: No contradiction found. Proceeding to Quality Check.")
            state["synthesis_mode"] = "standard"
        return "quality_check"


def increment_iteration(state: AgentState) -> dict:
    """
    Helper node to increment iteration count when looping back.

    This ensures we track how many refinement cycles we've gone through.

    Args:
        state: Current AgentState

    Returns:
        Dict with incremented iteration_count
    """
    current_count = state.get("iteration_count", 0)
    new_count = current_count + 1
    logger.info(f"📊 Iteration count: {current_count} → {new_count}")
    return {"iteration_count": new_count}


def quality_check_orchestration(state: AgentState) -> dict:
    """
    Epic 5: Orchestrate the 7 new quality check agents before synthesis.

    This node runs between the debate loop and final synthesis to ensure
    high-quality output through multi-agent validation.

    Execution flow (via AgentRegistry):
    1. Paper Curator - Filter & rank discovered papers
    2. Evidence Validator & Bias Detector (parallel) - Validate evidence & detect biases
    3. Consistency Checker - Check logical consistency
    4. Counter-Perspective & Novelty Assessor (parallel) - Generate alternatives & assess novelty

    Note: Synthesis Reviewer runs AFTER synthesizer as final QA gate.

    Args:
        state: Current AgentState with thesis, papers, etc.

    Returns:
        Dict with agent outputs added to state
    """
    logger.info("🔍 ==== QUALITY CHECK ORCHESTRATION ====")

    # Initialize agents
    paper_curator = PaperCuratorAgent()
    evidence_validator = EvidenceValidatorAgent()
    bias_detector = BiasDetectorAgent()
    consistency_checker = ConsistencyCheckerAgent()
    counter_perspective = CounterPerspectiveAgent()
    novelty_assessor = NoveltyAssessorAgent()

    # Register agents
    registry = AgentRegistry()
    registry.register(paper_curator)
    registry.register(evidence_validator)
    registry.register(bias_detector)
    registry.register(consistency_checker)
    registry.register(counter_perspective)
    registry.register(novelty_assessor)

    # Execute agents in dependency order with parallelization
    logger.info("📋 Calculating execution order...")
    execution_stages = registry.get_execution_order()

    updated_state = dict(state)

    for stage_num, stage_agents in enumerate(execution_stages, 1):
        agent_names = [a.get_role().value for a in stage_agents]
        logger.info(f"🎯 Stage {stage_num}: Executing {len(stage_agents)} agents - {agent_names}")

        # Run agents in this stage (they can run in parallel)
        for agent in stage_agents:
            result = agent.run(updated_state)
            updated_state.update(result)

    logger.info("✅ Quality check orchestration complete")

    return updated_state


def build_graph(use_checkpointer: bool = False, db_path: str = "dialectical_system.db", use_async: bool = False) -> StateGraph:
    """
    Build and compile the dialectical synthesis LangGraph.
    
    Graph Structure:
    1. START → Analyst: Generate initial thesis
    2. Analyst → Skeptic: Evaluate thesis for contradictions
    3. Skeptic → [Conditional]:
       - If contradiction & iterations remain → Increment → Analyst (loop)
       - Otherwise → Synthesizer
    4. Synthesizer → END: Generate final synthesis
    
    Implements:
    - FR-T1-001 through FR-T1-006: Core dialectical engine
    - FR-T1-007: KG integration (within agent nodes)
    - FR-T1-016: Gemini retry logic (within gemini_client)
    - FR-T2-010: State persistence via SqliteSaver checkpointer (Tier 2)
    
    Args:
        use_checkpointer: If True, enable state persistence with SqliteSaver.
                         Required for Tier 2 features (streaming, state queries).
        db_path: Path to SQLite database for checkpointing (default: dialectical_system.db)
        use_async: If True, use AsyncSqliteSaver for async operations (FastAPI streaming).
                  If False, use SqliteSaver for sync operations (CLI mode).
    
    Returns:
        Compiled StateGraph ready for execution
    
    Tier 1 Usage (CLI mode, no persistence):
        >>> graph = build_graph(use_checkpointer=False)
        >>> import uuid
        >>> result = graph.invoke({
        ...     "original_query": "What are the limitations of transformers?",
        ...     "messages": [],
        ...     "iteration_count": 0,
        ...     "procedural_memory": "",
        ...     "debate_memory": {"rejected_claims": [], "skeptic_objections": [], "weak_evidence_urls": []},
        ...     "current_claim_id": str(uuid.uuid4())
        ... })
    
    Tier 2 Usage (API mode with persistence):
        >>> graph = build_graph(use_checkpointer=True)
        >>> config = {"configurable": {"thread_id": "session_123"}}
        >>> import uuid
        >>> result = graph.invoke({
        ...     "original_query": "What are the limitations of transformers?",
        ...     "messages": [],
        ...     "iteration_count": 0,
        ...     "procedural_memory": "",
        ...     "debate_memory": {"rejected_claims": [], "skeptic_objections": [], "weak_evidence_urls": []},
        ...     "current_claim_id": str(uuid.uuid4())
        ... }, config=config)
        >>> # Resume from checkpoint
        >>> state = graph.get_state(config)
        >>> history = graph.get_state_history(config)
    
    Note:
        When use_checkpointer=True, you MUST provide a thread_id in the config
        parameter when invoking the graph. The thread_id is used to isolate
        different conversation sessions.
    """
    logger.info("🏗️  Building dialectical synthesis graph...")
    
    # Initialize StateGraph with AgentState schema
    workflow = StateGraph(AgentState)
    
    # Add agent nodes
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("skeptic", skeptic_node)
    workflow.add_node("quality_check", quality_check_orchestration)  # Epic 5: Quality check agents
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("increment_iteration", increment_iteration)

    # Add edges
    # START → Analyst (always starts with thesis generation)
    workflow.add_edge(START, "analyst")

    # Analyst → Skeptic (thesis evaluation)
    workflow.add_edge("analyst", "skeptic")

    # Skeptic → [Conditional Routing]
    # This is the key cyclic edge that enables iterative refinement
    # Epic 5: Now routes to quality_check instead of directly to synthesizer
    workflow.add_conditional_edges(
        "skeptic",
        route_debate,
        {
            "analyst": "increment_iteration",      # Loop back via iteration counter
            "quality_check": "quality_check"       # Epic 5: Proceed to quality checks
        }
    )

    # Epic 5: Quality Check → Synthesizer (after all quality checks pass)
    workflow.add_edge("quality_check", "synthesizer")

    # Increment → Analyst (complete the loop)
    workflow.add_edge("increment_iteration", "analyst")

    # Epic 5: Synthesizer → Synthesis Review (final QA gate)
    # Note: We add synthesis review as a function that runs within the synthesizer edge
    # to keep the graph simple, but log it for transparency
    logger.info("📝 Epic 5: Synthesis Reviewer will run post-synthesis (in synthesizer node or as post-processing)")

    # Synthesizer → END (final output)
    workflow.add_edge("synthesizer", END)
    
    # Tier 2: Add checkpointer for state persistence
    if use_checkpointer:
        if use_async:
            # Use AsyncSqliteSaver for FastAPI async endpoints (streaming)
            logger.info(f"🗄️  Initializing AsyncSqliteSaver checkpointer: {db_path}")
            
            # Create async SQLite connection
            # Note: This creates a connection that can be used asynchronously
            import asyncio
            
            async def setup_async_checkpointer():
                async with aiosqlite.connect(db_path) as conn:
                    checkpointer = AsyncSqliteSaver(conn)
                    await checkpointer.setup()
            
            # Run setup synchronously (this is a one-time operation)
            try:
                asyncio.run(setup_async_checkpointer())
            except RuntimeError:
                # Already in an event loop, skip setup (will be done on first use)
                pass
            
            # Create a connection string-based checkpointer for the graph
            # We'll use the context manager approach properly in the graph
            checkpointer = AsyncSqliteSaver.from_conn_string(db_path)
            compiled_graph = workflow.compile(checkpointer=checkpointer)
            logger.info("✅ Graph compiled with AsyncSqliteSaver (async mode)")
        else:
            # Use SqliteSaver for synchronous operations (CLI mode)
            logger.info(f"🗄️  Initializing SqliteSaver checkpointer: {db_path}")
            # Create a persistent SQLite connection for the checkpointer
            # Note: check_same_thread=False allows the connection to be used across threads
            conn = sqlite3.connect(db_path, check_same_thread=False)
            
            # Create SqliteSaver with the connection
            checkpointer = SqliteSaver(conn)
            
            # Setup the database schema (creates tables if they don't exist)
            checkpointer.setup()
            
            # Compile graph with checkpointer
            compiled_graph = workflow.compile(checkpointer=checkpointer)
            logger.info("✅ Graph compiled with SqliteSaver (sync mode)")
        
        logger.info(f"   ⚠️  IMPORTANT: You must provide a thread_id in config when invoking:")
        logger.info(f"       config = {{'configurable': {{'thread_id': 'your_session_id'}}}}")
        logger.info(f"   Database: {db_path}")
    else:
        # Tier 1: No checkpointer (CLI mode)
        compiled_graph = workflow.compile()
        logger.info("✅ Graph compiled (no checkpointer - CLI mode)")
    
    logger.info(f"   Nodes: analyst, skeptic, synthesizer, increment_iteration")
    logger.info(f"   Cyclic loop: skeptic → increment_iteration → analyst (conditional)")
    logger.info(f"   Max iterations: {os.getenv('MAX_ITERATIONS', '3')}")
    
    return compiled_graph


# Convenience function for graph visualization (optional, for debugging)
def visualize_graph(graph: StateGraph, output_path: str = "graph_visualization.png"):
    """
    Generate a visualization of the graph structure.
    
    Requires graphviz to be installed: pip install graphviz
    
    Args:
        graph: Compiled StateGraph
        output_path: Path to save visualization image
    """
    try:
        from IPython.display import Image, display
        viz = graph.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(viz)
        logger.info(f"📊 Graph visualization saved to {output_path}")
        return viz
    except ImportError:
        logger.warning("IPython or graphviz not available. Skipping visualization.")
        return None
    except Exception as e:
        logger.warning(f"Could not generate graph visualization: {e}")
        return None


if __name__ == "__main__":
    # Test graph compilation
    print("Testing graph compilation...")
    graph = build_graph()
    print("✅ Graph compilation test passed!")
    
    # Try to visualize (optional)
    try:
        visualize_graph(graph)
        print("✅ Graph visualization generated")
    except:
        print("ℹ️  Graph visualization skipped (requires graphviz)")

