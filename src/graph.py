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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def route_debate(state: AgentState) -> Literal["analyst", "synthesizer"]:
    """
    Conditional routing function for the dialectical debate loop with natural termination.
    
    Implements FR-T1-006: Conditional edge from Skeptic based on contradiction detection
    Implements T089: Circular argument detection and impasse synthesis
    Implements Natural Termination: Stop when genuinely stuck (Option 3)
    
    Decision Logic (in order):
    1. If circular argument detected: Set impasse mode and route to Synthesizer
    2. If genuinely stuck (consecutive high-similarity rejections): Set stuck mode and route to Synthesizer
    3. If contradiction found AND iterations remain: Loop back to Analyst for refinement
    4. Otherwise: Proceed to Synthesizer for final synthesis
    
    Natural Termination Criteria:
    - 2+ consecutive rejections with similarity >0.75 = genuinely stuck
    - Indicates exhaustion of theoretical approaches
    - More intelligent than arbitrary MAX_ITERATIONS cutoff
    
    Args:
        state: Current AgentState with current_antithesis and iteration_count
    
    Returns:
        "analyst": Route back to analyst for another iteration
        "synthesizer": Proceed to synthesizer for final synthesis
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
            logger.warning(f"   System has exhausted theoretical approaches. Proceeding to synthesis.")
            state["synthesis_mode"] = "impasse"
            return "synthesizer"
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
            return "synthesizer"
    
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
        # Proceed to synthesis
        if iteration_count >= max_iterations:
            logger.info(f"⏭️  ROUTING: Max iterations ({max_iterations}) reached (safety net). Proceeding to Synthesizer.")
            state["synthesis_mode"] = "exhausted_attempts"
        else:
            logger.info(f"✅ ROUTING: No contradiction found. Proceeding to Synthesizer.")
            state["synthesis_mode"] = "standard"
        return "synthesizer"


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
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("increment_iteration", increment_iteration)
    
    # Add edges
    # START → Analyst (always starts with thesis generation)
    workflow.add_edge(START, "analyst")
    
    # Analyst → Skeptic (thesis evaluation)
    workflow.add_edge("analyst", "skeptic")
    
    # Skeptic → [Conditional Routing]
    # This is the key cyclic edge that enables iterative refinement
    workflow.add_conditional_edges(
        "skeptic",
        route_debate,
        {
            "analyst": "increment_iteration",  # Loop back via iteration counter
            "synthesizer": "synthesizer"        # Proceed to synthesis
        }
    )
    
    # Increment → Analyst (complete the loop)
    workflow.add_edge("increment_iteration", "analyst")
    
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

