"""
FastAPI backend for CognitiveForge - Tier 2.

Provides REST and SSE streaming endpoints for the dialectical synthesis system.

Implements:
- FR-T2-012: API key authentication via X-API-Key header
- FR-T2-013: 401 responses for missing/invalid API keys
- FR-T2-014: Public health endpoint (no authentication)
- SC-T2-001: Response time < 2 seconds for state queries

Endpoints:
- GET /health: Public health check (no auth required)
- GET /stream_dialectics/{thread_id}: SSE streaming of graph execution (requires auth)
- GET /get_state/{thread_id}: Query current state (requires auth)
- GET /get_trace/{thread_id}: Query checkpoint history (requires auth)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Import authentication
from src.auth import get_api_key

# Import graph builder
from src.graph import build_graph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global graph cache
_graph_cache = {}
_checkpointer = None


# =============================================================================
# Lifespan Context Manager (for checkpointer)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    
    Manages the AsyncSqliteSaver checkpointer lifecycle using async context manager.
    The checkpointer must be kept alive throughout the application's lifetime.
    """
    global _checkpointer, _graph_cache
    
    logger.info("=" * 80)
    logger.info("🚀 CognitiveForge API Server Starting...")
    logger.info("=" * 80)
    logger.info("Tier 2: Persistence & Real-Time Visibility")
    
    # Initialize AsyncSqliteSaver checkpointer (MUST use async with to keep it alive)
    async with AsyncSqliteSaver.from_conn_string("dialectical_system.db") as checkpointer:
        _checkpointer = checkpointer
        
        # Setup the database schema
        await checkpointer.setup()
        logger.info("🗄️  AsyncSqliteSaver initialized: dialectical_system.db")
        
        # Build graph with the checkpointer
        # Note: We pass the sync version of build_graph but with use_checkpointer=False
        # because we'll manually use the async checkpointer
        from langgraph.graph import StateGraph
        from src.models import AgentState
        from src.agents.analyst import analyst_node
        from src.agents.skeptic import skeptic_node
        from src.agents.synthesizer import synthesizer_node
        from src.graph import route_debate, increment_iteration
        from langgraph.graph import START, END
        
        # Build workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("analyst", analyst_node)
        workflow.add_node("skeptic", skeptic_node)
        workflow.add_node("synthesizer", synthesizer_node)
        workflow.add_node("increment_iteration", increment_iteration)
        workflow.add_edge(START, "analyst")
        workflow.add_edge("analyst", "skeptic")
        workflow.add_conditional_edges(
            "skeptic",
            route_debate,
            {
                "analyst": "increment_iteration",
                "synthesizer": "synthesizer"
            }
        )
        workflow.add_edge("increment_iteration", "analyst")
        workflow.add_edge("synthesizer", END)
        
        # Compile with async checkpointer
        _graph_cache['async_graph'] = workflow.compile(checkpointer=checkpointer)
        logger.info("✅ Graph compiled with AsyncSqliteSaver")
        
        logger.info("")
        logger.info("Endpoints:")
        logger.info("  - GET  /health                          (public)")
        logger.info("  - GET  /get_state/{thread_id}           (requires X-API-Key)")
        logger.info("  - GET  /get_trace/{thread_id}           (requires X-API-Key)")
        logger.info("  - GET  /stream_dialectics/{thread_id}   (requires X-API-Key)")
        logger.info("")
        logger.info("Documentation:")
        logger.info("  - Swagger UI: http://localhost:8000/docs")
        logger.info("  - ReDoc:      http://localhost:8000/redoc")
        logger.info("=" * 80)
        
        yield  # Server runs here
        
    # Cleanup on shutdown
    logger.info("=" * 80)
    logger.info("🛑 CognitiveForge API Server Shutting Down...")
    logger.info("=" * 80)
    _checkpointer = None
    _graph_cache.clear()


# =============================================================================
# FastAPI App Initialization
# =============================================================================

app = FastAPI(
    title="CognitiveForge - Dialectical Synthesis API",
    description="Multi-agent dialectical reasoning system with state persistence",
    version="2.0.0",  # Tier 2
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan  # Use lifespan context manager
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health Check Endpoint (Public, No Authentication)
# =============================================================================

@app.get(
    "/health",
    tags=["Health"],
    summary="Health check endpoint",
    description="Check system health without authentication. Returns status of Neo4j, Gemini API, and checkpointer.",
    response_model=Dict[str, Any]
)
async def health_check():
    """
    Public health check endpoint.
    
    Implements FR-T2-014: Health endpoint WITHOUT authentication.
    
    Checks:
    - Neo4j connection (driver.verify_connectivity())
    - Gemini API key exists in environment
    - Checkpointer database exists
    
    Returns:
        {
            "status": "healthy" | "unhealthy",
            "neo4j": bool,
            "gemini": bool,
            "checkpointer": bool
        }
    
    Status Codes:
        200: System is healthy
        200: System has issues (status: unhealthy, with details)
    
    Note:
        This endpoint does NOT require authentication (public access).
    """
    health_status = {
        "status": "healthy",
        "neo4j": False,
        "gemini": False,
        "checkpointer": False
    }
    
    # Check Neo4j connection
    try:
        from neo4j import GraphDatabase
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password123")
        
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        driver.verify_connectivity()
        driver.close()
        
        health_status["neo4j"] = True
        logger.info("✅ Neo4j connection: OK")
        
    except Exception as e:
        health_status["neo4j"] = False
        health_status["status"] = "unhealthy"
        logger.error(f"❌ Neo4j connection failed: {e}")
    
    # Check Gemini API key
    try:
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if gemini_api_key and len(gemini_api_key) > 0:
            health_status["gemini"] = True
            logger.info("✅ Gemini API key: OK")
        else:
            health_status["gemini"] = False
            health_status["status"] = "unhealthy"
            logger.error("❌ Gemini API key: Missing")
    except Exception as e:
        health_status["gemini"] = False
        health_status["status"] = "unhealthy"
        logger.error(f"❌ Gemini API key check failed: {e}")
    
    # Check checkpointer database
    try:
        db_path = Path("dialectical_system.db")
        if db_path.exists() and db_path.is_file():
            health_status["checkpointer"] = True
            logger.info("✅ Checkpointer database: OK")
        else:
            # Database doesn't exist yet - this is OK on first run
            health_status["checkpointer"] = False
            logger.warning("⚠️  Checkpointer database: Not yet created (will be created on first use)")
    except Exception as e:
        health_status["checkpointer"] = False
        logger.error(f"❌ Checkpointer database check failed: {e}")
    
    # Return status
    if health_status["status"] == "healthy":
        logger.info(f"🟢 Health check: {health_status}")
        return health_status
    else:
        logger.warning(f"🟡 Health check: {health_status}")
        # Return 200 with unhealthy status (not 503) to allow clients to see details
        return health_status


# =============================================================================
# State Query Endpoint (Requires Authentication)
# =============================================================================

@app.get(
    "/get_state/{thread_id}",
    tags=["State"],
    summary="Get current state for a thread",
    description="Query the current state of a dialectical session by thread_id. Requires X-API-Key header.",
    response_model=Dict[str, Any]
)
async def get_state(
    thread_id: str,
    api_key: str = Depends(get_api_key)
):
    """
    Get the current state for a specific thread_id.
    
    Implements:
    - FR-T2-012: API key authentication required
    - FR-T2-013: 401 for missing/invalid API key
    - SC-T2-001: Response time < 2 seconds
    
    Args:
        thread_id: The session thread ID
        api_key: API key from X-API-Key header (injected by dependency)
    
    Returns:
        AgentState JSON with all state fields
    
    Raises:
        401: If API key is missing or invalid (handled by get_api_key dependency)
        404: If thread_id not found in checkpointer
    
    Example:
        curl -H "X-API-Key: your_secret_key" \\
             http://localhost:8000/get_state/session_123
    """
    import time
    start_time = time.time()
    
    try:
        # Get graph from cache (initialized at startup via lifespan)
        graph = _graph_cache.get('async_graph')
        if not graph:
            raise HTTPException(status_code=503, detail="Service not ready: Graph not initialized")
        
        # Get state for thread
        config = {"configurable": {"thread_id": thread_id}}
        state = graph.get_state(config)
        
        # Check if state exists
        if state is None or not state.values:
            logger.warning(f"Thread not found: {thread_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Thread '{thread_id}' not found"
            )
        
        # Convert state to JSON-serializable format
        state_dict = dict(state.values)
        
        # Convert Pydantic models to dicts if present
        if state_dict.get("current_thesis") and hasattr(state_dict["current_thesis"], "model_dump"):
            state_dict["current_thesis"] = state_dict["current_thesis"].model_dump()
        if state_dict.get("current_antithesis") and hasattr(state_dict["current_antithesis"], "model_dump"):
            state_dict["current_antithesis"] = state_dict["current_antithesis"].model_dump()
        if state_dict.get("final_synthesis") and hasattr(state_dict["final_synthesis"], "model_dump"):
            state_dict["final_synthesis"] = state_dict["final_synthesis"].model_dump()
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Retrieved state for thread '{thread_id}' in {elapsed_time:.3f}s")
        
        # Warn if response time exceeds 2 seconds (SC-T2-001)
        if elapsed_time > 2.0:
            logger.warning(f"⚠️  Response time exceeded 2s: {elapsed_time:.3f}s (SC-T2-001)")
        
        return state_dict
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving state for thread '{thread_id}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# =============================================================================
# Streaming Endpoint (Requires Authentication)
# =============================================================================

@app.get(
    "/stream_dialectics/{thread_id}",
    tags=["Streaming"],
    summary="Stream dialectical synthesis execution",
    description="Real-time SSE streaming of agent execution. Requires X-API-Key header.",
)
async def stream_dialectics(
    thread_id: str,
    query: str = Query(..., description="Research query to process"),
    api_key: str = Depends(get_api_key)
):
    """
    Stream the dialectical synthesis process via Server-Sent Events (SSE).
    
    Implements:
    - FR-T2-012: API key authentication required
    - SC-T2-003: First event arrives within 500ms
    
    This endpoint provides real-time visibility into the agent execution process,
    streaming updates as each node (Analyst, Skeptic, Synthesizer) completes.
    
    Args:
        thread_id: The session thread ID for state persistence
        query: The research query to process
        api_key: API key from X-API-Key header (injected by dependency)
    
    Returns:
        Server-Sent Events stream with format:
        data: {"node": "analyst", "data": {...}}
        
    Event Format:
        Each event contains:
        - node: Name of the agent node that produced this update
        - data: The node's output (thesis, antithesis, or synthesis)
    
    Raises:
        401: If API key is missing or invalid
        400: If query parameter is missing
        503: If Gemini API rate limit is exceeded after retries
    
    Example:
        curl -N -H "X-API-Key: your_secret_key" \\
             "http://localhost:8000/stream_dialectics/session_123?query=What+is+LangGraph"
    
    Client Example (Python):
        import requests
        
        response = requests.get(
            "http://localhost:8000/stream_dialectics/session_123",
            params={"query": "What is LangGraph?"},
            headers={"X-API-Key": "your_secret_key"},
            stream=True
        )
        
        for line in response.iter_lines():
            if line.startswith(b"data: "):
                event = json.loads(line[6:])
                print(f"{event['node']}: {event['data']}")
    """
    
    async def event_generator() -> AsyncIterator[str]:
        """
        Async generator that streams graph execution updates as SSE events.
        
        Yields:
            SSE-formatted strings: "data: {json}\n\n"
        """
        try:
            # Get graph from cache (initialized at startup via lifespan)
            graph = _graph_cache.get('async_graph')
            if not graph:
                error_event = {"node": "error", "data": {"error": "Service not ready: Graph not initialized"}}
                yield f"data: {json.dumps(error_event)}\n\n"
                return
            
            # Prepare initial state
            inputs = {
                "messages": [],
                "original_query": query,
                "current_thesis": None,
                "current_antithesis": None,
                "final_synthesis": None,
                "contradiction_report": "",
                "iteration_count": 0,
                "procedural_memory": ""
            }
            
            # Configuration with thread_id
            config = {"configurable": {"thread_id": thread_id}}
            
            logger.info(f"🚀 Starting streaming execution for thread '{thread_id}'")
            logger.info(f"   Query: {query}")
            
            # Stream graph execution
            event_count = 0
            async for chunk in graph.astream(inputs, config=config, stream_mode="updates"):
                # chunk is a dict with node name as key
                for node_name, node_output in chunk.items():
                    event_count += 1
                    
                    # Convert Pydantic models to dicts for JSON serialization
                    serializable_output = {}
                    for key, value in node_output.items():
                        if hasattr(value, "model_dump"):
                            # It's a Pydantic model
                            serializable_output[key] = value.model_dump()
                        elif isinstance(value, list) and len(value) > 0 and hasattr(value[0], "model_dump"):
                            # It's a list of Pydantic models
                            serializable_output[key] = [item.model_dump() for item in value]
                        else:
                            # Primitive type or already serializable
                            serializable_output[key] = value
                    
                    # Create SSE event
                    event_data = {
                        "node": node_name,
                        "data": serializable_output
                    }
                    
                    # Format as SSE: "data: {json}\n\n"
                    sse_message = f"data: {json.dumps(event_data)}\n\n"
                    
                    logger.info(f"📤 Event {event_count}: {node_name}")
                    yield sse_message
            
            logger.info(f"✅ Streaming completed for thread '{thread_id}' ({event_count} events)")
            
        except Exception as e:
            logger.error(f"❌ Streaming error for thread '{thread_id}': {e}")
            
            # Send error event
            error_event = {
                "node": "error",
                "data": {
                    "error": str(e),
                    "thread_id": thread_id
                }
            }
            yield f"data: {json.dumps(error_event)}\n\n"
            
            # Check if it's a rate limit error
            if "rate limit" in str(e).lower() or "429" in str(e):
                logger.error("🚫 Gemini API rate limit exceeded")
                raise HTTPException(
                    status_code=503,
                    detail="Service temporarily unavailable: API rate limit exceeded. Please try again later."
                )
    
    # Return SSE stream
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


# =============================================================================
# Trace Query Endpoint (Requires Authentication)
# =============================================================================

@app.get(
    "/get_trace/{thread_id}",
    tags=["State"],
    summary="Get checkpoint history for a thread",
    description="Query the full checkpoint history showing thesis evolution. Requires X-API-Key header.",
    response_model=list
)
async def get_trace(
    thread_id: str,
    api_key: str = Depends(get_api_key)
):
    """
    Get the checkpoint history (trace) for a specific thread_id.
    
    Implements:
    - SC-T2-002: Response time < 5 seconds for ≤10 checkpoints
    - SC-T2-006: Checkpoint count matches iteration_count
    
    Args:
        thread_id: The session thread ID
        api_key: API key from X-API-Key header (injected by dependency)
    
    Returns:
        Array of checkpoint objects with:
        - checkpoint_id: Unique identifier
        - iteration_count: Iteration number
        - current_thesis: Thesis at this checkpoint
        - current_antithesis: Antithesis at this checkpoint
        - final_synthesis: Synthesis (if final checkpoint)
        - metadata: Checkpoint metadata
    
    Raises:
        401: If API key is missing or invalid
        404: If thread_id not found
    
    Example:
        curl -H "X-API-Key: your_secret_key" \\
             http://localhost:8000/get_trace/session_123
    """
    import time
    start_time = time.time()
    
    try:
        # Get graph from cache (initialized at startup via lifespan)
        graph = _graph_cache.get('async_graph')
        if not graph:
            raise HTTPException(status_code=503, detail="Service not ready: Graph not initialized")
        
        # Get state history for thread
        config = {"configurable": {"thread_id": thread_id}}
        history = list(graph.get_state_history(config))
        
        # Check if history exists
        if not history or len(history) == 0:
            logger.warning(f"No trace found for thread: {thread_id}")
            raise HTTPException(
                status_code=404,
                detail=f"No trace found for thread '{thread_id}'"
            )
        
        # Convert history to JSON-serializable format
        trace = []
        for checkpoint in history:
            checkpoint_data = {
                "checkpoint_id": checkpoint.config.get("configurable", {}).get("checkpoint_id"),
                "iteration_count": checkpoint.values.get("iteration_count", 0),
                "metadata": checkpoint.metadata if hasattr(checkpoint, "metadata") else {}
            }
            
            # Add thesis if present
            if checkpoint.values.get("current_thesis"):
                thesis = checkpoint.values["current_thesis"]
                checkpoint_data["current_thesis"] = thesis.model_dump() if hasattr(thesis, "model_dump") else thesis
            
            # Add antithesis if present
            if checkpoint.values.get("current_antithesis"):
                antithesis = checkpoint.values["current_antithesis"]
                checkpoint_data["current_antithesis"] = antithesis.model_dump() if hasattr(antithesis, "model_dump") else antithesis
            
            # Add synthesis if present (final checkpoint)
            if checkpoint.values.get("final_synthesis"):
                synthesis = checkpoint.values["final_synthesis"]
                checkpoint_data["final_synthesis"] = synthesis.model_dump() if hasattr(synthesis, "model_dump") else synthesis
            
            trace.append(checkpoint_data)
        
        elapsed_time = time.time() - start_time
        checkpoint_count = len(trace)
        logger.info(f"✅ Retrieved {checkpoint_count} checkpoints for thread '{thread_id}' in {elapsed_time:.3f}s")
        
        # Warn if response time exceeds 5 seconds for ≤10 checkpoints (SC-T2-002)
        if checkpoint_count <= 10 and elapsed_time > 5.0:
            logger.warning(f"⚠️  Response time exceeded 5s for {checkpoint_count} checkpoints: {elapsed_time:.3f}s (SC-T2-002)")
        
        return trace
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving trace for thread '{thread_id}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# Note: Startup and shutdown are now handled by the lifespan context manager above


# =============================================================================
# Main Entry Point (for testing)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting FastAPI server in development mode...")
    logger.info("Use 'uvicorn src.api:app --reload --port 8000' for production")
    
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

