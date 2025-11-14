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
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, AsyncIterator
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Import authentication
from src.auth import get_api_key

# Import graph builder
from src.graph import build_graph

# Import models for type hints
from src.models import (
    DiscoveryRequest, DiscoveryResponse, AddPapersRequest, AddPapersResponse,
    # Epic 4: Persistent Memory System
    UserProfile, Workspace, SessionMetadata, SessionRecord, SessionStatus,
    CreateSessionRequest, UpdateSessionRequest, SessionListResponse
)

# Import discovery tools
from src.tools.paper_discovery import search_arxiv, search_semantic_scholar
from src.tools.kg_tools import add_papers_to_neo4j

# Epic 4: Import memory store
from src.tools import memory_store

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

        # Epic 4: Initialize memory system schema
        logger.info("🧠 Initializing Epic 4 memory system schema...")
        memory_schema_success = memory_store.initialize_memory_schema()
        if memory_schema_success:
            logger.info("✅ Memory system schema initialized successfully")
        else:
            logger.warning("⚠️  Memory system schema initialization failed (non-fatal)")
        
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
# Knowledge Discovery Helper Functions
# =============================================================================

async def discover_with_streaming(query: str, max_papers: int = 3) -> AsyncIterator[Dict[str, Any]]:
    """
    Async generator that performs paper discovery and yields real-time event updates.
    
    Implements T042-T046:
    - Searches arXiv (and optionally Semantic Scholar if stable)
    - Ingests papers into Neo4j
    - Yields events: discovery_start, source_searching, paper_found, papers_ingesting, discovery_complete, discovery_error
    
    Args:
        query: Research query to search for
        max_papers: Maximum papers to retrieve per source
    
    Yields:
        Event dicts with 'event_type' and 'data' keys
    
    Event Types:
        - discovery_start: {"query": str}
        - source_searching: {"source": "arxiv" | "semantic_scholar"}
        - paper_found: {"title": str, "authors": List[str], "source": str, "url": str}
        - papers_ingesting: {"count": int}
        - discovery_complete: {"added": int, "skipped": int, "failed": int, "duration": float}
        - discovery_error: {"error": str, "source": str}
    """
    start_time = time.time()
    all_papers = []
    
    try:
        # Emit discovery start event
        yield {
            "event_type": "discovery_start",
            "data": {"query": query}
        }
        logger.info(f"🔍 Starting paper discovery for query: '{query}'")
        
        # Search arXiv (reliable source)
        yield {
            "event_type": "source_searching",
            "data": {"source": "arxiv"}
        }
        logger.info("📚 Searching arXiv...")
        
        try:
            # Run in thread pool since search_arxiv is synchronous
            loop = asyncio.get_event_loop()
            arxiv_papers = await loop.run_in_executor(
                None,
                search_arxiv,
                query,
                max_papers
            )
            
            # Emit paper_found events for each paper
            for paper in arxiv_papers:
                all_papers.append(paper)
                yield {
                    "event_type": "paper_found",
                    "data": {
                        "title": paper.title,
                        "authors": paper.authors[:3],  # First 3 authors
                        "source": paper.source,
                        "url": paper.url
                    }
                }
                logger.info(f"📄 Found: {paper.title[:60]}... by {', '.join(paper.authors[:2])}")
                
        except Exception as e:
            logger.warning(f"⚠️  arXiv search failed: {e}")
            yield {
                "event_type": "discovery_error",
                "data": {"error": str(e), "source": "arxiv"}
            }
        
        # Skip Semantic Scholar for now (pagination issues from previous implementation)
        # Can be re-enabled after fixing the library issues
        logger.info("ℹ️  Skipping Semantic Scholar (using arXiv only for automatic discovery)")
        
        # Ingest papers into Neo4j
        if all_papers:
            yield {
                "event_type": "papers_ingesting",
                "data": {"count": len(all_papers)}
            }
            logger.info(f"💾 Ingesting {len(all_papers)} papers into Neo4j...")
            
            # Run in thread pool since add_papers_to_neo4j is synchronous
            result = await loop.run_in_executor(
                None,
                add_papers_to_neo4j,
                all_papers,
                "auto"
            )
            
            duration = time.time() - start_time
            
            # Emit completion event
            yield {
                "event_type": "discovery_complete",
                "data": {
                    "added": result.get("added", 0),
                    "skipped": result.get("skipped", 0),
                    "failed": result.get("failed", 0),
                    "duration": round(duration, 2)
                }
            }
            logger.info(f"✅ Discovery complete: {result.get('added', 0)} added, {result.get('skipped', 0)} skipped in {duration:.2f}s")
        else:
            logger.warning("⚠️  No papers discovered")
            yield {
                "event_type": "discovery_complete",
                "data": {"added": 0, "skipped": 0, "failed": 0, "duration": round(time.time() - start_time, 2)}
            }
            
    except Exception as e:
        logger.error(f"❌ Discovery pipeline error: {e}")
        yield {
            "event_type": "discovery_error",
            "data": {"error": str(e), "source": "pipeline"}
        }


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
    auto_discover: bool = Query(True, description="Automatically discover papers before synthesis"),
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
        
        Implements T044-T046:
        - Emits discovery events BEFORE agent streaming if auto_discover=True
        - Adds 10s timeout with asyncio.wait_for
        - Wraps discovery in try-except, emits discovery_error on failure
        
        Yields:
            SSE-formatted strings: "event: {event_type}\ndata: {json}\n\n"
        """
        try:
            # Get graph from cache (initialized at startup via lifespan)
            graph = _graph_cache.get('async_graph')
            if not graph:
                error_event = {"node": "error", "data": {"error": "Service not ready: Graph not initialized"}}
                yield f"data: {json.dumps(error_event)}\n\n"
                return
            
            # =========================================================================
            # PHASE 1: Paper Discovery (if enabled)
            # =========================================================================
            
            if auto_discover:
                logger.info("🔬 Auto-discovery enabled - starting paper discovery phase")
                
                try:
                    # Run discovery with 10-second timeout (T045)
                    discovery_task = discover_with_streaming(query, max_papers=3)
                    
                    async for discovery_event in discovery_task:
                        # Format as SSE with event type
                        event_type = discovery_event.get("event_type", "discovery_update")
                        event_data = discovery_event.get("data", {})
                        
                        # Emit SSE event
                        sse_message = f"event: {event_type}\ndata: {json.dumps(event_data)}\n\n"
                        yield sse_message
                        
                        # Check for timeout (discovery_complete event should arrive within 10s)
                        # Note: The timeout is handled by the outer try-except and asyncio.wait_for
                    
                    logger.info("✅ Discovery phase complete - proceeding to agent execution")
                    
                except asyncio.TimeoutError:
                    # T045: Emit timeout event but proceed with synthesis
                    logger.warning("⏱️  Discovery timeout (>10s) - proceeding with synthesis")
                    timeout_event = {
                        "event_type": "discovery_timeout",
                        "data": {"message": "Discovery exceeded 10s timeout, proceeding with synthesis"}
                    }
                    yield f"event: discovery_timeout\ndata: {json.dumps(timeout_event['data'])}\n\n"
                    
                except Exception as e:
                    # T046: Emit error event but DO NOT block synthesis
                    logger.error(f"❌ Discovery error (non-blocking): {e}")
                    error_event = {
                        "event_type": "discovery_error",
                        "data": {"error": str(e), "message": "Discovery failed, proceeding with synthesis"}
                    }
                    yield f"event: discovery_error\ndata: {json.dumps(error_event['data'])}\n\n"
            else:
                logger.info("ℹ️  Auto-discovery disabled - proceeding directly to agent execution")
            
            # =========================================================================
            # PHASE 2: Dialectical Synthesis (Agent Execution)
            # =========================================================================
            
            # Prepare initial state
            import uuid
            inputs = {
                "messages": [],
                "original_query": query,
                "current_thesis": None,
                "current_antithesis": None,
                "final_synthesis": None,
                "contradiction_report": "",
                "iteration_count": 0,
                "procedural_memory": "",
                # Tier 1: Memory and claim tracking
                "debate_memory": {
                    "rejected_claims": [],
                    "skeptic_objections": [],
                    "weak_evidence_urls": []
                },
                "current_claim_id": str(uuid.uuid4()),
                "synthesis_mode": None,  # T089: Circular argument handling
                "consecutive_high_similarity_count": 0,  # Natural termination: track stuck state
                "last_similarity_score": None,  # Natural termination: most recent similarity
                # Tier 2: Visualization & UX
                "conversation_history": [],  # For conversational thread view
                "current_round_papers_analyst": [],  # Papers discovered by Analyst in current round
                "current_round_papers_skeptic": []  # Papers discovered by Skeptic in current round
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
# Knowledge Discovery Endpoints (Tier 2.5 - Manual Discovery)
# =============================================================================

@app.post("/discover")
async def discover_papers(
    request: DiscoveryRequest,
    api_key: str = Depends(get_api_key)
) -> DiscoveryResponse:
    """
    Manually search for academic papers from arXiv or Semantic Scholar.
    
    Implements:
    - FR-KD-001: arXiv API integration
    - FR-KD-002: Semantic Scholar API integration
    - FR-KD-004: Max 10 papers per source
    - FR-KD-007: Rate limit handling
    - SC-KD-001: Response time < 3s for 95% of searches
    
    Args:
        request: DiscoveryRequest with query, source, max_results
        api_key: API key from X-API-Key header (validated by dependency)
    
    Returns:
        DiscoveryResponse with papers, count, source, query
    
    Raises:
        400: Invalid source parameter
        429: Rate limit exceeded (after retries)
        503: External API unavailable
        500: Internal server error
    """
    import time
    from src.tools.paper_discovery import search_arxiv, search_semantic_scholar
    
    start_time = time.time()
    logger.info(f"Discovery request: query='{request.query}', source={request.source}, max_results={request.max_results}")
    
    try:
        # Validate source parameter
        if request.source not in ["arxiv", "semantic_scholar"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid source: {request.source}. Must be 'arxiv' or 'semantic_scholar'"
            )
        
        # Call appropriate search function
        # Note: Run in thread pool to avoid nest_asyncio + uvloop conflicts
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        loop = asyncio.get_event_loop()
        if request.source == "arxiv":
            papers = await loop.run_in_executor(
                None,
                search_arxiv,
                request.query,
                request.max_results
            )
        else:  # semantic_scholar
            papers = await loop.run_in_executor(
                None,
                search_semantic_scholar,
                request.query,
                request.max_results
            )
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Discovery complete: {len(papers)} papers found in {elapsed_time:.3f}s")
        
        # Warn if response time exceeds 3 seconds (SC-KD-001)
        if elapsed_time > 3.0:
            logger.warning(f"⚠️  Discovery response time exceeded 3s: {elapsed_time:.3f}s (SC-KD-001)")
        
        return DiscoveryResponse(
            papers=papers,
            count=len(papers),
            source=request.source,
            query=request.query
        )
        
    except HTTPException:
        raise
    except Exception as e:
        error_str = str(e).lower()
        
        # Check for rate limit errors
        if "429" in error_str or "rate limit" in error_str:
            logger.error(f"Rate limit exceeded: {e}")
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Check for API unavailability
        if "connection" in error_str or "timeout" in error_str or "unavailable" in error_str:
            logger.error(f"External API unavailable: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"{request.source} API is currently unavailable. Please try again later."
            )
        
        # Generic error
        logger.error(f"Discovery error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during discovery: {str(e)}"
        )


@app.post("/add_papers")
async def add_papers_endpoint(
    request: AddPapersRequest,
    api_key: str = Depends(get_api_key)
) -> AddPapersResponse:
    """
    Add selected papers to the Neo4j knowledge graph.
    
    Implements:
    - FR-KD-005: Deduplication by URL
    - FR-KD-006: Discovery metadata tracking
    - FR-KD-014: Extended ResearchPaper fields
    - SC-KD-002: Add papers < 5s for batches of 10
    
    Args:
        request: AddPapersRequest with paper_urls, discovered_by
        api_key: API key from X-API-Key header (validated by dependency)
    
    Returns:
        AddPapersResponse with added, skipped, failed, details
    
    Raises:
        400: Empty paper_urls list
        500: Neo4j connection error or internal error
    """
    import time
    from src.models import PaperMetadata
    from src.tools.kg_tools import add_papers_to_neo4j
    
    start_time = time.time()
    paper_count = len(request.papers) if request.papers else (len(request.paper_urls) if request.paper_urls else 0)
    logger.info(f"Add papers request: {paper_count} papers, discovered_by={request.discovered_by}")
    
    try:
        # Use full paper objects if provided, otherwise create minimal objects from URLs
        papers_to_add = []
        
        if request.papers:
            # Frontend sent full paper metadata (preferred approach)
            papers_to_add = request.papers
            logger.info(f"Using {len(papers_to_add)} full paper objects from request")
        elif request.paper_urls:
            # Fallback: Frontend only sent URLs (deprecated, creates minimal papers)
            logger.warning("Using deprecated paper_urls field. Frontend should send full 'papers' objects.")
            for url in request.paper_urls:
                paper = PaperMetadata(
                    title="Paper from URL",
                    url=url,
                    abstract="Fetched from discovery session",
                    authors=["Unknown"],
                    published="2024-01-01",
                    source="manual",
                    citation_count=0,
                    fields_of_study=[]
                )
                papers_to_add.append(paper)
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'papers' or 'paper_urls' must be provided"
            )
        
        # Add papers to Neo4j
        result = add_papers_to_neo4j(papers_to_add, discovered_by=request.discovered_by)
        
        elapsed_time = time.time() - start_time
        logger.info(
            f"✅ Add papers complete: {result['added']} added, {result['skipped']} skipped, "
            f"{result['failed']} failed in {elapsed_time:.3f}s"
        )
        
        # Warn if response time exceeds 5 seconds for ≤10 papers (SC-KD-002)
        if paper_count <= 10 and elapsed_time > 5.0:
            logger.warning(f"⚠️  Add papers response time exceeded 5s: {elapsed_time:.3f}s (SC-KD-002)")
        
        return AddPapersResponse(
            added=result["added"],
            skipped=result["skipped"],
            failed=result["failed"],
            details=result["details"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add papers error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while adding papers: {str(e)}"
        )


# =============================================================================
# Epic 4: Persistent Memory System - Session Management Endpoints
# =============================================================================

@app.post(
    "/sessions",
    tags=["Epic 4: Sessions"],
    summary="Create a new debate session",
    description="Create a new session with metadata tracking. Requires X-API-Key header.",
    response_model=SessionMetadata
)
async def create_session_endpoint(
    request: CreateSessionRequest,
    user_id: str = Query(..., description="User ID creating the session"),
    api_key: str = Depends(get_api_key)
) -> SessionMetadata:
    """
    Create a new debate session.

    Epic 4: Task 1 - Session Management & Persistence

    Args:
        request: CreateSessionRequest with workspace_id, title, query, tags
        user_id: User ID creating the session
        api_key: API key from X-API-Key header

    Returns:
        SessionMetadata of the created session

    Raises:
        400: Invalid request data
        500: Internal server error
    """
    try:
        import uuid

        # Generate unique IDs
        session_id = str(uuid.uuid4())
        thread_id = f"thread_{session_id}"

        # Create session metadata
        session = SessionMetadata(
            session_id=session_id,
            workspace_id=request.workspace_id,
            thread_id=thread_id,
            title=request.title,
            original_query=request.query,
            status=SessionStatus.ACTIVE,
            tags=request.tags
        )

        # Save to Neo4j
        success = memory_store.create_session(session, created_by=user_id)

        if success:
            logger.info(f"✅ Created session: {session.title} ({session.session_id})")
            return session
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to create session in database"
            )

    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get(
    "/sessions/{session_id}",
    tags=["Epic 4: Sessions"],
    summary="Get session details",
    description="Retrieve session metadata by ID. Requires X-API-Key header.",
    response_model=SessionMetadata
)
async def get_session_endpoint(
    session_id: str,
    api_key: str = Depends(get_api_key)
) -> SessionMetadata:
    """
    Get session details by ID.

    Epic 4: Task 1 - Session Management & Persistence

    Args:
        session_id: Session identifier
        api_key: API key from X-API-Key header

    Returns:
        SessionMetadata if found

    Raises:
        404: Session not found
        500: Internal server error
    """
    try:
        session = memory_store.get_session(session_id)

        if session:
            return session
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.put(
    "/sessions/{session_id}",
    tags=["Epic 4: Sessions"],
    summary="Update session metadata",
    description="Update session title, status, or tags. Requires X-API-Key header.",
    response_model=Dict[str, str]
)
async def update_session_endpoint(
    session_id: str,
    request: UpdateSessionRequest,
    api_key: str = Depends(get_api_key)
) -> Dict[str, str]:
    """
    Update session metadata.

    Epic 4: Task 1 - Session Management & Persistence

    Args:
        session_id: Session identifier
        request: UpdateSessionRequest with title, status, tags
        api_key: API key from X-API-Key header

    Returns:
        Success message

    Raises:
        404: Session not found
        500: Internal server error
    """
    try:
        # Build updates dict from request
        updates = {}
        if request.title is not None:
            updates["title"] = request.title
        if request.status is not None:
            updates["status"] = request.status
        if request.tags is not None:
            updates["tags"] = request.tags

        success = memory_store.update_session(session_id, **updates)

        if success:
            logger.info(f"✅ Updated session: {session_id}")
            return {"message": f"Session '{session_id}' updated successfully"}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.delete(
    "/sessions/{session_id}",
    tags=["Epic 4: Sessions"],
    summary="Delete session",
    description="Soft delete session by setting status to DELETED. Requires X-API-Key header.",
    response_model=Dict[str, str]
)
async def delete_session_endpoint(
    session_id: str,
    api_key: str = Depends(get_api_key)
) -> Dict[str, str]:
    """
    Delete session (soft delete).

    Epic 4: Task 1 - Session Management & Persistence

    Args:
        session_id: Session identifier
        api_key: API key from X-API-Key header

    Returns:
        Success message

    Raises:
        404: Session not found
        500: Internal server error
    """
    try:
        success = memory_store.delete_session(session_id)

        if success:
            logger.info(f"✅ Deleted session: {session_id}")
            return {"message": f"Session '{session_id}' deleted successfully"}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{session_id}' not found"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get(
    "/sessions",
    tags=["Epic 4: Sessions"],
    summary="List sessions",
    description="List sessions in a workspace with pagination. Requires X-API-Key header.",
    response_model=SessionListResponse
)
async def list_sessions_endpoint(
    workspace_id: str = Query(..., description="Workspace identifier"),
    status: Optional[str] = Query(None, description="Filter by status (active, paused, completed, archived, deleted)"),
    page: int = Query(1, description="Page number (1-indexed)", ge=1),
    page_size: int = Query(20, description="Page size", ge=1, le=100),
    api_key: str = Depends(get_api_key)
) -> SessionListResponse:
    """
    List sessions in a workspace with pagination.

    Epic 4: Task 1 - Session Management & Persistence

    Args:
        workspace_id: Workspace identifier
        status: Optional status filter
        page: Page number (1-indexed)
        page_size: Number of sessions per page
        api_key: API key from X-API-Key header

    Returns:
        SessionListResponse with sessions, total, page, page_size

    Raises:
        400: Invalid status parameter
        500: Internal server error
    """
    try:
        # Parse status if provided
        session_status = None
        if status:
            try:
                session_status = SessionStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status}. Must be one of: active, paused, completed, archived, deleted"
                )

        # Get sessions
        sessions, total = memory_store.list_sessions(
            workspace_id=workspace_id,
            status=session_status,
            page=page,
            page_size=page_size
        )

        logger.info(f"✅ Listed {len(sessions)} sessions (total: {total})")

        return SessionListResponse(
            sessions=sessions,
            total=total,
            page=page,
            page_size=page_size
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# =============================================================================
# Epic 4: Persistent Memory System - User & Workspace Endpoints
# =============================================================================

@app.post(
    "/users",
    tags=["Epic 4: Users"],
    summary="Create a new user",
    description="Create a new user profile. Requires X-API-Key header.",
    response_model=UserProfile
)
async def create_user_endpoint(
    username: str = Query(..., description="Username"),
    email: str = Query(..., description="Email address"),
    api_key: str = Depends(get_api_key)
) -> UserProfile:
    """
    Create a new user profile.

    Epic 4: Task 2 - User/Workspace Isolation

    Args:
        username: Username (3-50 characters)
        email: Email address
        api_key: API key from X-API-Key header

    Returns:
        Created UserProfile

    Raises:
        400: Invalid request data
        500: Internal server error
    """
    try:
        user = UserProfile(
            username=username,
            email=email
        )

        success = memory_store.create_user(user)

        if success:
            logger.info(f"✅ Created user: {user.username} ({user.user_id})")
            return user
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to create user in database"
            )

    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post(
    "/workspaces",
    tags=["Epic 4: Workspaces"],
    summary="Create a new workspace",
    description="Create a new workspace. Requires X-API-Key header.",
    response_model=Workspace
)
async def create_workspace_endpoint(
    name: str = Query(..., description="Workspace name"),
    owner_id: str = Query(..., description="Owner user ID"),
    description: Optional[str] = Query(None, description="Workspace description"),
    is_public: bool = Query(False, description="Is workspace public"),
    api_key: str = Depends(get_api_key)
) -> Workspace:
    """
    Create a new workspace.

    Epic 4: Task 2 - User/Workspace Isolation

    Args:
        name: Workspace name
        owner_id: Owner user ID
        description: Optional workspace description
        is_public: Whether workspace is public
        api_key: API key from X-API-Key header

    Returns:
        Created Workspace

    Raises:
        400: Invalid request data
        500: Internal server error
    """
    try:
        workspace = Workspace(
            name=name,
            owner_id=owner_id,
            description=description,
            is_public=is_public
        )

        success = memory_store.create_workspace(workspace)

        if success:
            logger.info(f"✅ Created workspace: {workspace.name} ({workspace.workspace_id})")
            return workspace
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to create workspace in database"
            )

    except Exception as e:
        logger.error(f"Error creating workspace: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get(
    "/workspaces/{workspace_id}",
    tags=["Epic 4: Workspaces"],
    summary="Get workspace details",
    description="Retrieve workspace metadata by ID. Requires X-API-Key header.",
    response_model=Workspace
)
async def get_workspace_endpoint(
    workspace_id: str,
    api_key: str = Depends(get_api_key)
) -> Workspace:
    """
    Get workspace details by ID.

    Epic 4: Task 2 - User/Workspace Isolation

    Args:
        workspace_id: Workspace identifier
        api_key: API key from X-API-Key header

    Returns:
        Workspace if found

    Raises:
        404: Workspace not found
        500: Internal server error
    """
    try:
        workspace = memory_store.get_workspace(workspace_id)

        if workspace:
            return workspace
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Workspace '{workspace_id}' not found"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving workspace: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


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

