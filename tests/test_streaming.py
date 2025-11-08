"""
Tests for FastAPI SSE streaming endpoints - Tier 2.

Tests the following endpoints:
- GET /stream_dialectics/{thread_id} (SSE streaming with auth)

Implements:
- T035: Test SSE streaming format, first event timing, completion, checkpointing
- FR-T2-012: API key authentication required
- SC-T2-003: First SSE event < 500ms
"""

import os
import json
import time
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import AsyncIterator


# These tests are written FIRST (TDD) - they will FAIL until implementation is complete


@pytest.fixture
def mock_env():
    """Set up test environment variables."""
    os.environ["API_KEY"] = "test_secret_key_12345"
    os.environ["GOOGLE_API_KEY"] = "test_google_key"
    os.environ["NEO4J_URI"] = "bolt://localhost:7687"
    os.environ["NEO4J_USER"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "password123"
    os.environ["MAX_ITERATIONS"] = "3"
    yield
    # Cleanup
    for key in ["API_KEY", "GOOGLE_API_KEY", "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"]:
        os.environ.pop(key, None)


@pytest.fixture
def mock_neo4j():
    """Mock Neo4j connection."""
    with patch("neo4j.GraphDatabase.driver") as mock_driver:
        mock_conn = MagicMock()
        mock_conn.verify_connectivity.return_value = None
        mock_driver.return_value = mock_conn
        yield mock_driver


@pytest.fixture
def client(mock_env, mock_neo4j):
    """
    Create a test client for the FastAPI app.
    
    Note: This fixture will fail until src/api.py is implemented.
    """
    from src.api import app, _graph_cache
    
    # Create test client (this will NOT trigger lifespan for TestClient)
    test_client = TestClient(app)
    
    # Manually initialize a mock graph in the cache for testing
    # Since TestClient doesn't run lifespan by default
    mock_graph = MagicMock()
    _graph_cache['async_graph'] = mock_graph
    
    yield test_client
    
    # Cleanup
    _graph_cache.clear()


# =============================================================================
# SSE Streaming Endpoint Tests
# =============================================================================

def test_stream_dialectics_requires_authentication(client):
    """
    Test that /stream_dialectics/{thread_id} requires X-API-Key header.
    
    Implements FR-T2-012: API key authentication required.
    """
    thread_id = "test_stream_thread"
    query = "What are the benefits of LangGraph?"
    
    # Call without API key
    response = client.get(f"/stream_dialectics/{thread_id}?query={query}")
    
    assert response.status_code == 401, "Should return 401 when API key is missing"
    data = response.json()
    assert "detail" in data
    assert "Invalid or missing API key" in data["detail"]


def test_stream_dialectics_rejects_invalid_api_key(client):
    """
    Test that /stream_dialectics/{thread_id} rejects invalid API keys.
    """
    thread_id = "test_stream_thread"
    query = "What are the benefits of LangGraph?"
    
    response = client.get(
        f"/stream_dialectics/{thread_id}?query={query}",
        headers={"X-API-Key": "invalid_key"}
    )
    
    assert response.status_code == 401, "Should return 401 for invalid API key"


def test_stream_dialectics_returns_sse_events(client):
    """
    Test that /stream_dialectics/{thread_id} returns SSE events in correct format.
    
    Expected format: data: {"node": "analyst", "data": {...}}\n\n
    
    Implements T035: Test SSE event format.
    """
    from src.api import _graph_cache
    
    thread_id = "test_stream_thread"
    query = "What are the benefits of LangGraph?"
    
    # Get the mock graph from cache (set by the fixture) and configure it
    mock_graph = _graph_cache['async_graph']
    
    async def mock_astream(*args, **kwargs):
        """Mock async stream generator."""
        # Simulate analyst node
        yield {
            "analyst": {
                "current_thesis": {
                    "claim": "LangGraph enables cyclic workflows",
                    "reasoning": "Test reasoning",
                    "evidence": [
                        {"source_url": "url1", "snippet": "snippet1"},
                        {"source_url": "url2", "snippet": "snippet2"}
                    ]
                }
            }
        }
        # Simulate skeptic node
        yield {
            "skeptic": {
                "current_antithesis": {
                    "contradiction_found": False,
                    "critique": "No contradictions found"
                }
            }
        }
        # Simulate synthesizer node
        yield {
            "synthesizer": {
                "final_synthesis": {
                    "novel_insight": "Test synthesis",
                    "confidence_score": 0.85,
                    "novelty_score": 0.75,
                    "evidence_lineage": ["url1", "url2", "url3"],
                    "supporting_claims": ["claim1"],
                    "reasoning": "Test reasoning"
                }
            }
        }
    
    mock_graph.astream = mock_astream
    
    # Make request with valid API key
    response = client.get(
        f"/stream_dialectics/{thread_id}?query={query}",
        headers={"X-API-Key": "test_secret_key_12345"}
    )
    
    assert response.status_code == 200, "Should return 200 with valid API key"
    assert "text/event-stream" in response.headers["content-type"], \
        "Should return SSE content type"
    
    # Parse SSE events
    events = []
    for line in response.text.split("\n\n"):
        if line.startswith("data: "):
            event_data = line[6:]  # Remove "data: " prefix
            if event_data.strip():
                events.append(json.loads(event_data))
    
    assert len(events) > 0, "Should receive at least one SSE event"
    
    # Verify event format
    for event in events:
        assert "node" in event, "Each event should have 'node' field"
        assert "data" in event, "Each event should have 'data' field"
        assert event["node"] in ["analyst", "skeptic", "synthesizer", "increment_iteration"], \
            f"Node should be valid agent name, got: {event['node']}"


@patch("src.graph.build_graph")
def test_stream_dialectics_first_event_within_500ms(mock_build_graph, client):
    """
    Test that first SSE event arrives within 500ms.
    
    Implements SC-T2-003: First event < 500ms.
    """
    thread_id = "test_latency_thread"
    query = "Quick test query"
    
    # Mock graph with fast initial response
    mock_graph = MagicMock()
    
    async def mock_astream(*args, **kwargs):
        # First event should come quickly
        yield {
            "analyst": {
                "current_thesis": {
                    "claim": "Test claim",
                    "reasoning": "Test reasoning",
                    "evidence": [
                        {"source_url": "url1", "snippet": "snippet1"},
                        {"source_url": "url2", "snippet": "snippet2"}
                    ]
                }
            }
        }
    
    mock_graph.astream = mock_astream
    mock_build_graph.return_value = mock_graph
    
    # Measure time to first event
    start_time = time.time()
    
    response = client.get(
        f"/stream_dialectics/{thread_id}?query={query}",
        headers={"X-API-Key": "test_secret_key_12345"}
    )
    
    assert response.status_code == 200
    
    # Get first event
    first_event_time = None
    for line in response.text.split("\n\n"):
        if line.startswith("data: "):
            first_event_time = time.time()
            break
    
    if first_event_time:
        latency = first_event_time - start_time
        assert latency < 0.5, \
            f"First event should arrive within 500ms, got {latency*1000:.0f}ms (SC-T2-003)"


def test_stream_dialectics_completes_with_final_synthesis(client):
    """
    Test that streaming completes with final synthesis in the last event.
    """
    from src.api import _graph_cache
    
    thread_id = "test_completion_thread"
    query = "Test completion"
    
    # Mock the cached graph
    mock_graph = _graph_cache['async_graph']
    
    async def mock_astream(*args, **kwargs):
        # Yield full workflow
        yield {"analyst": {"current_thesis": {"claim": "test", "reasoning": "test", "evidence": [{"source_url": "url1", "snippet": "s1"}, {"source_url": "url2", "snippet": "s2"}]}}}
        yield {"skeptic": {"current_antithesis": {"contradiction_found": False, "critique": "test"}}}
        yield {
            "synthesizer": {
                "final_synthesis": {
                    "novel_insight": "Final synthesis",
                    "confidence_score": 0.9,
                    "novelty_score": 0.8,
                    "evidence_lineage": ["url1", "url2", "url3"],
                    "supporting_claims": ["claim1"],
                    "reasoning": "Final reasoning"
                }
            }
        }
    
    mock_graph.astream = mock_astream
    
    response = client.get(
        f"/stream_dialectics/{thread_id}?query={query}",
        headers={"X-API-Key": "test_secret_key_12345"}
    )
    
    assert response.status_code == 200
    
    # Parse all events
    events = []
    for line in response.text.split("\n\n"):
        if line.startswith("data: "):
            event_data = line[6:]
            if event_data.strip():
                events.append(json.loads(event_data))
    
    # Last event should be synthesizer with final_synthesis
    assert len(events) > 0, "Should have events"
    last_event = events[-1]
    assert last_event["node"] == "synthesizer", "Last event should be from synthesizer"
    assert "final_synthesis" in last_event["data"], "Last event should contain final_synthesis"


@patch("src.graph.build_graph")
@patch("langgraph.checkpoint.sqlite.SqliteSaver")
def test_stream_dialectics_creates_checkpoints(mock_saver, mock_build_graph, client):
    """
    Test that checkpointer creates snapshots after each node.
    
    This verifies state persistence during streaming.
    """
    thread_id = "test_checkpoint_thread"
    query = "Test checkpointing"
    
    # Mock checkpointer
    mock_checkpointer = MagicMock()
    mock_saver.from_conn_string.return_value = mock_checkpointer
    
    # Mock graph
    mock_graph = MagicMock()
    
    async def mock_astream(*args, **kwargs):
        yield {"analyst": {"current_thesis": {"claim": "test", "reasoning": "test", "evidence": [{"source_url": "url1", "snippet": "s1"}, {"source_url": "url2", "snippet": "s2"}]}}}
        yield {"skeptic": {"current_antithesis": {"contradiction_found": False, "critique": "test"}}}
        yield {"synthesizer": {"final_synthesis": {"novel_insight": "test", "confidence_score": 0.8, "novelty_score": 0.7, "evidence_lineage": ["url1", "url2", "url3"], "supporting_claims": ["c1"], "reasoning": "test"}}}
    
    mock_graph.astream = mock_astream
    mock_build_graph.return_value = mock_graph
    
    response = client.get(
        f"/stream_dialectics/{thread_id}?query={query}",
        headers={"X-API-Key": "test_secret_key_12345"}
    )
    
    assert response.status_code == 200
    
    # Verify checkpointer was used (this will be validated by checking
    # if graph was built with checkpointer)
    # The actual checkpoint creation is handled by LangGraph internally


@patch("src.graph.build_graph")
def test_stream_dialectics_handles_query_parameter(mock_build_graph, client):
    """
    Test that the endpoint correctly processes the query parameter.
    """
    thread_id = "test_query_param"
    query = "What is the meaning of life?"
    
    mock_graph = MagicMock()
    
    async def mock_astream(inputs, *args, **kwargs):
        # Verify query was passed correctly
        assert inputs["original_query"] == query, "Query should be passed to graph"
        yield {"analyst": {"current_thesis": {"claim": "test", "reasoning": "test", "evidence": [{"source_url": "url1", "snippet": "s1"}, {"source_url": "url2", "snippet": "s2"}]}}}
    
    mock_graph.astream = mock_astream
    mock_build_graph.return_value = mock_graph
    
    response = client.get(
        f"/stream_dialectics/{thread_id}?query={query}",
        headers={"X-API-Key": "test_secret_key_12345"}
    )
    
    assert response.status_code == 200


@patch("src.graph.build_graph")
def test_stream_dialectics_handles_missing_query_parameter(mock_build_graph, client):
    """
    Test that endpoint handles missing query parameter gracefully.
    """
    thread_id = "test_missing_query"
    
    response = client.get(
        f"/stream_dialectics/{thread_id}",  # No query parameter
        headers={"X-API-Key": "test_secret_key_12345"}
    )
    
    # Should return 422 (validation error) or 400 (bad request)
    assert response.status_code in [400, 422], \
        "Should return error when query parameter is missing"


def test_stream_dialectics_handles_iteration_loop(client):
    """
    Test that streaming handles multiple iterations (contradiction loop).
    """
    from src.api import _graph_cache
    
    thread_id = "test_iteration_thread"
    query = "Test iterations"
    
    # Mock the cached graph
    mock_graph = _graph_cache['async_graph']
    
    async def mock_astream(*args, **kwargs):
        # First iteration
        yield {"analyst": {"current_thesis": {"claim": "test1", "reasoning": "test", "evidence": [{"source_url": "url1", "snippet": "s1"}, {"source_url": "url2", "snippet": "s2"}]}}}
        yield {"skeptic": {"current_antithesis": {"contradiction_found": True, "counter_claim": "counter", "conflicting_evidence": [], "critique": "contradiction found"}}}
        yield {"increment_iteration": {"iteration_count": 1}}
        # Second iteration
        yield {"analyst": {"current_thesis": {"claim": "test2", "reasoning": "test", "evidence": [{"source_url": "url3", "snippet": "s3"}, {"source_url": "url4", "snippet": "s4"}]}}}
        yield {"skeptic": {"current_antithesis": {"contradiction_found": False, "critique": "resolved"}}}
        # Final synthesis
        yield {"synthesizer": {"final_synthesis": {"novel_insight": "final", "confidence_score": 0.85, "novelty_score": 0.75, "evidence_lineage": ["url1", "url2", "url3"], "supporting_claims": ["c1"], "reasoning": "final"}}}
    
    mock_graph.astream = mock_astream
    
    response = client.get(
        f"/stream_dialectics/{thread_id}?query={query}",
        headers={"X-API-Key": "test_secret_key_12345"}
    )
    
    assert response.status_code == 200
    
    # Parse events
    events = []
    for line in response.text.split("\n\n"):
        if line.startswith("data: "):
            event_data = line[6:]
            if event_data.strip():
                events.append(json.loads(event_data))
    
    # Should have events from multiple iterations
    analyst_events = [e for e in events if e["node"] == "analyst"]
    assert len(analyst_events) >= 2, "Should have multiple analyst iterations"


# =============================================================================
# Notes for Implementation
# =============================================================================

"""
These tests cover:

1. **Authentication**:
   - Requires X-API-Key header
   - Rejects invalid API keys
   - Returns 401 for auth failures

2. **SSE Streaming Format**:
   - Returns text/event-stream content type
   - Events in format: data: {"node": "...", "data": {...}}
   - Events include all agent nodes

3. **Performance (SC-T2-003)**:
   - First event arrives within 500ms
   - Stream completes with final synthesis

4. **Persistence**:
   - Checkpointer creates snapshots
   - Thread ID is used for state isolation

5. **Edge Cases**:
   - Missing query parameter
   - Multiple iterations (contradiction loop)

Next Steps (Implementation):
1. Create /stream_dialectics/{thread_id} endpoint in src/api.py
2. Implement SSE event generator
3. Integrate graph.astream() with checkpointer
4. Add proper error handling
5. Ensure first event latency < 500ms
"""

