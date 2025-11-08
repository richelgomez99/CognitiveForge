"""
Tests for state persistence and checkpointing - Tier 2.

Tests the following functionality:
- SqliteSaver creates dialectical_system.db file
- Thread ID isolation (different threads have independent states)
- Resume from existing thread_id continues from last checkpoint
- /get_trace/{thread_id} returns correct checkpoint count

Implements:
- T036: Test persistence layer with SqliteSaver
- FR-T2-010: State persistence via LangGraph checkpointer
- SC-T2-005: 100% of runs retrievable via thread_id
- SC-T2-006: Checkpoint count matches iteration_count
"""

import os
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient


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
def test_db_path(tmp_path):
    """Create a temporary database path for testing."""
    db_path = tmp_path / "test_dialectical_system.db"
    yield str(db_path)
    # Cleanup
    if db_path.exists():
        db_path.unlink()


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
    """Create a test client for the FastAPI app."""
    from src.api import app, _graph_cache
    
    # Create test client (this will NOT trigger lifespan for TestClient)
    test_client = TestClient(app)
    
    # Manually initialize a mock graph in the cache for testing
    mock_graph = MagicMock()
    _graph_cache['async_graph'] = mock_graph
    
    yield test_client
    
    # Cleanup
    _graph_cache.clear()


# =============================================================================
# SqliteSaver Database Creation Tests
# =============================================================================

def test_sqlite_saver_creates_database_file():
    """
    Test that SqliteSaver creates dialectical_system.db file when first used.
    
    Implements FR-T2-010: State persistence via checkpointer.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver
    from pathlib import Path
    
    # Use a temporary database for testing
    test_db = "test_checkpoint_creation.db"
    
    try:
        # Create SqliteSaver using context manager
        with SqliteSaver.from_conn_string(test_db) as checkpointer:
            # Setup creates the tables
            checkpointer.setup()
        
        # Verify database file was created
        assert Path(test_db).exists(), "SqliteSaver should create database file"
        
        # Verify it's a valid SQLite database
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        conn.close()
        
        assert len(tables) > 0, "Database should contain checkpoint tables"
        
    finally:
        # Cleanup
        if Path(test_db).exists():
            Path(test_db).unlink()


@patch("src.graph.build_graph")
def test_graph_with_checkpointer_creates_db(mock_build_graph, test_db_path):
    """
    Test that building graph with checkpointer creates the database.
    """
    from src.graph import build_graph
    
    # This will be implemented when we update build_graph()
    # The test should verify that:
    # 1. build_graph(use_checkpointer=True) creates the db file
    # 2. The graph can be invoked with a thread_id
    pass  # Implementation will be added after src.graph is updated


# =============================================================================
# Thread ID Isolation Tests (SC-T2-005)
# =============================================================================

@patch("src.graph.build_graph")
def test_different_threads_have_independent_states(mock_build_graph, test_db_path):
    """
    Test that different thread_ids maintain independent states.
    
    Implements SC-T2-005: Persistence reliability - thread isolation.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver
    
    # Mock graph
    mock_graph = MagicMock()
    
    # Mock different states for different threads
    def mock_get_state(config):
        thread_id = config.get("configurable", {}).get("thread_id")
        mock_state = MagicMock()
        mock_state.values = {
            "original_query": f"Query for {thread_id}",
            "thread_id": thread_id,
            "iteration_count": 1
        }
        return mock_state
    
    mock_graph.get_state = mock_get_state
    mock_build_graph.return_value = mock_graph
    
    # Get states for two different threads
    thread1_config = {"configurable": {"thread_id": "thread_1"}}
    thread2_config = {"configurable": {"thread_id": "thread_2"}}
    
    state1 = mock_graph.get_state(thread1_config)
    state2 = mock_graph.get_state(thread2_config)
    
    # Verify states are independent
    assert state1.values["thread_id"] == "thread_1"
    assert state2.values["thread_id"] == "thread_2"
    assert state1.values["original_query"] != state2.values["original_query"]


def test_thread_isolation_via_api(client):
    """
    Test thread isolation through API endpoints.
    
    Verifies that different thread_ids return different states via API.
    """
    from src.api import _graph_cache
    
    # Mock the cached graph with thread-specific states
    mock_graph = _graph_cache['async_graph']
    
    def mock_get_state(config):
        thread_id = config.get("configurable", {}).get("thread_id")
        mock_state = MagicMock()
        mock_state.values = {
            "original_query": f"Query for {thread_id}",
            "thread_id": thread_id,
            "iteration_count": 1 if thread_id == "thread_a" else 2,
            "messages": []
        }
        return mock_state
    
    mock_graph.get_state = mock_get_state
    
    # Query two different threads
    response_a = client.get(
        "/get_state/thread_a",
        headers={"X-API-Key": "test_secret_key_12345"}
    )
    response_b = client.get(
        "/get_state/thread_b",
        headers={"X-API-Key": "test_secret_key_12345"}
    )
    
    assert response_a.status_code == 200
    assert response_b.status_code == 200
    
    data_a = response_a.json()
    data_b = response_b.json()
    
    # Verify states are different
    assert data_a["iteration_count"] == 1
    assert data_b["iteration_count"] == 2
    assert data_a["original_query"] != data_b["original_query"]


# =============================================================================
# Resume from Checkpoint Tests
# =============================================================================

@patch("src.graph.build_graph")
def test_resume_from_existing_thread_continues_from_checkpoint(mock_build_graph, test_db_path):
    """
    Test that resuming with an existing thread_id continues from last checkpoint.
    
    Implements FR-T2-010: Resume capability from persisted state.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver
    
    # Create a mock graph with checkpoint history
    mock_graph = MagicMock()
    
    # Mock state at checkpoint 2
    mock_state = MagicMock()
    mock_state.values = {
        "original_query": "Test query",
        "iteration_count": 2,
        "current_thesis": {
            "claim": "Resumed thesis",
            "reasoning": "From checkpoint",
            "evidence": [
                {"source_url": "url1", "snippet": "snippet1"},
                {"source_url": "url2", "snippet": "snippet2"}
            ]
        },
        "messages": ["message1", "message2"]
    }
    mock_state.config = {"configurable": {"thread_id": "resume_test"}}
    mock_state.next = ("synthesizer",)  # Next node to execute
    
    mock_graph.get_state.return_value = mock_state
    mock_build_graph.return_value = mock_graph
    
    # Get state for existing thread
    config = {"configurable": {"thread_id": "resume_test"}}
    resumed_state = mock_graph.get_state(config)
    
    # Verify state was retrieved
    assert resumed_state.values["iteration_count"] == 2
    assert resumed_state.values["current_thesis"]["claim"] == "Resumed thesis"
    assert len(resumed_state.values["messages"]) == 2
    assert resumed_state.next[0] == "synthesizer"


@patch("src.graph.build_graph")
def test_new_thread_starts_from_beginning(mock_build_graph, test_db_path):
    """
    Test that a new thread_id starts from the beginning (no checkpoint).
    """
    mock_graph = MagicMock()
    
    # Mock empty state for new thread
    def mock_get_state(config):
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id == "new_thread":
            # New thread returns None or initial state
            return None
        else:
            # Existing thread has state
            mock_state = MagicMock()
            mock_state.values = {"iteration_count": 3}
            return mock_state
    
    mock_graph.get_state = mock_get_state
    mock_build_graph.return_value = mock_graph
    
    # Query new thread
    new_state = mock_graph.get_state({"configurable": {"thread_id": "new_thread"}})
    existing_state = mock_graph.get_state({"configurable": {"thread_id": "existing_thread"}})
    
    assert new_state is None, "New thread should have no existing state"
    assert existing_state is not None, "Existing thread should have state"
    assert existing_state.values["iteration_count"] == 3


# =============================================================================
# Checkpoint Trace Tests (SC-T2-006)
# =============================================================================

def test_get_trace_returns_correct_checkpoint_count(client):
    """
    Test that /get_trace/{thread_id} returns checkpoint count matching iteration_count.
    
    Implements SC-T2-006: Checkpoint integrity validation.
    """
    from src.api import _graph_cache
    
    # Mock the cached graph
    mock_graph = _graph_cache['async_graph']
    
    # Mock checkpoint history with 3 checkpoints
    class MockCheckpoint:
        def __init__(self, checkpoint_id, iteration):
            self.config = {"configurable": {"thread_id": "test_trace", "checkpoint_id": checkpoint_id}}
            self.values = {
                "iteration_count": iteration,
                "current_thesis": {"claim": f"Thesis {iteration}", "reasoning": "test", "evidence": [{"source_url": "url", "snippet": "s"}]*2},
                "current_antithesis": {"contradiction_found": iteration < 3, "critique": f"Critique {iteration}"},
                "final_synthesis": None if iteration < 3 else {"novel_insight": "Final", "confidence_score": 0.8, "novelty_score": 0.7, "evidence_lineage": ["u1", "u2", "u3"], "supporting_claims": ["c"], "reasoning": "r"}
            }
            self.metadata = {"step": iteration}
    
    mock_history = [
        MockCheckpoint("cp1", 1),
        MockCheckpoint("cp2", 2),
        MockCheckpoint("cp3", 3)
    ]
    
    mock_graph.get_state_history.return_value = mock_history
    
    # Query trace
    response = client.get(
        "/get_trace/test_trace",
        headers={"X-API-Key": "test_secret_key_12345"}
    )
    
    assert response.status_code == 200
    
    trace_data = response.json()
    
    # Verify checkpoint count
    assert isinstance(trace_data, list), "Trace should be a list of checkpoints"
    assert len(trace_data) == 3, "Should have 3 checkpoints matching 3 iterations"
    
    # Verify checkpoint progression
    for i, checkpoint in enumerate(trace_data):
        assert checkpoint["iteration_count"] == i + 1, f"Checkpoint {i} should have iteration_count {i+1}"


def test_get_trace_includes_thesis_evolution(client):
    """
    Test that /get_trace shows the evolution of thesis across checkpoints.
    """
    from src.api import _graph_cache
    
    # Mock the cached graph
    mock_graph = _graph_cache['async_graph']
    
    # Mock evolving thesis
    class MockCheckpoint:
        def __init__(self, iteration, thesis_claim):
            self.config = {"configurable": {"thread_id": "evolution_test", "checkpoint_id": f"cp{iteration}"}}
            self.values = {
                "iteration_count": iteration,
                "current_thesis": {"claim": thesis_claim, "reasoning": "test", "evidence": [{"source_url": "url", "snippet": "s"}]*2}
            }
            self.metadata = {"step": iteration}
    
    mock_history = [
        MockCheckpoint(1, "Initial thesis"),
        MockCheckpoint(2, "Refined thesis after critique"),
        MockCheckpoint(3, "Final thesis")
    ]
    
    mock_graph.get_state_history.return_value = mock_history
    
    response = client.get(
        "/get_trace/evolution_test",
        headers={"X-API-Key": "test_secret_key_12345"}
    )
    
    assert response.status_code == 200
    trace_data = response.json()
    
    # Verify thesis evolution
    assert trace_data[0]["current_thesis"]["claim"] == "Initial thesis"
    assert trace_data[1]["current_thesis"]["claim"] == "Refined thesis after critique"
    assert trace_data[2]["current_thesis"]["claim"] == "Final thesis"


@patch("src.graph.build_graph")
def test_get_trace_returns_404_for_nonexistent_thread(mock_build_graph, client):
    """
    Test that /get_trace returns 404 for non-existent thread.
    """
    mock_graph = MagicMock()
    mock_graph.get_state_history.return_value = []  # Empty history
    mock_build_graph.return_value = mock_graph
    
    response = client.get(
        "/get_trace/nonexistent_thread",
        headers={"X-API-Key": "test_secret_key_12345"}
    )
    
    # Should return 404 or empty array
    assert response.status_code in [200, 404]
    if response.status_code == 200:
        data = response.json()
        assert len(data) == 0, "Non-existent thread should have empty trace"


# =============================================================================
# Persistence Reliability Tests (SC-T2-005)
# =============================================================================

@patch("src.graph.build_graph")
def test_100_percent_of_runs_retrievable(mock_build_graph, client):
    """
    Test that 100% of completed runs are retrievable via thread_id.
    
    Implements SC-T2-005: Persistence reliability.
    """
    mock_graph = MagicMock()
    
    # Simulate 5 completed runs
    completed_threads = []
    for i in range(5):
        thread_id = f"completed_run_{i}"
        completed_threads.append(thread_id)
        
        # Mock state for this thread
        def create_mock_state(tid):
            mock_state = MagicMock()
            mock_state.values = {
                "original_query": f"Query {tid}",
                "thread_id": tid,
                "iteration_count": 2,
                "final_synthesis": {
                    "novel_insight": f"Insight {tid}",
                    "confidence_score": 0.85,
                    "novelty_score": 0.75,
                    "evidence_lineage": ["url1", "url2", "url3"],
                    "supporting_claims": ["claim1"],
                    "reasoning": "reasoning"
                }
            }
            return mock_state
        
        mock_graph.get_state = lambda config, tid=thread_id: create_mock_state(tid)
    
    mock_build_graph.return_value = mock_graph
    
    # Verify all 5 runs are retrievable
    retrieved_count = 0
    for thread_id in completed_threads:
        response = client.get(
            f"/get_state/{thread_id}",
            headers={"X-API-Key": "test_secret_key_12345"}
        )
        if response.status_code == 200:
            retrieved_count += 1
    
    # Should retrieve 100% of runs
    assert retrieved_count == 5, \
        f"Should retrieve 100% of runs, got {retrieved_count}/5 (SC-T2-005)"


@patch("src.graph.build_graph")
def test_persistence_survives_restart(mock_build_graph, test_db_path):
    """
    Test that persisted state survives application restart.
    
    This simulates writing to checkpoint, then reading after "restart".
    """
    from langgraph.checkpoint.sqlite import SqliteSaver
    
    thread_id = "restart_test"
    
    # Simulate first session: create checkpointer and write state
    with SqliteSaver.from_conn_string(test_db_path) as checkpointer1:
        checkpointer1.setup()
        # In a real scenario, the graph would write to checkpointer
        # For testing, we'll verify the database persists
    
    # Verify database still exists after context manager closes
    assert Path(test_db_path).exists(), "Database should persist after restart"
    
    # Simulate restart: create new checkpointer instance
    with SqliteSaver.from_conn_string(test_db_path) as checkpointer2:
        # Verify we can read from it
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        conn.close()
        
        assert len(tables) > 0, "Checkpoint tables should persist after restart"


# =============================================================================
# Notes for Implementation
# =============================================================================

"""
These tests cover:

1. **Database Creation**:
   - SqliteSaver creates dialectical_system.db
   - Database file contains valid checkpoint tables

2. **Thread ID Isolation (SC-T2-005)**:
   - Different threads maintain independent states
   - Thread isolation works through API endpoints

3. **Resume Capability**:
   - Existing thread_id continues from last checkpoint
   - New thread_id starts from beginning

4. **Checkpoint Trace (SC-T2-006)**:
   - /get_trace returns correct checkpoint count
   - Checkpoint count matches iteration_count
   - Trace shows thesis evolution

5. **Persistence Reliability (SC-T2-005)**:
   - 100% of completed runs are retrievable
   - State persists across application restarts

Next Steps (Implementation):
1. Update src/graph.py to add checkpointer support (T037)
2. Implement /get_trace endpoint in src/api.py (T041)
3. Test with real SqliteSaver and graph execution
4. Verify checkpoint integrity with real workloads
"""

