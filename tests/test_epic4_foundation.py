"""
Tests for Epic 4: Persistent Memory System - Foundation (Tasks 1-2)

Tests basic functionality of:
- Data models (SessionMetadata, UserProfile, Workspace)
- Model validation
- Enums
- Import integrity
"""

import pytest
from datetime import datetime
from pydantic import ValidationError


def test_imports():
    """Test that all Epic 4 models can be imported."""
    from src.models import (
        SessionStatus, Permission, MemoryPatternType,
        UserProfile, Workspace, WorkspacePermission,
        SessionMetadata, SessionRecord, CreateSessionRequest, UpdateSessionRequest, SessionListResponse,
        DebateMoment, MemoryPattern,
        MemorySearchRequest, MemorySearchResult, MemorySearchResponse
    )

    # Test that enums have correct values
    assert SessionStatus.ACTIVE.value == "active"
    assert SessionStatus.PAUSED.value == "paused"
    assert SessionStatus.COMPLETED.value == "completed"
    assert SessionStatus.ARCHIVED.value == "archived"
    assert SessionStatus.DELETED.value == "deleted"

    assert Permission.VIEW.value == "view"
    assert Permission.EDIT.value == "edit"
    assert Permission.ADMIN.value == "admin"

    assert MemoryPatternType.CLAIM_STRUCTURE.value == "claim_structure"
    assert MemoryPatternType.SKEPTIC_OBJECTION.value == "skeptic_objection_type"


def test_user_profile_creation():
    """Test creating a valid UserProfile."""
    from src.models import UserProfile

    user = UserProfile(
        username="testuser",
        email="test@example.com"
    )

    # Check that UUID is generated
    assert user.user_id is not None
    assert len(user.user_id) > 0

    # Check fields
    assert user.username == "testuser"
    assert user.email == "test@example.com"
    assert isinstance(user.created_at, datetime)
    assert user.last_login is None
    assert user.preferences == {}


def test_user_profile_validation():
    """Test UserProfile validation."""
    from src.models import UserProfile

    # Username too short
    with pytest.raises(ValidationError):
        UserProfile(username="ab", email="test@example.com")

    # Invalid email
    with pytest.raises(ValidationError):
        UserProfile(username="testuser", email="invalid-email")


def test_workspace_creation():
    """Test creating a valid Workspace."""
    from src.models import Workspace

    workspace = Workspace(
        name="Test Workspace",
        owner_id="user-123"
    )

    # Check that UUID is generated
    assert workspace.workspace_id is not None
    assert len(workspace.workspace_id) > 0

    # Check fields
    assert workspace.name == "Test Workspace"
    assert workspace.owner_id == "user-123"
    assert isinstance(workspace.created_at, datetime)
    assert workspace.description is None
    assert workspace.is_public is False
    assert workspace.settings == {}


def test_session_metadata_creation():
    """Test creating a valid SessionMetadata."""
    from src.models import SessionMetadata, SessionStatus

    session = SessionMetadata(
        workspace_id="workspace-123",
        thread_id="thread-456",
        title="Test Session",
        original_query="What is machine learning?"
    )

    # Check that UUID is generated
    assert session.session_id is not None
    assert len(session.session_id) > 0

    # Check fields
    assert session.workspace_id == "workspace-123"
    assert session.thread_id == "thread-456"
    assert session.title == "Test Session"
    assert session.original_query == "What is machine learning?"
    assert session.status == SessionStatus.ACTIVE
    assert isinstance(session.created_at, datetime)
    assert isinstance(session.updated_at, datetime)
    assert session.completed_at is None
    assert session.iteration_count == 0
    assert session.final_synthesis_id is None
    assert session.tags == []
    assert session.metadata == {}


def test_session_metadata_validation():
    """Test SessionMetadata validation."""
    from src.models import SessionMetadata

    # Title too short
    with pytest.raises(ValidationError):
        SessionMetadata(
            workspace_id="workspace-123",
            thread_id="thread-456",
            title="ab",  # Too short
            original_query="What is machine learning?"
        )

    # Query too short
    with pytest.raises(ValidationError):
        SessionMetadata(
            workspace_id="workspace-123",
            thread_id="thread-456",
            title="Test Session",
            original_query="short"  # Too short
        )


def test_create_session_request():
    """Test CreateSessionRequest."""
    from src.models import CreateSessionRequest

    request = CreateSessionRequest(
        workspace_id="workspace-123",
        title="My Research Session",
        query="How do neural networks learn?",
        tags=["ai", "neural-networks"]
    )

    assert request.workspace_id == "workspace-123"
    assert request.title == "My Research Session"
    assert request.query == "How do neural networks learn?"
    assert request.tags == ["ai", "neural-networks"]


def test_update_session_request():
    """Test UpdateSessionRequest with partial updates."""
    from src.models import UpdateSessionRequest, SessionStatus

    # Update only title
    request1 = UpdateSessionRequest(title="Updated Title")
    assert request1.title == "Updated Title"
    assert request1.status is None
    assert request1.tags is None

    # Update only status
    request2 = UpdateSessionRequest(status=SessionStatus.COMPLETED)
    assert request2.title is None
    assert request2.status == SessionStatus.COMPLETED
    assert request2.tags is None

    # Update all fields
    request3 = UpdateSessionRequest(
        title="New Title",
        status=SessionStatus.PAUSED,
        tags=["updated", "tags"]
    )
    assert request3.title == "New Title"
    assert request3.status == SessionStatus.PAUSED
    assert request3.tags == ["updated", "tags"]


def test_debate_moment_creation():
    """Test creating a valid DebateMoment."""
    from src.models import DebateMoment

    moment = DebateMoment(
        session_id="session-123",
        round_number=1,
        agent_type="analyst",
        content="This is a thesis claim about neural networks.",
        paper_urls=["https://arxiv.org/abs/1234.5678"]
    )

    # Check that UUID is generated
    assert moment.moment_id is not None
    assert len(moment.moment_id) > 0

    # Check fields
    assert moment.session_id == "session-123"
    assert moment.round_number == 1
    assert moment.agent_type == "analyst"
    assert moment.content == "This is a thesis claim about neural networks."
    assert isinstance(moment.timestamp, datetime)
    assert moment.similarity_score is None
    assert moment.is_rejected is False
    assert moment.paper_urls == ["https://arxiv.org/abs/1234.5678"]
    assert moment.embedding is None


def test_memory_pattern_creation():
    """Test creating a valid MemoryPattern."""
    from src.models import MemoryPattern, MemoryPatternType

    pattern = MemoryPattern(
        workspace_id="workspace-123",
        pattern_type=MemoryPatternType.CLAIM_STRUCTURE,
        description="Pattern: Claims about neural network generalization tend to be rejected when lacking empirical evidence."
    )

    # Check that UUID is generated
    assert pattern.pattern_id is not None
    assert len(pattern.pattern_id) > 0

    # Check fields
    assert pattern.workspace_id == "workspace-123"
    assert pattern.pattern_type == MemoryPatternType.CLAIM_STRUCTURE
    assert "Pattern:" in pattern.description
    assert pattern.frequency == 1
    assert isinstance(pattern.last_seen, datetime)
    assert pattern.example_sessions == []
    assert pattern.embedding is None
    assert pattern.metadata == {}


def test_memory_search_request():
    """Test MemorySearchRequest."""
    from src.models import MemorySearchRequest, MemoryPatternType

    request = MemorySearchRequest(
        query="neural network generalization",
        workspace_id="workspace-123",
        limit=5,
        similarity_threshold=0.8,
        pattern_types=[MemoryPatternType.CLAIM_STRUCTURE, MemoryPatternType.EVIDENCE_QUALITY]
    )

    assert request.query == "neural network generalization"
    assert request.workspace_id == "workspace-123"
    assert request.limit == 5
    assert request.similarity_threshold == 0.8
    assert len(request.pattern_types) == 2


def test_memory_store_imports():
    """Test that memory_store module can be imported."""
    from src.tools import memory_store

    # Check that key functions exist
    assert hasattr(memory_store, 'initialize_memory_schema')
    assert hasattr(memory_store, 'create_user')
    assert hasattr(memory_store, 'get_user')
    assert hasattr(memory_store, 'create_workspace')
    assert hasattr(memory_store, 'get_workspace')
    assert hasattr(memory_store, 'create_session')
    assert hasattr(memory_store, 'get_session')
    assert hasattr(memory_store, 'update_session')
    assert hasattr(memory_store, 'delete_session')
    assert hasattr(memory_store, 'list_sessions')
    assert hasattr(memory_store, 'add_debate_moment')
    assert hasattr(memory_store, 'get_session_moments')
    assert hasattr(memory_store, 'add_memory_pattern')
    assert hasattr(memory_store, 'search_similar_moments')


def test_session_list_response():
    """Test SessionListResponse."""
    from src.models import SessionListResponse, SessionMetadata

    session1 = SessionMetadata(
        workspace_id="workspace-123",
        thread_id="thread-1",
        title="Session 1",
        original_query="What is machine learning?"
    )

    session2 = SessionMetadata(
        workspace_id="workspace-123",
        thread_id="thread-2",
        title="Session 2",
        original_query="How do neural networks work?"
    )

    response = SessionListResponse(
        sessions=[session1, session2],
        total=2,
        page=1,
        page_size=20
    )

    assert len(response.sessions) == 2
    assert response.total == 2
    assert response.page == 1
    assert response.page_size == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
