"""
Epic 4: Persistent Memory System - Neo4j Storage Layer

This module provides the persistent memory system for CognitiveForge, enabling:
1. User and workspace management
2. Session persistence and resumption
3. Debate moment tracking
4. Cross-session pattern recognition
5. Semantic memory search

All data is stored in Neo4j for relationship-based queries and long-term persistence.
"""

import os
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from dotenv import load_dotenv
from neo4j import GraphDatabase
import numpy as np

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import models
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import (
    UserProfile, Workspace, WorkspacePermission,
    SessionMetadata, SessionRecord, SessionStatus,
    DebateMoment, MemoryPattern, MemoryPatternType,
    Permission
)


def _get_neo4j_connection():
    """Get Neo4j connection details from environment."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password123")
    return uri, user, password


# =============================================================================
# Schema Management
# =============================================================================

def initialize_memory_schema():
    """
    Initialize Neo4j schema for Epic 4 memory system.

    Creates indexes and constraints for efficient queries.
    Should be called once during system initialization.

    Returns:
        bool: True if successful, False on error
    """
    try:
        uri, user, password = _get_neo4j_connection()
        driver = GraphDatabase.driver(uri, auth=(user, password))

        with driver.session() as session:
            # User constraints
            session.run("""
                CREATE CONSTRAINT user_id_unique IF NOT EXISTS
                FOR (u:User) REQUIRE u.user_id IS UNIQUE
            """)

            session.run("""
                CREATE INDEX user_email IF NOT EXISTS
                FOR (u:User) ON (u.email)
            """)

            # Workspace constraints
            session.run("""
                CREATE CONSTRAINT workspace_id_unique IF NOT EXISTS
                FOR (w:Workspace) REQUIRE w.workspace_id IS UNIQUE
            """)

            session.run("""
                CREATE INDEX workspace_owner IF NOT EXISTS
                FOR (w:Workspace) ON (w.owner_id)
            """)

            # Session constraints
            session.run("""
                CREATE CONSTRAINT session_id_unique IF NOT EXISTS
                FOR (s:Session) REQUIRE s.session_id IS UNIQUE
            """)

            session.run("""
                CREATE INDEX session_thread_id IF NOT EXISTS
                FOR (s:Session) ON (s.thread_id)
            """)

            session.run("""
                CREATE INDEX session_workspace IF NOT EXISTS
                FOR (s:Session) ON (s.workspace_id)
            """)

            session.run("""
                CREATE INDEX session_status IF NOT EXISTS
                FOR (s:Session) ON (s.status)
            """)

            # DebateMoment constraints
            session.run("""
                CREATE CONSTRAINT moment_id_unique IF NOT EXISTS
                FOR (m:DebateMoment) REQUIRE m.moment_id IS UNIQUE
            """)

            session.run("""
                CREATE INDEX moment_session IF NOT EXISTS
                FOR (m:DebateMoment) ON (m.session_id)
            """)

            # MemoryPattern constraints
            session.run("""
                CREATE CONSTRAINT pattern_id_unique IF NOT EXISTS
                FOR (p:MemoryPattern) REQUIRE p.pattern_id IS UNIQUE
            """)

            session.run("""
                CREATE INDEX pattern_workspace IF NOT EXISTS
                FOR (p:MemoryPattern) ON (p.workspace_id)
            """)

            session.run("""
                CREATE INDEX pattern_type IF NOT EXISTS
                FOR (p:MemoryPattern) ON (p.pattern_type)
            """)

            logger.info("Memory system schema initialized successfully")

        driver.close()
        return True

    except Exception as e:
        logger.error(f"Failed to initialize memory schema: {e}")
        return False


# =============================================================================
# User Management
# =============================================================================

def create_user(user: UserProfile) -> bool:
    """
    Create a new user in the knowledge graph.

    Args:
        user: UserProfile object

    Returns:
        bool: True if successful, False on error
    """
    try:
        uri, user_uri, password = _get_neo4j_connection()
        driver = GraphDatabase.driver(uri, auth=(user_uri, password))

        with driver.session() as session:
            session.run("""
                CREATE (u:User {
                    user_id: $user_id,
                    username: $username,
                    email: $email,
                    created_at: datetime($created_at),
                    last_login: $last_login,
                    preferences: $preferences
                })
            """,
                user_id=user.user_id,
                username=user.username,
                email=user.email,
                created_at=user.created_at.isoformat(),
                last_login=user.last_login.isoformat() if user.last_login else None,
                preferences=user.preferences
            )

            logger.info(f"Created user: {user.username} ({user.user_id})")

        driver.close()
        return True

    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        return False


def get_user(user_id: str) -> Optional[UserProfile]:
    """
    Get a user by ID.

    Args:
        user_id: User identifier

    Returns:
        UserProfile if found, None otherwise
    """
    try:
        uri, user, password = _get_neo4j_connection()
        driver = GraphDatabase.driver(uri, auth=(user, password))

        with driver.session() as session:
            result = session.run("""
                MATCH (u:User {user_id: $user_id})
                RETURN u
            """, user_id=user_id)

            record = result.single()
            if record:
                u = record["u"]
                return UserProfile(
                    user_id=u["user_id"],
                    username=u["username"],
                    email=u["email"],
                    created_at=datetime.fromisoformat(u["created_at"]),
                    last_login=datetime.fromisoformat(u["last_login"]) if u.get("last_login") else None,
                    preferences=u.get("preferences", {})
                )

        driver.close()
        return None

    except Exception as e:
        logger.error(f"Failed to get user: {e}")
        return None


# =============================================================================
# Workspace Management
# =============================================================================

def create_workspace(workspace: Workspace) -> bool:
    """
    Create a new workspace and link to owner.

    Args:
        workspace: Workspace object

    Returns:
        bool: True if successful, False on error
    """
    try:
        uri, user, password = _get_neo4j_connection()
        driver = GraphDatabase.driver(uri, auth=(user, password))

        with driver.session() as session:
            # Create workspace node
            session.run("""
                CREATE (w:Workspace {
                    workspace_id: $workspace_id,
                    name: $name,
                    owner_id: $owner_id,
                    created_at: datetime($created_at),
                    description: $description,
                    is_public: $is_public,
                    settings: $settings
                })
            """,
                workspace_id=workspace.workspace_id,
                name=workspace.name,
                owner_id=workspace.owner_id,
                created_at=workspace.created_at.isoformat(),
                description=workspace.description,
                is_public=workspace.is_public,
                settings=workspace.settings
            )

            # Create ownership relationship
            session.run("""
                MATCH (u:User {user_id: $owner_id})
                MATCH (w:Workspace {workspace_id: $workspace_id})
                CREATE (u)-[:OWNS]->(w)
            """,
                owner_id=workspace.owner_id,
                workspace_id=workspace.workspace_id
            )

            logger.info(f"Created workspace: {workspace.name} ({workspace.workspace_id})")

        driver.close()
        return True

    except Exception as e:
        logger.error(f"Failed to create workspace: {e}")
        return False


def get_workspace(workspace_id: str) -> Optional[Workspace]:
    """
    Get a workspace by ID.

    Args:
        workspace_id: Workspace identifier

    Returns:
        Workspace if found, None otherwise
    """
    try:
        uri, user, password = _get_neo4j_connection()
        driver = GraphDatabase.driver(uri, auth=(user, password))

        with driver.session() as session:
            result = session.run("""
                MATCH (w:Workspace {workspace_id: $workspace_id})
                RETURN w
            """, workspace_id=workspace_id)

            record = result.single()
            if record:
                w = record["w"]
                return Workspace(
                    workspace_id=w["workspace_id"],
                    name=w["name"],
                    owner_id=w["owner_id"],
                    created_at=datetime.fromisoformat(w["created_at"]),
                    description=w.get("description"),
                    is_public=w.get("is_public", False),
                    settings=w.get("settings", {})
                )

        driver.close()
        return None

    except Exception as e:
        logger.error(f"Failed to get workspace: {e}")
        return None


def list_user_workspaces(user_id: str) -> List[Workspace]:
    """
    List all workspaces owned by or accessible to a user.

    Args:
        user_id: User identifier

    Returns:
        List of Workspace objects
    """
    try:
        uri, user, password = _get_neo4j_connection()
        driver = GraphDatabase.driver(uri, auth=(user, password))

        workspaces = []

        with driver.session() as session:
            # Get owned and shared workspaces
            result = session.run("""
                MATCH (u:User {user_id: $user_id})-[:OWNS]->(w:Workspace)
                RETURN w
                UNION
                MATCH (u:User {user_id: $user_id})-[:HAS_PERMISSION]->(w:Workspace)
                RETURN w
            """, user_id=user_id)

            for record in result:
                w = record["w"]
                workspaces.append(Workspace(
                    workspace_id=w["workspace_id"],
                    name=w["name"],
                    owner_id=w["owner_id"],
                    created_at=datetime.fromisoformat(w["created_at"]),
                    description=w.get("description"),
                    is_public=w.get("is_public", False),
                    settings=w.get("settings", {})
                ))

        driver.close()
        return workspaces

    except Exception as e:
        logger.error(f"Failed to list workspaces: {e}")
        return []


# =============================================================================
# Session Management
# =============================================================================

def create_session(session: SessionMetadata, created_by: str) -> bool:
    """
    Create a new debate session and link to workspace.

    Args:
        session: SessionMetadata object
        created_by: User ID who created the session

    Returns:
        bool: True if successful, False on error
    """
    try:
        uri, user, password = _get_neo4j_connection()
        driver = GraphDatabase.driver(uri, auth=(user, password))

        with driver.session() as db_session:
            # Create session node
            db_session.run("""
                CREATE (s:Session {
                    session_id: $session_id,
                    workspace_id: $workspace_id,
                    thread_id: $thread_id,
                    title: $title,
                    original_query: $original_query,
                    status: $status,
                    created_at: datetime($created_at),
                    updated_at: datetime($updated_at),
                    completed_at: $completed_at,
                    iteration_count: $iteration_count,
                    final_synthesis_id: $final_synthesis_id,
                    tags: $tags,
                    metadata: $metadata
                })
            """,
                session_id=session.session_id,
                workspace_id=session.workspace_id,
                thread_id=session.thread_id,
                title=session.title,
                original_query=session.original_query,
                status=session.status.value,
                created_at=session.created_at.isoformat(),
                updated_at=session.updated_at.isoformat(),
                completed_at=session.completed_at.isoformat() if session.completed_at else None,
                iteration_count=session.iteration_count,
                final_synthesis_id=session.final_synthesis_id,
                tags=session.tags,
                metadata=session.metadata
            )

            # Link to workspace
            db_session.run("""
                MATCH (w:Workspace {workspace_id: $workspace_id})
                MATCH (s:Session {session_id: $session_id})
                CREATE (w)-[:CONTAINS]->(s)
            """,
                workspace_id=session.workspace_id,
                session_id=session.session_id
            )

            # Link to creator
            db_session.run("""
                MATCH (u:User {user_id: $user_id})
                MATCH (s:Session {session_id: $session_id})
                CREATE (u)-[:CREATED]->(s)
            """,
                user_id=created_by,
                session_id=session.session_id
            )

            logger.info(f"Created session: {session.title} ({session.session_id})")

        driver.close()
        return True

    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        return False


def get_session(session_id: str) -> Optional[SessionMetadata]:
    """
    Get a session by ID.

    Args:
        session_id: Session identifier

    Returns:
        SessionMetadata if found, None otherwise
    """
    try:
        uri, user, password = _get_neo4j_connection()
        driver = GraphDatabase.driver(uri, auth=(user, password))

        with driver.session() as db_session:
            result = db_session.run("""
                MATCH (s:Session {session_id: $session_id})
                RETURN s
            """, session_id=session_id)

            record = result.single()
            if record:
                s = record["s"]
                return SessionMetadata(
                    session_id=s["session_id"],
                    workspace_id=s["workspace_id"],
                    thread_id=s["thread_id"],
                    title=s["title"],
                    original_query=s["original_query"],
                    status=SessionStatus(s["status"]),
                    created_at=datetime.fromisoformat(s["created_at"]),
                    updated_at=datetime.fromisoformat(s["updated_at"]),
                    completed_at=datetime.fromisoformat(s["completed_at"]) if s.get("completed_at") else None,
                    iteration_count=s["iteration_count"],
                    final_synthesis_id=s.get("final_synthesis_id"),
                    tags=s.get("tags", []),
                    metadata=s.get("metadata", {})
                )

        driver.close()
        return None

    except Exception as e:
        logger.error(f"Failed to get session: {e}")
        return None


def update_session(session_id: str, **updates) -> bool:
    """
    Update session metadata.

    Args:
        session_id: Session identifier
        **updates: Fields to update (title, status, tags, etc.)

    Returns:
        bool: True if successful, False on error
    """
    try:
        uri, user, password = _get_neo4j_connection()
        driver = GraphDatabase.driver(uri, auth=(user, password))

        with driver.session() as db_session:
            # Build SET clause dynamically
            set_clauses = ["s.updated_at = datetime()"]
            params = {"session_id": session_id}

            if "title" in updates:
                set_clauses.append("s.title = $title")
                params["title"] = updates["title"]

            if "status" in updates:
                set_clauses.append("s.status = $status")
                params["status"] = updates["status"].value if isinstance(updates["status"], SessionStatus) else updates["status"]

            if "tags" in updates:
                set_clauses.append("s.tags = $tags")
                params["tags"] = updates["tags"]

            if "iteration_count" in updates:
                set_clauses.append("s.iteration_count = $iteration_count")
                params["iteration_count"] = updates["iteration_count"]

            if "final_synthesis_id" in updates:
                set_clauses.append("s.final_synthesis_id = $final_synthesis_id")
                params["final_synthesis_id"] = updates["final_synthesis_id"]

            if "completed_at" in updates:
                set_clauses.append("s.completed_at = datetime($completed_at)")
                params["completed_at"] = updates["completed_at"].isoformat() if updates["completed_at"] else None

            query = f"""
                MATCH (s:Session {{session_id: $session_id}})
                SET {', '.join(set_clauses)}
                RETURN s
            """

            db_session.run(query, **params)
            logger.info(f"Updated session: {session_id}")

        driver.close()
        return True

    except Exception as e:
        logger.error(f"Failed to update session: {e}")
        return False


def list_sessions(
    workspace_id: str,
    status: Optional[SessionStatus] = None,
    page: int = 1,
    page_size: int = 20
) -> Tuple[List[SessionMetadata], int]:
    """
    List sessions in a workspace with pagination.

    Args:
        workspace_id: Workspace identifier
        status: Optional status filter
        page: Page number (1-indexed)
        page_size: Number of sessions per page

    Returns:
        Tuple of (sessions, total_count)
    """
    try:
        uri, user, password = _get_neo4j_connection()
        driver = GraphDatabase.driver(uri, auth=(user, password))

        sessions = []
        total = 0

        with driver.session() as db_session:
            # Build query
            where_clause = "WHERE s.workspace_id = $workspace_id"
            params = {"workspace_id": workspace_id}

            if status:
                where_clause += " AND s.status = $status"
                params["status"] = status.value

            # Get total count
            count_query = f"""
                MATCH (s:Session)
                {where_clause}
                RETURN count(s) AS total
            """
            count_result = db_session.run(count_query, **params)
            total = count_result.single()["total"]

            # Get paginated results
            skip = (page - 1) * page_size
            params["skip"] = skip
            params["limit"] = page_size

            list_query = f"""
                MATCH (s:Session)
                {where_clause}
                RETURN s
                ORDER BY s.created_at DESC
                SKIP $skip
                LIMIT $limit
            """

            result = db_session.run(list_query, **params)

            for record in result:
                s = record["s"]
                sessions.append(SessionMetadata(
                    session_id=s["session_id"],
                    workspace_id=s["workspace_id"],
                    thread_id=s["thread_id"],
                    title=s["title"],
                    original_query=s["original_query"],
                    status=SessionStatus(s["status"]),
                    created_at=datetime.fromisoformat(s["created_at"]),
                    updated_at=datetime.fromisoformat(s["updated_at"]),
                    completed_at=datetime.fromisoformat(s["completed_at"]) if s.get("completed_at") else None,
                    iteration_count=s["iteration_count"],
                    final_synthesis_id=s.get("final_synthesis_id"),
                    tags=s.get("tags", []),
                    metadata=s.get("metadata", {})
                ))

        driver.close()
        return sessions, total

    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        return [], 0


def delete_session(session_id: str) -> bool:
    """
    Delete a session (soft delete by setting status to DELETED).

    Args:
        session_id: Session identifier

    Returns:
        bool: True if successful, False on error
    """
    return update_session(session_id, status=SessionStatus.DELETED)


# =============================================================================
# Debate Moment Tracking
# =============================================================================

def add_debate_moment(moment: DebateMoment) -> bool:
    """
    Add a debate moment to the knowledge graph.

    Args:
        moment: DebateMoment object

    Returns:
        bool: True if successful, False on error
    """
    try:
        uri, user, password = _get_neo4j_connection()
        driver = GraphDatabase.driver(uri, auth=(user, password))

        with driver.session() as session:
            # Create moment node
            session.run("""
                CREATE (m:DebateMoment {
                    moment_id: $moment_id,
                    session_id: $session_id,
                    round_number: $round_number,
                    agent_type: $agent_type,
                    content: $content,
                    timestamp: datetime($timestamp),
                    similarity_score: $similarity_score,
                    is_rejected: $is_rejected,
                    paper_urls: $paper_urls,
                    embedding: $embedding
                })
            """,
                moment_id=moment.moment_id,
                session_id=moment.session_id,
                round_number=moment.round_number,
                agent_type=moment.agent_type,
                content=moment.content,
                timestamp=moment.timestamp.isoformat(),
                similarity_score=moment.similarity_score,
                is_rejected=moment.is_rejected,
                paper_urls=moment.paper_urls,
                embedding=moment.embedding
            )

            # Link to session
            session.run("""
                MATCH (s:Session {session_id: $session_id})
                MATCH (m:DebateMoment {moment_id: $moment_id})
                CREATE (s)-[:HAS_MOMENT]->(m)
            """,
                session_id=moment.session_id,
                moment_id=moment.moment_id
            )

            # Link to cited papers
            for paper_url in moment.paper_urls:
                session.run("""
                    MATCH (m:DebateMoment {moment_id: $moment_id})
                    MATCH (p:ResearchPaper {url: $url})
                    MERGE (m)-[:CITES]->(p)
                """,
                    moment_id=moment.moment_id,
                    url=paper_url
                )

            logger.info(f"Added debate moment: {moment.agent_type} round {moment.round_number}")

        driver.close()
        return True

    except Exception as e:
        logger.error(f"Failed to add debate moment: {e}")
        return False


def get_session_moments(session_id: str) -> List[DebateMoment]:
    """
    Get all debate moments for a session.

    Args:
        session_id: Session identifier

    Returns:
        List of DebateMoment objects
    """
    try:
        uri, user, password = _get_neo4j_connection()
        driver = GraphDatabase.driver(uri, auth=(user, password))

        moments = []

        with driver.session() as session:
            result = session.run("""
                MATCH (s:Session {session_id: $session_id})-[:HAS_MOMENT]->(m:DebateMoment)
                RETURN m
                ORDER BY m.round_number, m.timestamp
            """, session_id=session_id)

            for record in result:
                m = record["m"]
                moments.append(DebateMoment(
                    moment_id=m["moment_id"],
                    session_id=m["session_id"],
                    round_number=m["round_number"],
                    agent_type=m["agent_type"],
                    content=m["content"],
                    timestamp=datetime.fromisoformat(m["timestamp"]),
                    similarity_score=m.get("similarity_score"),
                    is_rejected=m.get("is_rejected", False),
                    paper_urls=m.get("paper_urls", []),
                    embedding=m.get("embedding")
                ))

        driver.close()
        return moments

    except Exception as e:
        logger.error(f"Failed to get session moments: {e}")
        return []


# =============================================================================
# Memory Pattern Management
# =============================================================================

def add_memory_pattern(pattern: MemoryPattern) -> bool:
    """
    Add a memory pattern to the knowledge graph.

    Args:
        pattern: MemoryPattern object

    Returns:
        bool: True if successful, False on error
    """
    try:
        uri, user, password = _get_neo4j_connection()
        driver = GraphDatabase.driver(uri, auth=(user, password))

        with driver.session() as session:
            session.run("""
                CREATE (p:MemoryPattern {
                    pattern_id: $pattern_id,
                    workspace_id: $workspace_id,
                    pattern_type: $pattern_type,
                    description: $description,
                    frequency: $frequency,
                    last_seen: datetime($last_seen),
                    example_sessions: $example_sessions,
                    embedding: $embedding,
                    metadata: $metadata
                })
            """,
                pattern_id=pattern.pattern_id,
                workspace_id=pattern.workspace_id,
                pattern_type=pattern.pattern_type.value,
                description=pattern.description,
                frequency=pattern.frequency,
                last_seen=pattern.last_seen.isoformat(),
                example_sessions=pattern.example_sessions,
                embedding=pattern.embedding,
                metadata=pattern.metadata
            )

            # Link to workspace
            session.run("""
                MATCH (w:Workspace {workspace_id: $workspace_id})
                MATCH (p:MemoryPattern {pattern_id: $pattern_id})
                CREATE (w)-[:HAS_PATTERN]->(p)
            """,
                workspace_id=pattern.workspace_id,
                pattern_id=pattern.pattern_id
            )

            logger.info(f"Added memory pattern: {pattern.pattern_type.value}")

        driver.close()
        return True

    except Exception as e:
        logger.error(f"Failed to add memory pattern: {e}")
        return False


def search_similar_moments(
    query_embedding: List[float],
    workspace_id: str,
    limit: int = 10,
    similarity_threshold: float = 0.75
) -> List[Tuple[DebateMoment, float]]:
    """
    Search for similar debate moments using cosine similarity.

    Args:
        query_embedding: Query vector embedding
        workspace_id: Workspace to search within
        limit: Maximum results to return
        similarity_threshold: Minimum similarity score (0-1)

    Returns:
        List of (DebateMoment, similarity_score) tuples
    """
    try:
        uri, user, password = _get_neo4j_connection()
        driver = GraphDatabase.driver(uri, auth=(user, password))

        results = []

        with driver.session() as session:
            # Get all moments in workspace with embeddings
            result = session.run("""
                MATCH (w:Workspace {workspace_id: $workspace_id})-[:CONTAINS]->(s:Session)-[:HAS_MOMENT]->(m:DebateMoment)
                WHERE m.embedding IS NOT NULL
                RETURN m, s.session_id AS session_id
            """, workspace_id=workspace_id)

            query_vec = np.array(query_embedding)

            for record in result:
                m = record["m"]
                moment_embedding = m.get("embedding")

                if not moment_embedding:
                    continue

                # Compute cosine similarity
                moment_vec = np.array(moment_embedding)
                similarity = np.dot(query_vec, moment_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(moment_vec)
                )

                if similarity >= similarity_threshold:
                    moment = DebateMoment(
                        moment_id=m["moment_id"],
                        session_id=m["session_id"],
                        round_number=m["round_number"],
                        agent_type=m["agent_type"],
                        content=m["content"],
                        timestamp=datetime.fromisoformat(m["timestamp"]),
                        similarity_score=m.get("similarity_score"),
                        is_rejected=m.get("is_rejected", False),
                        paper_urls=m.get("paper_urls", []),
                        embedding=moment_embedding
                    )
                    results.append((moment, float(similarity)))

            # Sort by similarity descending
            results.sort(key=lambda x: x[1], reverse=True)

            # Return top k
            return results[:limit]

        driver.close()

    except Exception as e:
        logger.error(f"Failed to search similar moments: {e}")
        return []
