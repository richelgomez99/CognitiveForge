"""
Epic 4: Task 5 - Session Lifecycle Management

Provides session lifecycle management capabilities:
1. Archival - Archive old sessions based on retention policies
2. Cleanup - Remove deleted sessions from database
3. Export - Export sessions to JSON format
4. Backup - Create point-in-time snapshots
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import models and storage
import sys
from pathlib import Path as PathLib
project_root = PathLib(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import SessionStatus, SessionMetadata
from tools import memory_store


def get_retention_days() -> int:
    """Get memory retention days from environment (default: 90)."""
    return int(os.getenv("MEMORY_RETENTION_DAYS", "90"))


def get_archive_storage() -> str:
    """Get archive storage backend from environment (default: neo4j)."""
    return os.getenv("MEMORY_ARCHIVE_STORAGE", "neo4j")


# =============================================================================
# Archival
# =============================================================================

def find_sessions_to_archive(workspace_id: str) -> List[SessionMetadata]:
    """
    Find sessions that should be archived based on retention policy.

    Sessions are eligible for archival if:
    - Status is COMPLETED
    - Older than MEMORY_RETENTION_DAYS
    - Not already ARCHIVED

    Args:
        workspace_id: Workspace identifier

    Returns:
        List of SessionMetadata objects eligible for archival
    """
    retention_days = get_retention_days()
    cutoff_date = datetime.utcnow() - timedelta(days=retention_days)

    logger.info(f"Finding sessions to archive (older than {retention_days} days, cutoff: {cutoff_date.date()})")

    # Get completed sessions
    sessions, _ = memory_store.list_sessions(
        workspace_id=workspace_id,
        status=SessionStatus.COMPLETED,
        page=1,
        page_size=1000  # Get a large batch
    )

    # Filter by age
    sessions_to_archive = [
        s for s in sessions
        if s.completed_at and s.completed_at < cutoff_date
    ]

    logger.info(f"Found {len(sessions_to_archive)} sessions eligible for archival")

    return sessions_to_archive


def archive_session(session_id: str) -> bool:
    """
    Archive a session by changing its status to ARCHIVED.

    Args:
        session_id: Session identifier

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Archiving session: {session_id}")

    try:
        # Update status to ARCHIVED
        success = memory_store.update_session(
            session_id,
            status=SessionStatus.ARCHIVED
        )

        if success:
            logger.info(f"✅ Session archived: {session_id}")
        else:
            logger.warning(f"⚠️ Failed to archive session: {session_id}")

        return success

    except Exception as e:
        logger.error(f"Error archiving session: {e}")
        return False


def archive_old_sessions(workspace_id: str) -> Dict[str, int]:
    """
    Archive all old sessions in a workspace based on retention policy.

    Args:
        workspace_id: Workspace identifier

    Returns:
        Dict with archived count and failed count
    """
    sessions_to_archive = find_sessions_to_archive(workspace_id)

    archived = 0
    failed = 0

    for session in sessions_to_archive:
        if archive_session(session.session_id):
            archived += 1
        else:
            failed += 1

    logger.info(f"✅ Archival complete: {archived} archived, {failed} failed")

    return {
        "archived": archived,
        "failed": failed,
        "total_eligible": len(sessions_to_archive)
    }


# =============================================================================
# Cleanup
# =============================================================================

def cleanup_deleted_sessions(workspace_id: str, permanent: bool = False) -> Dict[str, int]:
    """
    Clean up deleted sessions.

    By default, only counts deleted sessions (dry run).
    Set permanent=True to actually remove them from the database.

    Args:
        workspace_id: Workspace identifier
        permanent: If True, permanently delete sessions from database

    Returns:
        Dict with count of sessions cleaned up
    """
    logger.info(f"Cleaning up deleted sessions (permanent={permanent})")

    # Get deleted sessions
    sessions, _ = memory_store.list_sessions(
        workspace_id=workspace_id,
        status=SessionStatus.DELETED,
        page=1,
        page_size=1000
    )

    if not sessions:
        logger.info("No deleted sessions found")
        return {"cleaned_up": 0}

    if not permanent:
        logger.info(f"Dry run: Found {len(sessions)} deleted sessions (use permanent=True to remove)")
        return {"cleaned_up": 0, "found": len(sessions)}

    # Note: For now, we'll just keep them marked as DELETED
    # In a future version, we could implement hard deletion from Neo4j
    logger.warning("Permanent deletion not yet implemented - sessions remain marked as DELETED")

    return {
        "cleaned_up": 0,
        "found": len(sessions),
        "note": "Permanent deletion not yet implemented"
    }


# =============================================================================
# Export
# =============================================================================

def export_session_to_json(session_id: str) -> Dict[str, Any]:
    """
    Export a session to JSON format.

    Includes:
    - Session metadata
    - All debate moments
    - Compressed summary

    Args:
        session_id: Session identifier

    Returns:
        Dict with complete session data
    """
    logger.info(f"Exporting session to JSON: {session_id}")

    # Get session metadata
    session = memory_store.get_session(session_id)
    if not session:
        logger.error(f"Session not found: {session_id}")
        return {}

    # Get all moments
    moments = memory_store.get_session_moments(session_id)

    # Create export
    export_data = {
        "session_metadata": {
            "session_id": session.session_id,
            "workspace_id": session.workspace_id,
            "thread_id": session.thread_id,
            "title": session.title,
            "original_query": session.original_query,
            "status": session.status.value,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "iteration_count": session.iteration_count,
            "final_synthesis_id": session.final_synthesis_id,
            "tags": session.tags,
            "metadata": session.metadata
        },
        "debate_moments": [
            {
                "moment_id": m.moment_id,
                "round_number": m.round_number,
                "agent_type": m.agent_type,
                "content": m.content,
                "timestamp": m.timestamp.isoformat(),
                "similarity_score": m.similarity_score,
                "is_rejected": m.is_rejected,
                "paper_urls": m.paper_urls
            }
            for m in moments
        ],
        "summary": {
            "total_moments": len(moments),
            "total_rounds": max((m.round_number for m in moments), default=0),
            "analyst_moments": len([m for m in moments if m.agent_type == "analyst"]),
            "skeptic_moments": len([m for m in moments if m.agent_type == "skeptic"]),
            "synthesizer_moments": len([m for m in moments if m.agent_type == "synthesizer"])
        },
        "export_metadata": {
            "exported_at": datetime.utcnow().isoformat(),
            "version": "1.0"
        }
    }

    logger.info(f"✅ Session exported: {len(moments)} moments")

    return export_data


def save_export_to_file(export_data: Dict[str, Any], output_path: str) -> bool:
    """
    Save exported session data to a JSON file.

    Args:
        export_data: Exported session data
        output_path: Path to save file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"✅ Export saved to: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error saving export: {e}")
        return False


# =============================================================================
# Backup
# =============================================================================

def create_workspace_backup(workspace_id: str, backup_dir: str = "./backups") -> Dict[str, Any]:
    """
    Create a backup of all sessions in a workspace.

    Exports all sessions to JSON files in the backup directory.

    Args:
        workspace_id: Workspace identifier
        backup_dir: Directory to save backup files

    Returns:
        Dict with backup summary
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = Path(backup_dir) / f"workspace_{workspace_id}_{timestamp}"

    logger.info(f"Creating workspace backup: {backup_path}")

    # Get all sessions
    sessions, total = memory_store.list_sessions(
        workspace_id=workspace_id,
        status=None,  # All statuses
        page=1,
        page_size=1000
    )

    if not sessions:
        logger.info("No sessions found to backup")
        return {
            "backed_up": 0,
            "failed": 0,
            "backup_path": None
        }

    # Create backup directory
    backup_path.mkdir(parents=True, exist_ok=True)

    # Export each session
    backed_up = 0
    failed = 0

    for session in sessions:
        try:
            export_data = export_session_to_json(session.session_id)
            if export_data:
                output_file = backup_path / f"session_{session.session_id}.json"
                if save_export_to_file(export_data, str(output_file)):
                    backed_up += 1
                else:
                    failed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Error backing up session {session.session_id}: {e}")
            failed += 1

    # Create backup metadata
    metadata = {
        "workspace_id": workspace_id,
        "backup_timestamp": datetime.utcnow().isoformat(),
        "total_sessions": total,
        "backed_up": backed_up,
        "failed": failed
    }

    metadata_file = backup_path / "backup_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✅ Backup complete: {backed_up} sessions backed up, {failed} failed")
    logger.info(f"   Backup location: {backup_path}")

    return {
        "backed_up": backed_up,
        "failed": failed,
        "backup_path": str(backup_path),
        "total_sessions": total
    }
