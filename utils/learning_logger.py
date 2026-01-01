import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import UUID

logger = logging.getLogger(__name__)

# Get backend directory (parent of utils directory)
BACKEND_DIR = Path(__file__).parent.parent
LEARNING_LOG_FILE = os.getenv("LEARNING_LOG_FILE", "learning_changes.json")
LOG_DIR = Path(os.getenv("LOG_DIR", str(BACKEND_DIR / "logs")))
LOG_FILE_PATH = LOG_DIR / LEARNING_LOG_FILE


def ensure_log_file():
    """Ensure log directory and file exist."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE_PATH.exists():
        initial_data = {
            "previous_changes": [],
            "current_changes": [],
            "future_changes": [],
            "last_updated": None,
        }
        with open(LOG_FILE_PATH, "w") as f:
            json.dump(initial_data, f, indent=2, default=str)


def load_log() -> Dict[str, Any]:
    """Load the learning log from JSON file."""
    ensure_log_file()
    try:
        with open(LOG_FILE_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading learning log: {str(e)}")
        return {
            "previous_changes": [],
            "current_changes": [],
            "future_changes": [],
            "last_updated": None,
        }


def save_log(data: Dict[str, Any]):
    """Save the learning log to JSON file."""
    ensure_log_file()
    data["last_updated"] = datetime.utcnow().isoformat()
    try:
        with open(LOG_FILE_PATH, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Error saving learning log: {str(e)}")


def log_calibration_change(
    previous_offset: Optional[float],
    new_offset: float,
    sample_size: int,
    feedback_count: int,
    change_type: str = "update",
):
    """
    Log a calibration change.
    
    Args:
        previous_offset: Previous calibration offset
        new_offset: New calibration offset
        sample_size: Sample size used for calculation
        feedback_count: Total feedback count
        change_type: Type of change (update, initial, reset)
    """
    log_data = load_log()
    
    change_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "type": "calibration",
        "change_type": change_type,
        "previous_offset": previous_offset,
        "new_offset": new_offset,
        "offset_delta": new_offset - (previous_offset or 0.0),
        "sample_size": sample_size,
        "feedback_count": feedback_count,
    }
    
    log_data["current_changes"].append(change_entry)
    
    # Cleanup: Keep only the 5 most recent good learning jobs
    _cleanup_learning_jobs(log_data)
    
    save_log(log_data)
    logger.info(f"Logged calibration change: {change_type} from {previous_offset} to {new_offset}")


def log_prompt_version_change(
    version_id: UUID,
    version_name: str,
    change_type: str,
    previous_state: Optional[Dict[str, Any]] = None,
    new_state: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
    comparison: Optional[Dict[str, Any]] = None,
):
    """
    Log a prompt version change.
    
    Args:
        version_id: Prompt version ID
        version_name: Version name
        change_type: Type of change (created, activated, promoted, evaluated, updated)
        previous_state: Previous state (for promotions/updates)
        new_state: New state
        metrics: Performance metrics
        comparison: Comparison data (for promotions)
    """
    log_data = load_log()
    
    change_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "type": "prompt_version",
        "change_type": change_type,
        "version_id": str(version_id),
        "version_name": version_name,
        "previous_state": previous_state,
        "new_state": new_state,
        "metrics": metrics,
        "comparison": comparison,
    }
    
    log_data["current_changes"].append(change_entry)
    
    # Cleanup: Keep only the 5 most recent good learning jobs
    _cleanup_learning_jobs(log_data)
    
    save_log(log_data)
    logger.info(f"Logged prompt version change: {change_type} for {version_name}")


def log_feedback_pattern_change(
    patterns: Dict[str, Any],
    feedback_count: int,
    change_type: str = "analysis",
):
    """
    Log feedback pattern analysis.
    
    Args:
        patterns: Extracted patterns
        feedback_count: Number of feedback items analyzed
        change_type: Type of change (analysis, update)
    """
    log_data = load_log()
    
    change_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "type": "feedback_patterns",
        "change_type": change_type,
        "patterns": patterns,
        "feedback_count": feedback_count,
    }
    
    log_data["current_changes"].append(change_entry)
    
    # Cleanup: Keep only the 5 most recent good learning jobs
    _cleanup_learning_jobs(log_data)
    
    save_log(log_data)
    logger.info(f"Logged feedback pattern change: {change_type} with {feedback_count} feedback items")


def log_future_change(
    change_type: str,
    description: str,
    planned_date: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Log a planned/future change.
    
    Args:
        change_type: Type of planned change
        description: Description of the planned change
        planned_date: Planned date (ISO format)
        metadata: Additional metadata
    """
    log_data = load_log()
    
    future_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "type": change_type,
        "description": description,
        "planned_date": planned_date,
        "metadata": metadata,
    }
    
    log_data["future_changes"].append(future_entry)
    
    # Keep only last 50 future changes
    if len(log_data["future_changes"]) > 50:
        log_data["future_changes"] = log_data["future_changes"][-50:]
    
    save_log(log_data)
    logger.info(f"Logged future change: {change_type} - {description}")


def mark_future_change_completed(
    change_type: str,
    description: Optional[str] = None,
):
    """
    Mark a future change as completed and move it to current changes.
    
    Args:
        change_type: Type of change to mark as completed
        description: Optional description to match
    """
    log_data = load_log()
    
    future_changes = log_data.get("future_changes", [])
    completed = []
    remaining = []
    
    for change in future_changes:
        if change.get("type") == change_type:
            if description is None or change.get("description") == description:
                completed.append(change)
                # Move to current changes
                change["completed_at"] = datetime.utcnow().isoformat()
                log_data["current_changes"].append(change)
            else:
                remaining.append(change)
        else:
            remaining.append(change)
    
    log_data["future_changes"] = remaining
    save_log(log_data)
    
    if completed:
        logger.info(f"Marked {len(completed)} future change(s) as completed: {change_type}")


def get_learning_log() -> Dict[str, Any]:
    """Get the complete learning log."""
    return load_log()


def get_changes_by_type(change_type: str, include_previous: bool = True) -> List[Dict[str, Any]]:
    """
    Get changes filtered by type.
    
    Args:
        change_type: Type of change to filter (calibration, prompt_version, feedback_patterns)
        include_previous: Whether to include previous changes
    
    Returns:
        List of matching changes
    """
    log_data = load_log()
    changes = []
    
    if include_previous:
        changes.extend([c for c in log_data.get("previous_changes", []) if c.get("type") == change_type])
    
    changes.extend([c for c in log_data.get("current_changes", []) if c.get("type") == change_type])
    
    return sorted(changes, key=lambda x: x.get("timestamp", ""), reverse=True)


def get_recent_changes(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Get recent changes across all types.
    
    Args:
        limit: Maximum number of changes to return
    
    Returns:
        List of recent changes
    """
    log_data = load_log()
    all_changes = []
    
    all_changes.extend(log_data.get("previous_changes", []))
    all_changes.extend(log_data.get("current_changes", []))
    
    return sorted(all_changes, key=lambda x: x.get("timestamp", ""), reverse=True)[:limit]


def _cleanup_learning_jobs(log_data: Dict[str, Any]):
    """
    Cleanup learning jobs: Keep only the 5 most recent good learning jobs.
    
    A "good" learning job is one that:
    - Has type "calibration" or "prompt_version"
    - Has a successful outcome (not an error)
    
    Args:
        log_data: The learning log data dictionary
    """
    # Get all learning-related changes (calibration and prompt_version)
    all_learning_jobs = []
    
    # Collect from current changes
    for change in log_data.get("current_changes", []):
        if change.get("type") in ["calibration", "prompt_version"]:
            all_learning_jobs.append(change)
    
    # Collect from previous changes
    for change in log_data.get("previous_changes", []):
        if change.get("type") in ["calibration", "prompt_version"]:
            all_learning_jobs.append(change)
    
    # Sort by timestamp (most recent first)
    all_learning_jobs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    # Keep only the 5 most recent good jobs
    good_jobs = all_learning_jobs[:5]
    good_job_timestamps = {job.get("timestamp") for job in good_jobs}
    
    # Filter current_changes to keep only good jobs
    log_data["current_changes"] = [
        change for change in log_data.get("current_changes", [])
        if change.get("type") not in ["calibration", "prompt_version"] 
        or change.get("timestamp") in good_job_timestamps
    ]
    
    # Filter previous_changes to keep only good jobs
    log_data["previous_changes"] = [
        change for change in log_data.get("previous_changes", [])
        if change.get("type") not in ["calibration", "prompt_version"]
        or change.get("timestamp") in good_job_timestamps
    ]
    
    if len(all_learning_jobs) > 5:
        logger.info(f"Cleaned up learning jobs: kept {len(good_jobs)} most recent, removed {len(all_learning_jobs) - len(good_jobs)} old jobs")

