from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


@dataclass(frozen=True)  # Immutable
class TaskDefinition:
    """Immutable task definition (creation only).

    Contains the essential identification and data for a task.
    The data dictionary must be JSON-serializable for all implementations.
    """

    id: str
    data: Dict[str, Any]  # Document as JSON-serializable


class TaskStatus(Enum):
    """Enumeration of possible task execution states."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskState:
    """Mutable task state representing execution status and results.

    Contains the immutable task definition plus execution state information.
    Result dictionaries must be JSON-serializable for all implementations.
    """

    task: TaskDefinition  # Immutable definition
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None  # Document as JSON-serializable
    error: Optional[str] = None
