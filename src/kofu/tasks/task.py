from typing import Protocol, Any, runtime_checkable


@runtime_checkable
class Task(Protocol):
    """Protocol defining the interface for executable tasks.

    A task must:
    1. Be callable with no arguments
    2. Provide a unique ID

    The ID can either be:
    - Set at task creation time
    - Derived from task parameters
    """

    @property
    def id(self) -> str:
        """Get the unique ID for this task."""
        ...

    def __call__(self) -> Any:
        """Execute the task and return its result."""
        ...
