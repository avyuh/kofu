from abc import ABC, abstractmethod
from typing import List, Iterator
from contextlib import contextmanager
from collections import defaultdict
from typing import Any, Dict, Optional

from .task_state import TaskDefinition, TaskState, TaskStatus


class TaskStore(ABC):
    """Abstract base class for task storage, optimized for bulk operations.

    Designed for Kofu's executors and storage extensions only. End users should
    use the utility functions defined at the module level instead of directly
    interacting with TaskStore methods.

    Implementations must be thread-safe and handle concurrent access appropriately.
    """

    # Bulk operations (primary interface)
    @abstractmethod
    def get_many(self, task_ids: List[str]) -> List[TaskState]:
        """Retrieve states of multiple tasks by their IDs.

        If some task IDs don't exist, the implementation should return only the TaskState objects for existing IDs

        Args:
            task_ids: List of task IDs to retrieve

        Returns:
            List of TaskState objects for the requested tasks

        Raises:
            KeyError: If any task ID doesn't exist (implementation-specific)
        """
        pass

    @abstractmethod
    def put_many(self, tasks: List[TaskDefinition]) -> None:
        """Create multiple new tasks with PENDING status.

        If a task with the same ID already exists, the implementation should overwrite existing tasks with new PENDING tasks (resetting them)

        Args:
            tasks: List of Task objects to create

        Raises:
            ValueError: If any task ID already exists (implementation-specific)
        """
        pass

    @abstractmethod
    def set_many(self, states: List[TaskState]) -> None:
        """Update states of multiple existing tasks.

        If some task IDs don't exist, the implementation should silently ignore non-existent task updates.

        Args:
            states: List of TaskState objects to update

        Raises:
            KeyError: If any task ID doesn't exist (implementation-specific)
        """
        pass

    @abstractmethod
    def delete_many(self, task_ids: List[str]) -> None:
        """Delete multiple tasks by their IDs.

        If some task IDs don't exist, the implementation should silently ignore them.

        Args:
            task_ids: List of task IDs to delete
        """
        pass

    # Iteration and length
    @abstractmethod
    def __iter__(self) -> Iterator[TaskState]:
        """Iterate over all task states.

        The implementation may choose to load tasks in batches internally
        for efficient iteration over large datasets.

        Returns:
            Iterator yielding TaskState objects for all tasks
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of tasks.

        Returns:
            Total count of tasks in the store
        """
        pass

    # Utilities
    @abstractmethod
    def clear(self) -> None:
        """Remove all tasks from the store."""
        pass

    @contextmanager
    def atomic(self) -> Iterator[None]:
        """Context manager for atomic operations on the task store.

        Guarantees that multiple operations performed within the context are:
        1. Atomic: Either all succeed or none do
        2. Isolated: Not visible to other operations until committed

        This method provides a no-op default implementation. Concrete implementations
        should override this method to provide actual transaction support.

        Nested atomic blocks may either:
        1. Be flattened into a single transaction (SQLite-style)
        2. Create nested transactions if the backend supports it
        3. Raise an exception if nested transactions aren't supported

        The implementation should document its behavior with nested calls.

        Example:
            with task_store.atomic():
                task_store.put_many([...])
                task_store.set_many([...])
                # Either all succeed or none do

        Yields:
            None
        """
        try:
            yield
        finally:
            pass

    # Utility functions which can be optionally overriden in a more optimal implementation
    def __getitem__(self, task_id: str) -> TaskState:
        """Retrieve a task state by ID.

        Args:
            task_id: ID of the task to retrieve

        Returns:
            TaskState for the specified task

        Raises:
            KeyError: If the task ID doesn't exist
        """
        states = self.get_many([task_id])
        if not states:
            raise KeyError(f"Task with ID {task_id} not found")
        return states[0]

    def __setitem__(self, task_id: str, state: TaskState) -> None:
        """Update a task state.

        Args:
            task_id: ID of the task to update
            state: New state for the task

        Raises:
            ValueError: If task_id doesn't match state.task.id
        """
        if task_id != state.task.id:
            raise ValueError(f"Task ID mismatch: {task_id} != {state.task.id}")
        self.set_many([state])

    def __contains__(self, task_id: str) -> bool:
        """Check if a task with the given ID exists.

        Args:
            task_id: ID of the task to check

        Returns:
            True if the task exists, False otherwise
        """
        try:
            self[task_id]
            return True
        except KeyError:
            return False

    def put(self, task: TaskDefinition) -> None:
        """Create a new task with PENDING status.

        Args:
            task: Task definition to create

        Raises:
            ValueError: If a task with the same ID already exists (implementation-specific)
        """
        self.put_many([task])

    def delete(self, task_id: str) -> None:
        """Delete a task by ID.

        If the task ID doesn't exist, this operation silently succeeds.

        Args:
            task_id: ID of the task to delete
        """
        self.delete_many([task_id])

    def query(self, status: Optional[TaskStatus] = None) -> Iterator[TaskState]:
        """Yield tasks matching the given status (or all tasks if None).

        Concrete implementations can optimize this method.
        By default, it iterates over all tasks and filters in-memory.

        Args:
            status: Optional status to filter by, or None for all tasks

        Returns:
            Iterator yielding TaskState objects matching the query
        """
        for state in self:
            if status is None or state.status == status:
                yield state

    def reset_many(self, task_ids: List[str]) -> None:
        """Reset specified tasks to PENDING, clearing result and error.

        Args:
            task_ids: List of task IDs to reset

        Raises:
            KeyError: If any task ID doesn't exist (implementation-specific)
        """
        states = self.get_many(task_ids)
        reset_states = [
            TaskState(
                task=state.task,
                status=TaskStatus.PENDING,
                result=None,
                error=None,
            )
            for state in states
        ]
        self.set_many(reset_states)

    def reset_all(self) -> None:
        """Reset all tasks to PENDING, clearing result and error."""
        all_states = list(self)
        reset_states = [
            TaskState(
                task=state.task,
                status=TaskStatus.PENDING,
                result=None,
                error=None,
            )
            for state in all_states
        ]
        self.set_many(reset_states)

    def reset_failed(self) -> None:
        """Reset all failed tasks to PENDING, clearing result and error."""
        failed_states = list(self.query(TaskStatus.FAILED))
        reset_states = [
            TaskState(
                task=state.task,
                status=TaskStatus.PENDING,
                result=None,
                error=None,
            )
            for state in failed_states
        ]
        self.set_many(reset_states)


def get_all_tasks(store: TaskStore) -> List[TaskState]:
    """Get a list of all tasks in the store.

    Args:
        store: TaskStore instance to query

    Returns:
        List of all TaskState objects in the store
    """
    return list(store)


def get_failed_tasks(store: TaskStore) -> List[TaskState]:
    """Get a list of all failed tasks in the store.

    Args:
        store: TaskStore instance to query

    Returns:
        List of TaskState objects with FAILED status
    """
    return list(store.query(TaskStatus.FAILED))


def get_task_data(store: TaskStore, task_id: str) -> Dict[str, Any]:
    """Get the data dictionary for a specific task.

    Args:
        store: TaskStore instance to query
        task_id: ID of the task to retrieve data for

    Returns:
        Data dictionary associated with the task

    Raises:
        KeyError: If the task ID doesn't exist
    """
    return store[task_id].task.data


def get_pending_tasks(store: TaskStore) -> List[str]:
    """Get a list of IDs for all pending tasks.

    Args:
        store: TaskStore instance to query

    Returns:
        List of task IDs with PENDING status
    """
    return [state.task.id for state in store.query(status=TaskStatus.PENDING)]


def get_errors(store: TaskStore) -> Dict[str, str]:
    """Get a dictionary of error messages for all failed tasks.

    Args:
        store: TaskStore instance to query

    Returns:
        Dictionary mapping task IDs to error messages for failed tasks
    """
    return {
        state.task.id: state.error
        for state in store.query(status=TaskStatus.FAILED)
        if state.error
    }


def get_result(store: TaskStore, task_id: str) -> Any:
    """Get the result for a completed task.

    Args:
        store: TaskStore instance to query
        task_id: ID of the completed task

    Returns:
        Result data for the task

    Raises:
        KeyError: If the task ID doesn't exist
        ValueError: If the task is not completed
    """
    state = store[task_id]
    if state.status != TaskStatus.COMPLETED:
        raise ValueError(
            f"Task {task_id} is not completed (status: {state.status.value})"
        )
    return state.result


def get_results(store: TaskStore, task_ids: List[str]) -> Dict[str, Any]:
    """Get results for multiple completed tasks.

    Tasks that aren't completed will be excluded from the results.

    Args:
        store: TaskStore instance to query
        task_ids: List of task IDs to retrieve results for

    Returns:
        Dictionary mapping task IDs to results for completed tasks

    Raises:
        KeyError: If any task ID doesn't exist (implementation-specific)
    """
    states = store.get_many(task_ids)
    return {
        state.task.id: state.result
        for state in states
        if state.status == TaskStatus.COMPLETED
    }


def get_all_results(store: TaskStore) -> Dict[str, Any]:
    """Get results for all completed tasks.

    Args:
        store: TaskStore instance to query

    Returns:
        Dictionary mapping task IDs to results for all completed tasks
    """
    return {
        state.task.id: state.result
        for state in store.query(status=TaskStatus.COMPLETED)
    }


def get_status(store: TaskStore, task_id: str) -> TaskStatus:
    """Get the status of a specific task.

    Args:
        store: TaskStore instance to query
        task_id: ID of the task to check

    Returns:
        TaskStatus enum value for the task

    Raises:
        KeyError: If the task ID doesn't exist
    """
    return store[task_id].status


def get_status_summary(store: TaskStore) -> Dict[TaskStatus, int]:
    """Get a summary of task statuses.

    Args:
        store: TaskStore instance to query

    Returns:
        Dictionary mapping TaskStatus values to counts
    """
    summary = defaultdict(int)
    for state in store:
        summary[state.status] += 1
    return dict(summary)
