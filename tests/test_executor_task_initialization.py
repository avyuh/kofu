import pytest
from kofu import LocalThreadedExecutor, SQLiteMemory


# Mock task for testing
class MockTask:
    def __init__(self, task_id, should_fail=False):
        self.task_id = task_id
        self.should_fail = should_fail

    def get_id(self):
        return self.task_id

    def __call__(self):
        if self.should_fail:
            raise Exception(f"Task {self.task_id} failed")
        return f"Processed {self.task_id}"


@pytest.fixture
def sqlite_memory():
    """Fixture to provide a fresh SQLiteMemory instance for each test."""
    return SQLiteMemory(":memory:")  # In-memory SQLite for testing


# Test Case 1: Tasks are initialized in memory if not present
def test_tasks_initialized_in_memory(sqlite_memory):
    tasks = [MockTask(f"task_{i}") for i in range(3)]

    # Create the executor with tasks not present in memory
    executor = LocalThreadedExecutor(
        tasks=tasks, memory=sqlite_memory, max_concurrency=2
    )

    # Initially, the memory should have no task definitions
    assert sqlite_memory.get_pending_tasks() == []

    # Run the executor
    executor.run()

    # Ensure that the tasks are now stored in memory
    pending_tasks = sqlite_memory.get_pending_tasks()
    assert len(pending_tasks) == 0  # No pending tasks after execution
    completed_tasks = sqlite_memory.get_completed_tasks()
    assert len(completed_tasks) == 3  # All tasks should be marked as completed


# Test Case 2: Tasks that are already defined in memory should not be overwritten
def test_tasks_not_overwritten(sqlite_memory):
    tasks = [MockTask(f"task_{i}") for i in range(3)]

    # Store task definitions in memory, marking task_0 as completed
    sqlite_memory.store_tasks([(task.get_id(), {}) for task in tasks])
    sqlite_memory.update_task_statuses([("task_0", "completed", None, None)])

    # Create the executor
    executor = LocalThreadedExecutor(
        tasks=tasks, memory=sqlite_memory, max_concurrency=2
    )

    # Run the executor
    executor.run()

    # Ensure task_0 is still marked as completed and wasn't overwritten
    assert sqlite_memory.get_task_status("task_0") == "completed"

    # Ensure task_1 and task_2 were executed
    assert sqlite_memory.get_task_status("task_1") == "completed"
    assert sqlite_memory.get_task_status("task_2") == "completed"


# Test Case 3: All tasks are completed when memory is initially empty
def test_all_tasks_completed(sqlite_memory):
    tasks = [MockTask(f"task_{i}") for i in range(3)]

    # Create the executor
    executor = LocalThreadedExecutor(
        tasks=tasks, memory=sqlite_memory, max_concurrency=2
    )

    # Run the executor
    executor.run()

    # Ensure that all tasks are marked as completed
    assert len(sqlite_memory.get_completed_tasks()) == 3


# Test Case: LocalThreadedExecutor uses SQLiteMemory by default when no memory is provided
def test_local_threaded_executor_uses_sqlite_memory_by_default(tmp_path):
    # Create tasks
    tasks = [MockTask(f"task_{i}") for i in range(3)]

    # Define a path for SQLiteMemory
    sqlite_path = tmp_path / "test_tasks.db"

    # Ensure no explicit memory is provided, only the path
    executor = LocalThreadedExecutor(
        tasks=tasks, path=str(sqlite_path), max_concurrency=2
    )

    # Run the executor
    executor.run()

    # Check that tasks are completed and persisted in SQLite
    assert executor.memory.get_task_status("task_0") == "completed"
    assert executor.memory.get_task_status("task_1") == "completed"
    assert executor.memory.get_task_status("task_2") == "completed"


# Test Case: LocalThreadedExecutor raises an error when neither path nor memory is provided
def test_local_threaded_executor_raises_error_without_path_or_memory():
    # Create tasks
    tasks = [MockTask(f"task_{i}") for i in range(3)]

    # Expect an error when neither memory nor path is provided
    with pytest.raises(
        ValueError, match="Either a memory instance or a path must be provided"
    ):
        LocalThreadedExecutor(tasks=tasks, max_concurrency=2)


# Test Case: LocalThreadedExecutor works with custom memory (SQLiteMemory provided)
def test_local_threaded_executor_works_with_custom_memory():
    # Create tasks
    tasks = [MockTask(f"task_{i}") for i in range(3)]

    # Use custom SQLiteMemory
    memory = SQLiteMemory(":memory:")  # In-memory SQLite for testing

    # Ensure custom memory is used instead of default
    executor = LocalThreadedExecutor(tasks=tasks, memory=memory, max_concurrency=2)

    # Run the executor
    executor.run()

    # Check that tasks are completed and stored in the custom memory
    assert memory.get_task_status("task_0") == "completed"
    assert memory.get_task_status("task_1") == "completed"
    assert memory.get_task_status("task_2") == "completed"
