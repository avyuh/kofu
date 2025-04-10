import pytest
from kofu import (
    LocalThreadedExecutor,
)  # Assuming this is the file with the executor implementation


# Task definition for testing
class ExampleTaskWithException:
    def __init__(self, task_id, url, should_fail=False):
        self.task_id = task_id
        self.url = url
        self.should_fail = should_fail

    def get_id(self):
        return self.task_id

    def __call__(self):
        if self.should_fail:
            raise Exception(f"Task {self.task_id} failed")
        return f"Processed {self.url}"


# Fixture to provide a fresh SQLiteMemory for each test
@pytest.fixture
def sqlite_memory():
    from kofu.memory import SQLiteMemory

    return SQLiteMemory(":memory:")  # Use in-memory SQLite for testing


# Test task execution with exceptions (error handling)
def test_task_execution_with_exceptions(sqlite_memory):
    tasks = [
        ExampleTaskWithException("task_1", "http://example.com", should_fail=True),
        ExampleTaskWithException("task_2", "http://example.org", should_fail=False),
    ]
    sqlite_memory.store_tasks(
        [
            ("task_1", {"url": "http://example.com"}),
            ("task_2", {"url": "http://example.org"}),
        ]
    )

    executor = LocalThreadedExecutor(
        tasks=tasks, memory=sqlite_memory, max_concurrency=2
    )

    # Run the executor
    executor.run()

    # Check task_1 failed and task_2 succeeded
    assert sqlite_memory.get_task_status("task_1") == "failed"
    assert sqlite_memory.get_failed_tasks() == [
        ("task_1", "Exception: Task task_1 failed")
    ]
    assert sqlite_memory.get_task_status("task_2") == "completed"
    assert sqlite_memory.get_task_result("task_2") == "Processed http://example.org"


# Test status summary after task execution
def test_status_summary_after_execution(sqlite_memory, capsys):
    tasks = [
        ExampleTaskWithException("task_1", "http://example.com", should_fail=False),
        ExampleTaskWithException("task_2", "http://example.org", should_fail=True),
        ExampleTaskWithException("task_3", "http://example.net", should_fail=False),
    ]
    sqlite_memory.store_tasks(
        [
            ("task_1", {"url": "http://example.com"}),
            ("task_2", {"url": "http://example.org"}),
            ("task_3", {"url": "http://example.net"}),
        ]
    )

    executor = LocalThreadedExecutor(
        tasks=tasks, memory=sqlite_memory, max_concurrency=2
    )

    # Run the executor
    executor.run()

    # Capture the status summary output
    captured = capsys.readouterr()

    # Check that the summary contains the correct information
    assert "Pending tasks: 0" in captured.out
    assert "Completed tasks: 2" in captured.out
    assert "Failed tasks: 1" in captured.out


# Test failed tasks are retried (if applicable)
def test_failed_tasks_are_retried(sqlite_memory):
    execution_count = {}

    # Define a task class that tracks the number of execution attempts
    class ExampleTaskWithRetry(ExampleTaskWithException):
        def __call__(self):
            if self.task_id not in execution_count:
                execution_count[self.task_id] = 0
            execution_count[self.task_id] += 1

            # Fail the task on the first attempt, succeed on the second
            if execution_count[self.task_id] == 1:
                raise Exception(f"Task {self.task_id} failed on first attempt")
            return f"Processed {self.url} on retry"

    tasks = [ExampleTaskWithRetry("task_1", "http://example.com", should_fail=True)]
    sqlite_memory.store_tasks([("task_1", {"url": "http://example.com"})])

    executor = LocalThreadedExecutor(
        tasks=tasks, memory=sqlite_memory, max_concurrency=1
    )

    # Run the executor
    executor.run()

    # Ensure the task was retried and succeeded on the second attempt
    assert (
        execution_count["task_1"] == 2
    )  # Task should be executed twice (once failed, once succeeded)
    assert sqlite_memory.get_task_status("task_1") == "completed"
    assert (
        sqlite_memory.get_task_result("task_1")
        == "Processed http://example.com on retry"
    )
