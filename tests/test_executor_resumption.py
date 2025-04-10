import pytest
from kofu import (
    LocalThreadedExecutor,
)  # Assuming this is the file with the executor implementation


# Task definition for testing
class ExampleTask:
    def __init__(self, task_id, url):
        self.task_id = task_id
        self.url = url

    def get_id(self):
        return self.task_id

    def __call__(self):
        return f"Processed {self.url}"


# Mock stop condition (always returns False for now)
def always_false():
    return False


# Fixture to provide a fresh SQLiteMemory for each test
@pytest.fixture
def sqlite_memory():
    from kofu.memory import SQLiteMemory

    return SQLiteMemory(":memory:")  # Use in-memory SQLite for testing


# Test resumption after some tasks are completed
def test_resumption_after_some_tasks_completed(sqlite_memory):
    tasks = [
        ExampleTask("task_1", "http://example.com"),
        ExampleTask("task_2", "http://example.org"),
        ExampleTask("task_3", "http://example.net"),
    ]
    sqlite_memory.store_tasks(
        [
            ("task_1", {"url": "http://example.com"}),
            ("task_2", {"url": "http://example.org"}),
            ("task_3", {"url": "http://example.net"}),
        ]
    )

    # Simulate task_1 and task_2 as completed
    sqlite_memory.update_task_statuses(
        [
            ("task_1", "completed", {"html": "<html>Processed</html>"}, None),
            ("task_2", "completed", {"html": "<html>Processed</html>"}, None),
        ]
    )

    executor = LocalThreadedExecutor(
        tasks=tasks, memory=sqlite_memory, max_concurrency=2, stop_all_when=always_false
    )

    # Run the executor (should only execute task_3)
    executor.run()

    # Verify that task_1 and task_2 were skipped and task_3 was executed
    assert (
        sqlite_memory.get_task_status("task_1") == "completed"
    )  # Should remain completed
    assert (
        sqlite_memory.get_task_status("task_2") == "completed"
    )  # Should remain completed
    assert (
        sqlite_memory.get_task_status("task_3") == "completed"
    )  # Should now be marked as completed
    assert sqlite_memory.get_task_result("task_3") == "Processed http://example.net"


# Test resumption after crash (interrupted execution)
def test_resumption_after_incomplete_execution(sqlite_memory):
    tasks = [
        ExampleTask("task_1", "http://example.com"),
        ExampleTask("task_2", "http://example.org"),
        ExampleTask("task_3", "http://example.net"),
    ]
    sqlite_memory.store_tasks(
        [
            ("task_1", {"url": "http://example.com"}),
            ("task_2", {"url": "http://example.org"}),
            ("task_3", {"url": "http://example.net"}),
        ]
    )

    # Simulate incomplete execution by marking task_1 as completed
    sqlite_memory.update_task_statuses(
        [("task_1", "completed", {"html": "<html>Processed</html>"}, None)]
    )

    # Create and run the executor (it should execute task_2 and task_3, as task_1 is already completed)
    executor = LocalThreadedExecutor(
        tasks=tasks, memory=sqlite_memory, max_concurrency=2, stop_all_when=always_false
    )
    executor.run()

    # Ensure task_1 was skipped, and task_2 and task_3 were executed
    assert sqlite_memory.get_task_status("task_1") == "completed"
    assert sqlite_memory.get_task_status("task_2") == "completed"
    assert sqlite_memory.get_task_status("task_3") == "completed"


# Test memory persistence for task statuses (pending, completed, failed)
def test_memory_persistence_for_task_statuses(sqlite_memory):
    tasks = [
        ExampleTask("task_1", "http://example.com"),
        ExampleTask("task_2", "http://example.org"),
    ]
    sqlite_memory.store_tasks(
        [
            ("task_1", {"url": "http://example.com"}),
            ("task_2", {"url": "http://example.org"}),
        ]
    )

    executor = LocalThreadedExecutor(
        tasks=tasks, memory=sqlite_memory, max_concurrency=2, stop_all_when=always_false
    )

    # Before execution
    assert sqlite_memory.get_task_status("task_1") == "pending"
    assert sqlite_memory.get_task_status("task_2") == "pending"

    # Run the executor
    executor.run()

    # After execution, both tasks should be marked as completed
    assert sqlite_memory.get_task_status("task_1") == "completed"
    assert sqlite_memory.get_task_status("task_2") == "completed"

    # Mark task_1 as failed and test persistence of failed status
    sqlite_memory.update_task_statuses([("task_1", "failed", None, "TimeoutError")])
    assert sqlite_memory.get_task_status("task_1") == "failed"
    assert sqlite_memory.get_failed_tasks() == [("task_1", "TimeoutError")]
