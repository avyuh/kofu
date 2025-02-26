import pytest
from kofu import LocalThreadedExecutor  # Assuming this is the file with the executor implementation

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

# Test single task execution
def test_single_task_execution(sqlite_memory):
    task = ExampleTask("task_1", "http://example.com")
    sqlite_memory.store_tasks([("task_1", {"url": "http://example.com"})])

    executor = LocalThreadedExecutor(tasks=[task], memory=sqlite_memory, max_concurrency=1, stop_all_when=always_false)
    
    # Before execution
    assert sqlite_memory.get_task_status("task_1") == "pending"

    # Run the executor
    executor.run()

    # After execution
    assert sqlite_memory.get_task_status("task_1") == "completed"
    assert sqlite_memory.get_task_result("task_1") == "Processed http://example.com"

# Test multiple task execution
def test_multiple_task_execution(sqlite_memory):
    tasks = [
        ExampleTask("task_1", "http://example.com"),
        ExampleTask("task_2", "http://example.org"),
    ]
    sqlite_memory.store_tasks([
        ("task_1", {"url": "http://example.com"}),
        ("task_2", {"url": "http://example.org"}),
    ])

    executor = LocalThreadedExecutor(tasks=tasks, memory=sqlite_memory, max_concurrency=2, stop_all_when=always_false)
    
    # Before execution
    assert sqlite_memory.get_task_status("task_1") == "pending"
    assert sqlite_memory.get_task_status("task_2") == "pending"

    # Run the executor
    executor.run()

    # After execution
    assert sqlite_memory.get_task_status("task_1") == "completed"
    assert sqlite_memory.get_task_status("task_2") == "completed"
    assert sqlite_memory.get_task_result("task_1") == "Processed http://example.com"
    assert sqlite_memory.get_task_result("task_2") == "Processed http://example.org"

# Test skipping already completed tasks
def test_skip_completed_tasks(sqlite_memory):
    tasks = [
        ExampleTask("task_1", "http://example.com"),
        ExampleTask("task_2", "http://example.org"),
    ]
    sqlite_memory.store_tasks([
        ("task_1", {"url": "http://example.com"}),
        ("task_2", {"url": "http://example.org"}),
    ])

    # Mark task_1 as completed
    sqlite_memory.update_task_statuses([("task_1", "completed", {"html": "<html>Processed</html>"}, None)])

    executor = LocalThreadedExecutor(tasks=tasks, memory=sqlite_memory, max_concurrency=2, stop_all_when=always_false)
    
    # Before execution
    assert sqlite_memory.get_task_status("task_1") == "completed"
    assert sqlite_memory.get_task_status("task_2") == "pending"

    # Run the executor
    executor.run()

    # Ensure task_1 was skipped and task_2 was completed
    assert sqlite_memory.get_task_status("task_1") == "completed"  # Should remain completed
    assert sqlite_memory.get_task_status("task_2") == "completed"  # Should be marked as completed now
    assert sqlite_memory.get_task_result("task_2") == "Processed http://example.org"
