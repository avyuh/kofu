import pytest
import threading
from kofu.memory import SQLiteMemory  # Assuming the SQLiteMemory implementation is in sqlite_memory.py

# Test setup: fixture to create a fresh SQLiteMemory instance
@pytest.fixture
def sqlite_memory():
    return SQLiteMemory(":memory:")  # Use in-memory SQLite for testing

# Basic CRUD Operations

def test_store_task(sqlite_memory):
    tasks = [("task_1", {"url": "http://example.com"})]
    sqlite_memory.store_tasks(tasks)
    assert sqlite_memory.get_task_status("task_1") == "pending"

def test_update_task_status_completed(sqlite_memory):
    tasks = [("task_1", {"url": "http://example.com"})]
    sqlite_memory.store_tasks(tasks)
    sqlite_memory.update_task_statuses([("task_1", "completed", {"html": "<html>...</html>"}, None)])
    assert sqlite_memory.get_task_status("task_1") == "completed"
    assert sqlite_memory.get_task_result("task_1") == {"html": "<html>...</html>"}

def test_update_task_status_failed(sqlite_memory):
    tasks = [("task_1", {"url": "http://example.com"})]
    sqlite_memory.store_tasks(tasks)
    sqlite_memory.update_task_statuses([("task_1", "failed", None, "TimeoutError")])
    assert sqlite_memory.get_task_status("task_1") == "failed"
    failed_tasks = sqlite_memory.get_failed_tasks()
    assert failed_tasks == [("task_1", "TimeoutError")]

def test_get_pending_tasks(sqlite_memory):
    tasks = [
        ("task_1", {"url": "http://example.com"}),
        ("task_2", {"url": "http://example.org"})
    ]
    sqlite_memory.store_tasks(tasks)
    sqlite_memory.update_task_statuses([("task_1", "completed", {"html": "<html>...</html>"}, None)])
    pending_tasks = sqlite_memory.get_pending_tasks()
    assert pending_tasks == ["task_2"]

def test_get_completed_tasks(sqlite_memory):
    tasks = [
        ("task_1", {"url": "http://example.com"}),
        ("task_2", {"url": "http://example.org"})
    ]
    sqlite_memory.store_tasks(tasks)
    sqlite_memory.update_task_statuses([("task_1", "completed", {"html": "<html>...</html>"}, None)])
    completed_tasks = sqlite_memory.get_completed_tasks()
    assert completed_tasks == ["task_1"]

# Batch Operations

def test_store_multiple_tasks(sqlite_memory):
    tasks = [
        ("task_1", {"url": "http://example.com"}),
        ("task_2", {"url": "http://example.org"})
    ]
    sqlite_memory.store_tasks(tasks)
    assert sqlite_memory.get_task_status("task_1") == "pending"
    assert sqlite_memory.get_task_status("task_2") == "pending"

def test_update_multiple_task_statuses(sqlite_memory):
    tasks = [
        ("task_1", {"url": "http://example.com"}),
        ("task_2", {"url": "http://example.org"})
    ]
    sqlite_memory.store_tasks(tasks)
    updates = [
        ("task_1", "completed", {"html": "<html>Task 1</html>"}, None),
        ("task_2", "failed", None, "TimeoutError")
    ]
    sqlite_memory.update_task_statuses(updates)
    assert sqlite_memory.get_task_status("task_1") == "completed"
    assert sqlite_memory.get_task_status("task_2") == "failed"
    assert sqlite_memory.get_task_result("task_1") == {"html": "<html>Task 1</html>"}
    assert sqlite_memory.get_failed_tasks() == [("task_2", "TimeoutError")]

# Concurrency Tests

def test_concurrent_task_storage(sqlite_memory):
    def store_task(task_id):
        sqlite_memory.store_tasks([(task_id, {"url": f"http://example{task_id}.com"})])

    threads = []
    for i in range(10):
        t = threading.Thread(target=store_task, args=(f"task_{i}",))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    for i in range(10):
        assert sqlite_memory.get_task_status(f"task_{i}") == "pending"

def test_concurrent_task_status_updates(sqlite_memory):
    tasks = [(f"task_{i}", {"url": f"http://example{i}.com"}) for i in range(10)]
    sqlite_memory.store_tasks(tasks)

    def update_task_status(task_id, status):
        sqlite_memory.update_task_statuses([(task_id, status, {"html": f"<html>{task_id}</html>"}, None)])

    threads = []
    for i in range(10):
        t = threading.Thread(target=update_task_status, args=(f"task_{i}", "completed"))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    for i in range(10):
        assert sqlite_memory.get_task_status(f"task_{i}") == "completed"
        assert sqlite_memory.get_task_result(f"task_{i}") == {"html": f"<html>task_{i}</html>"}

# Error Handling and Recovery

def test_update_failed_task_and_retry(sqlite_memory):
    tasks = [("task_1", {"url": "http://example.com"})]
    sqlite_memory.store_tasks(tasks)
    sqlite_memory.update_task_statuses([("task_1", "failed", None, "TimeoutError")])
    sqlite_memory.update_task_statuses([("task_1", "completed", {"html": "<html>...</html>"}, None)])
    assert sqlite_memory.get_task_status("task_1") == "completed"
    assert sqlite_memory.get_task_result("task_1") == {"html": "<html>...</html>"}

# Selective and Full Data Deletion

def test_clear_all_tasks(sqlite_memory):
    tasks = [
        ("task_1", {"url": "http://example.com"}),
        ("task_2", {"url": "http://example.org"})
    ]
    sqlite_memory.store_tasks(tasks)
    sqlite_memory.clear()
    assert sqlite_memory.get_pending_tasks() == []

def test_clear_specific_tasks(sqlite_memory):
    tasks = [
        ("task_1", {"url": "http://example.com"}),
        ("task_2", {"url": "http://example.org"})
    ]
    sqlite_memory.store_tasks(tasks)
    sqlite_memory.clear_tasks(["task_1"])
    assert sqlite_memory.get_task_status("task_2") == "pending"
    with pytest.raises(KeyError):  # Task 1 should be cleared
        sqlite_memory.get_task_status("task_1")

# Data Dumping

def test_dump_all(sqlite_memory):
    tasks = [
        ("task_1", {"url": "http://example.com"}),
        ("task_2", {"url": "http://example.org"})
    ]
    sqlite_memory.store_tasks(tasks)
    sqlite_memory.update_task_statuses([("task_1", "completed", {"html": "<html>Task 1</html>"}, None)])
    sqlite_memory.update_task_statuses([("task_2", "failed", None, "TimeoutError")])

    dump = sqlite_memory.dump_all()

    assert "task_definitions" in dump
    assert "task_statuses" in dump
    assert "task_results" in dump
    assert "task_errors" in dump

    assert dump["task_definitions"] == {
        "task_1": {"url": "http://example.com"},
        "task_2": {"url": "http://example.org"}
    }

    assert dump["task_statuses"] == {
        "task_1": "completed",
        "task_2": "failed"
    }

    assert dump["task_results"] == {
        "task_1": {"html": "<html>Task 1</html>"}
    }

    assert dump["task_errors"] == {
        "task_2": "TimeoutError"
    }

# Edge Cases

def test_store_task_with_invalid_data(sqlite_memory):
    with pytest.raises(TypeError):
        sqlite_memory.store_tasks([("task_1", None)])  # Invalid task data

def test_update_nonexistent_task(sqlite_memory):
    with pytest.raises(KeyError):
        sqlite_memory.update_task_statuses([("task_999", "completed", {"html": "<html>...</html>"}, None)])

def test_clear_nonexistent_tasks(sqlite_memory):
    tasks = [("task_1", {"url": "http://example.com"})]
    sqlite_memory.store_tasks(tasks)
    sqlite_memory.clear_tasks(["task_999"])  # Non-existent task should not cause issues
    assert sqlite_memory.get_task_status("task_1") == "pending"

def test_get_task_status_not_found(sqlite_memory):
    with pytest.raises(KeyError, match="Task with ID task_999 not found"):
        sqlite_memory.get_task_status("task_999")

def test_store_task_with_invalid_data(sqlite_memory):
    with pytest.raises(TypeError, match="Task data for task task_1 is not JSON serializable"):
        sqlite_memory.store_tasks([("task_1", set([1, 2, 3]))])  # Sets are not JSON serializable

def test_auto_create_directories_for_sqlite_memory(tmp_path):
    # Prepare a directory path inside tmp_path that doesn't exist yet
    non_existent_dir = tmp_path / "non_existent_dir"
    sqlite_file_path = non_existent_dir / "test_tasks.db"

    # Ensure that the directory does not exist before initializing SQLiteMemory
    assert not non_existent_dir.exists()

    # Initialize SQLiteMemory with the non-existent directory path
    memory = SQLiteMemory(path=str(sqlite_file_path))

    # Check that the directory was automatically created
    assert non_existent_dir.exists()

    # Optional: Verify that we can now interact with SQLiteMemory (store a task)
    memory.store_tasks([("task_1", {"data": "test"})])
    assert memory.get_task_status("task_1") == "pending"
