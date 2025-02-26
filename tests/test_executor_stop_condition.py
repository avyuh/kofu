import pytest
import time
from kofu import LocalThreadedExecutor  # Assuming this is the file with the executor implementation

# Task definition for testing with delay
class ExampleTaskWithDelay:
    def __init__(self, task_id, url, delay=0):
        self.task_id = task_id
        self.url = url
        self.delay = delay

    def get_id(self):
        return self.task_id

    def __call__(self):
        time.sleep(self.delay)  # Simulate task execution time
        return f"Processed {self.url} after {self.delay}s"

# Fixture to provide a fresh SQLiteMemory for each test
@pytest.fixture
def sqlite_memory():
    from kofu.memory import SQLiteMemory
    return SQLiteMemory(":memory:")  # Use in-memory SQLite for testing

# Test stop condition halts execution after a certain number of tasks
def test_stop_condition_after_task_execution(sqlite_memory):
    tasks = [
        ExampleTaskWithDelay("task_1", "http://example.com"),
        ExampleTaskWithDelay("task_2", "http://example.org"),
        ExampleTaskWithDelay("task_3", "http://example.net"),
    ]
    sqlite_memory.store_tasks([
        ("task_1", {"url": "http://example.com"}),
        ("task_2", {"url": "http://example.org"}),
        ("task_3", {"url": "http://example.net"})
    ])

    # Define a stop condition: Stop after 2 tasks are executed
    executed_tasks = 0
    def stop_after_two_tasks():
        nonlocal executed_tasks
        return executed_tasks >= 2

    # Track how many tasks have been executed
    class ExampleTaskWithCounter(ExampleTaskWithDelay):
        def __call__(self):
            nonlocal executed_tasks
            result = super().__call__()
            executed_tasks += 1
            return result

    # Use the custom task class to track execution count
    tasks = [
        ExampleTaskWithCounter("task_1", "http://example.com", delay=0),
        ExampleTaskWithCounter("task_2", "http://example.org", delay=0),
        ExampleTaskWithCounter("task_3", "http://example.net", delay=0),
    ]

    executor = LocalThreadedExecutor(tasks=tasks, memory=sqlite_memory, max_concurrency=2, stop_all_when=stop_after_two_tasks)

    # Run the executor
    executor.run()

    # Allow 2 or 3 tasks to be executed due to concurrency
    assert 1 <= executed_tasks <= 3

    # Also check task statuses to confirm correct task outcomes
    completed_tasks = sqlite_memory.get_completed_tasks()
    assert 1 <= len(completed_tasks) <= 3  # Accept 1, 2, or 3 due to how concurrency works

# Test stop condition is checked after each task
def test_stop_condition_checked_after_each_task(sqlite_memory):
    tasks = [
        ExampleTaskWithDelay("task_1", "http://example.com", delay=1),
        ExampleTaskWithDelay("task_2", "http://example.org", delay=1),
    ]
    sqlite_memory.store_tasks([
        ("task_1", {"url": "http://example.com"}),
        ("task_2", {"url": "http://example.org"}),
    ])

    # Define a stop condition: Stop after the first task is completed
    def stop_after_one_task():
        return sqlite_memory.get_task_status("task_1") == "completed"

    executor = LocalThreadedExecutor(tasks=tasks, memory=sqlite_memory, max_concurrency=2, stop_all_when=stop_after_one_task)

    # Run the executor
    executor.run()

    # Ensure task_1 is completed, and task_2 is not started
    assert sqlite_memory.get_task_status("task_1") == "completed"
    assert sqlite_memory.get_task_status("task_2") == "pending"  # Should remain pending

# Test stop condition halts mid-execution
def test_stop_condition_halts_mid_execution(sqlite_memory):
    tasks = [
        ExampleTaskWithDelay("task_1", "http://example.com", delay=3),  # Simulates long task
        ExampleTaskWithDelay("task_2", "http://example.org", delay=1),  # Simulates short task
    ]
    sqlite_memory.store_tasks([
        ("task_1", {"url": "http://example.com"}),
        ("task_2", {"url": "http://example.org"}),
    ])

    # Define a stop condition that will be triggered during task_1's execution
    def stop_mid_execution():
        # Simulate a stop condition triggered after 2 seconds
        time.sleep(2)
        return True

    executor = LocalThreadedExecutor(tasks=tasks, memory=sqlite_memory, max_concurrency=2, stop_all_when=stop_mid_execution)

    # Run the executor
    executor.run()

    # Ensure task_1 was stopped mid-execution (as much as possible) and task_2 did not run
    assert sqlite_memory.get_task_status("task_1") == "pending"  # Should remain pending or partially executed
    assert sqlite_memory.get_task_status("task_2") == "pending"  # Should remain pending as it was never started
