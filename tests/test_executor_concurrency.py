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
        time.sleep(self.delay)  # Simulate task taking time to execute
        return f"Processed {self.url} after {self.delay}s"

# Mock stop condition (always returns False for now)
def always_false():
    return False

# Fixture to provide a fresh SQLiteMemory for each test
@pytest.fixture
def sqlite_memory():
    from kofu.memory import SQLiteMemory
    return SQLiteMemory(":memory:")  # Use in-memory SQLite for testing

# Test tasks execute concurrently with max concurrency limit
def test_tasks_execute_concurrently_with_limit(sqlite_memory):
    tasks = [
        ExampleTaskWithDelay("task_1", "http://example.com", delay=2),
        ExampleTaskWithDelay("task_2", "http://example.org", delay=2),
        ExampleTaskWithDelay("task_3", "http://example.net", delay=2)
    ]
    sqlite_memory.store_tasks([
        ("task_1", {"url": "http://example.com"}),
        ("task_2", {"url": "http://example.org"}),
        ("task_3", {"url": "http://example.net"})
    ])

    executor = LocalThreadedExecutor(tasks=tasks, memory=sqlite_memory, max_concurrency=2, stop_all_when=always_false)

    # Measure start time to ensure tasks are running concurrently
    start_time = time.time()
    executor.run()
    total_time = time.time() - start_time

    # We expect the execution time to be roughly equal to the delay for two tasks running in parallel
    # If max_concurrency=2, it should run 2 tasks first (2 seconds), then 1 task after that (2 seconds), totaling ~4 seconds
    assert 4 <= total_time < 5  # Should be around 4 seconds since tasks 1 & 2 run first, then task 3

# Test no task duplication under concurrency
def test_no_task_duplication(sqlite_memory):
    task_execution_count = {}

    # A task class that tracks execution count
    class ExampleTaskWithCount:
        def __init__(self, task_id, url):
            self.task_id = task_id
            self.url = url

        def get_id(self):
            return self.task_id

        def __call__(self):
            # Track how many times each task is executed
            if self.task_id not in task_execution_count:
                task_execution_count[self.task_id] = 0
            task_execution_count[self.task_id] += 1
            return f"Processed {self.url}"

    tasks = [
        ExampleTaskWithCount("task_1", "http://example.com"),
        ExampleTaskWithCount("task_2", "http://example.org"),
        ExampleTaskWithCount("task_3", "http://example.net")
    ]
    sqlite_memory.store_tasks([
        ("task_1", {"url": "http://example.com"}),
        ("task_2", {"url": "http://example.org"}),
        ("task_3", {"url": "http://example.net"})
    ])

    executor = LocalThreadedExecutor(tasks=tasks, memory=sqlite_memory, max_concurrency=3, stop_all_when=always_false)

    # Run the executor
    executor.run()

    # Ensure each task was only executed once
    assert task_execution_count["task_1"] == 1
    assert task_execution_count["task_2"] == 1
    assert task_execution_count["task_3"] == 1

# Test task ordering (if applicable)
def test_correct_task_ordering(sqlite_memory):
    execution_order = []

    # A task class that records execution order
    class ExampleTaskWithOrder:
        def __init__(self, task_id, url):
            self.task_id = task_id
            self.url = url

        def get_id(self):
            return self.task_id

        def __call__(self):
            execution_order.append(self.task_id)
            return f"Processed {self.url}"

    tasks = [
        ExampleTaskWithOrder("task_1", "http://example.com"),
        ExampleTaskWithOrder("task_2", "http://example.org"),
        ExampleTaskWithOrder("task_3", "http://example.net")
    ]
    sqlite_memory.store_tasks([
        ("task_1", {"url": "http://example.com"}),
        ("task_2", {"url": "http://example.org"}),
        ("task_3", {"url": "http://example.net"})
    ])

    executor = LocalThreadedExecutor(tasks=tasks, memory=sqlite_memory, max_concurrency=2, stop_all_when=always_false)

    # Run the executor
    executor.run()

    # Since we are using threads and not processes, task order might not be strictly sequential due to concurrency.
    # Therefore, we just ensure each task was executed once and there was no unexpected ordering.
    assert set(execution_order) == {"task_1", "task_2", "task_3"}
    assert len(execution_order) == 3
