import pytest
from kofu.tasks.simple_fn import SimpleFn
from kofu import LocalThreadedExecutor, SQLiteMemory

@pytest.fixture
def sqlite_memory():
    """Fixture to provide a fresh SQLiteMemory instance for each test."""
    return SQLiteMemory(":memory:")  # In-memory SQLite for testing

# Test 1: Basic execution of SimpleFn
def test_simple_fn_execution(sqlite_memory):
    def sample_fn(x, y):
        return x + y

    task = SimpleFn(task_id="task_1", fn=sample_fn, args=(5, 3))

    # Ensure the task ID is correct
    assert task.get_id() == "task_1"

    # Ensure the task executes correctly
    result = task()
    assert result == 8

# Test 2: Using SimpleFn in LocalThreadedExecutor
def test_simple_fn_with_executor(sqlite_memory):
    def multiply_fn(a, b):
        return a * b

    # Create SimpleFn tasks
    tasks = [
        SimpleFn(task_id=f"task_{i}", fn=multiply_fn, args=(i, 2)) for i in range(5)
    ]

    # Run the tasks using LocalThreadedExecutor
    executor = LocalThreadedExecutor(tasks=tasks, memory=sqlite_memory, max_concurrency=2)
    executor.run()

    print("@@", sqlite_memory.dump_all())
    # Check that all tasks completed successfully
    for i in range(5):
        assert sqlite_memory.get_task_status(f"task_{i}") == "completed"
        assert sqlite_memory.get_task_result(f"task_{i}") == i * 2

# Test 3: Handling keyword arguments in SimpleFn
def test_simple_fn_with_kwargs(sqlite_memory):
    def concat_fn(a, b, sep=" "):
        return f"{a}{sep}{b}"

    # Create SimpleFn task with keyword argument
    task = SimpleFn(task_id="task_1", fn=concat_fn, args=("Hello", "World"), kwargs={"sep": ", "})

    # Ensure the task executes correctly
    result = task()
    assert result == "Hello, World"

# Test 4: Using SimpleFn for tasks that raise exceptions
def test_simple_fn_with_error(sqlite_memory):
    def faulty_fn():
        raise ValueError("Something went wrong!")

    # Create a task that raises an error
    task = SimpleFn(task_id="task_1", fn=faulty_fn)

    # Run the task with LocalThreadedExecutor
    executor = LocalThreadedExecutor(tasks=[task], memory=sqlite_memory, max_concurrency=1)
    executor.run()

    # Ensure the task failed
    assert sqlite_memory.get_task_status("task_1") == "failed"
    failed_tasks = sqlite_memory.get_failed_tasks()
    assert len(failed_tasks) == 1
    assert "ValueError" in failed_tasks[0][1]
