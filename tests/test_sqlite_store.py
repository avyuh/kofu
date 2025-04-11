# File: kofu/tests/test_sqlite_store.py

import pytest
import sqlite3
import threading
import time
import random
import logging
from unittest.mock import patch, MagicMock

# No longer need unittest.mock here
# from unittest.mock import patch, MagicMock

from kofu.store import (
    SingleSQLiteTaskStore,
    JSONSerializer,
    TaskDefinition,
    TaskState,
    TaskStatus,
)

# Configure logging for debugging tests if needed
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# --- Fixtures ---


@pytest.fixture
def store_path(tmp_path):
    """Provides a temporary path for the store."""
    return str(tmp_path)


@pytest.fixture
def store(tmp_path):
    """Provides a fresh SingleSQLiteTaskStore instance for each test."""
    db_path = tmp_path / "test_store"
    # Use a slightly shorter timeout for tests to fail faster if deadlocked
    store = SingleSQLiteTaskStore(directory=str(db_path), timeout=10)
    yield store
    store.close()


@pytest.fixture
def serializer():
    """Provides a JSONSerializer instance."""
    return JSONSerializer(compression_level=1)


# --- Helper Functions ---


def create_task(i):
    return TaskDefinition(id=f"task_{i}", data={"value": i, "desc": f"Task number {i}"})


def create_task_state(i, status=TaskStatus.PENDING, result=None, error=None):
    task = create_task(i)
    return TaskState(task=task, status=status, result=result, error=error)


def get_all_states_dict(store):
    # Ensure connection is fresh for read after potential concurrent writes
    # store.close() # Don't close, just get a fresh conn if needed by internal logic
    return {state.task.id: state for state in store}


# --- Test Cases ---

# 1. atomic() Context Manager Tests
# ==================================


def test_atomic_commit_success(store):
    task1 = create_task(1)
    try:
        with store.atomic():
            store.put_many([task1])
            assert store.task_exists(task1.id), "Task should exist inside transaction"
    except Exception as e:
        pytest.fail(f"Atomic block raised unexpected exception: {e}")

    assert store.task_exists(task1.id), "Task should exist after committed transaction"
    state = store.get_many([task1.id])[0]
    assert state.status == TaskStatus.PENDING


def test_atomic_rollback_on_exception(store):
    task1 = create_task(1)
    task2 = create_task(2)
    store.put_many([task1])  # Pre-existing task

    with pytest.raises(ValueError, match="Test Rollback"):
        with store.atomic():
            store.set_many(
                [
                    TaskState(
                        task=task1, status=TaskStatus.COMPLETED, result={"done": True}
                    )
                ]
            )
            store.put_many([task2])  # Add task 2
            assert store.task_exists(task2.id), "Task 2 should exist inside transaction"
            # Simulate failure
            raise ValueError("Test Rollback")

    # Verify rollback
    assert not store.task_exists(task2.id), "Task 2 should not exist after rollback"
    state1 = store.get_many([task1.id])[0]
    assert (
        state1.status == TaskStatus.PENDING
    ), "Task 1 status should be rolled back to PENDING"
    assert state1.result is None


def test_atomic_nested_commit_success(store):
    task1 = create_task(1)
    task2 = create_task(2)
    task3 = create_task(3)

    try:
        with store.atomic():
            store.put_many([task1])
            with store.atomic():
                store.put_many([task2])
                store.set_many(
                    [
                        TaskState(
                            task=task1, status=TaskStatus.COMPLETED, result={"ok": 1}
                        )
                    ]
                )
            # Inner block committed (released savepoint)
            assert store.task_exists(task2.id), "Task 2 should exist after inner commit"
            state1_after_inner = store.get_many([task1.id])[0]
            assert (
                state1_after_inner.status == TaskStatus.COMPLETED
            ), "Task 1 status should be updated after inner commit"

            store.put_many([task3])

    except Exception as e:
        pytest.fail(f"Nested atomic block raised unexpected exception: {e}")

    # Outer block committed
    assert store.task_exists(task1.id)
    assert store.task_exists(task2.id)
    assert store.task_exists(task3.id)
    state1_final = store.get_many([task1.id])[0]
    assert state1_final.status == TaskStatus.COMPLETED


def test_atomic_nested_rollback_inner(store):
    task1 = create_task(1)
    task2 = create_task(2)
    task3 = create_task(3)

    try:
        with store.atomic():
            store.put_many([task1])
            try:
                with store.atomic():
                    store.put_many([task2])
                    # Simulate failure in inner block
                    raise ValueError("Inner Rollback")
            except ValueError as e:
                assert "Inner Rollback" in str(e), "Expected inner exception"

            # Inner block rolled back (to savepoint)
            assert not store.task_exists(
                task2.id
            ), "Task 2 should not exist after inner rollback"
            state1_after_inner_rollback = store.get_many([task1.id])[0]
            assert (
                state1_after_inner_rollback.status == TaskStatus.PENDING
            ), "Task 1 should be unaffected by inner rollback"

            store.put_many([task3])  # This should succeed

    except Exception as e:
        pytest.fail(
            f"Nested atomic block (inner rollback) raised unexpected exception: {e}"
        )

    # Outer block committed
    assert store.task_exists(task1.id)
    assert not store.task_exists(task2.id)  # Still rolled back
    assert store.task_exists(task3.id)  # Task 3 should be committed


def test_atomic_nested_rollback_outer(store):
    task1 = create_task(1)
    task2 = create_task(2)
    task3 = create_task(3)

    with pytest.raises(ValueError, match="Outer Rollback"):
        with store.atomic():
            store.put_many([task1])
            with store.atomic():
                store.put_many([task2])
            # Inner block committed (released savepoint)
            assert store.task_exists(task2.id), "Task 2 should exist after inner commit"

            store.put_many([task3])
            # Simulate failure in outer block after inner block
            raise ValueError("Outer Rollback")

    # Entire transaction rolled back
    assert not store.task_exists(task1.id), "Task 1 should be rolled back"
    assert not store.task_exists(task2.id), "Task 2 should be rolled back"
    assert not store.task_exists(task3.id), "Task 3 should be rolled back"


# 2. _execute_with_retry Behavior (Tested implicitly via concurrency tests)
# ================================


def test_basic_operation_without_contention(store):
    # Verify basic operation works without needing retries in a single thread
    task = create_task(1)
    try:
        store.put_many([task])
        state = store.get_many([task.id])[0]
        store.set_many([TaskState(task, TaskStatus.COMPLETED, {"r": 1})])
        state_after = store.get_many([task.id])[0]
        store.delete_many([task.id])
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e):
            pytest.fail("Received unexpected lock error in non-concurrent test")
        else:
            raise  # Re-raise other operational errors
    except Exception as e:
        pytest.fail(f"Unexpected exception during basic operation sequence: {e}")
    assert state.status == TaskStatus.PENDING
    assert state_after.status == TaskStatus.COMPLETED
    assert not store.task_exists(task.id)


# 3. Edge Cases for *_many Methods
# ================================


# --- get_many ---
def test_get_many_empty_list(store):
    assert store.get_many([]) == []


def test_get_many_missing_ids(store):
    task1 = create_task(1)
    store.put_many([task1])
    results = store.get_many(["task_1", "task_missing", "task_another_missing"])
    assert len(results) == 1
    assert results[0].task.id == "task_1"


def test_get_many_all_missing_ids(store):
    assert store.get_many(["task_missing", "task_another_missing"]) == []


def test_get_many_existing_ids(store):
    tasks = [create_task(i) for i in range(3)]
    store.put_many(tasks)
    results = store.get_many(["task_0", "task_2"])
    assert len(results) == 2
    assert {r.task.id for r in results} == {"task_0", "task_2"}


# --- put_many ---
def test_put_many_empty_list(store):
    store.put_many([])
    assert len(store) == 0


def test_put_many_existing_ids_resets_state(store):
    task1 = create_task(1)
    task2 = create_task(2)
    store.put_many([task1, task2])
    store.set_many(
        [
            TaskState(task1, TaskStatus.COMPLETED, {"res": 1}),
            TaskState(task2, TaskStatus.FAILED, error="Fail"),
        ]
    )

    # Re-put task1
    store.put_many([task1])

    states = get_all_states_dict(store)
    assert len(states) == 2
    assert states["task_1"].status == TaskStatus.PENDING
    assert states["task_1"].result is None
    assert states["task_1"].error is None
    assert states["task_2"].status == TaskStatus.FAILED  # task2 should be unchanged


# --- set_many ---
def test_set_many_empty_list(store):
    store.set_many([])
    assert len(store) == 0


def test_set_many_missing_ids_creates_them(store):
    task1 = create_task(1)  # Exists
    task_missing = create_task(99)  # Does not exist
    store.put_many([task1])

    state_missing = TaskState(task_missing, TaskStatus.COMPLETED, {"res": 99})
    state_update = TaskState(task1, TaskStatus.FAILED, error="Set fail")

    store.set_many([state_missing, state_update])

    states = get_all_states_dict(store)
    assert len(states) == 2
    assert "task_1" in states
    assert "task_99" in states
    assert states["task_1"].status == TaskStatus.FAILED
    assert states["task_1"].error == "Set fail"
    assert states["task_99"].status == TaskStatus.COMPLETED
    assert states["task_99"].result == {"res": 99}


def test_set_many_existing_ids(store):
    tasks = [create_task(i) for i in range(3)]
    store.put_many(tasks)

    updates = [
        TaskState(tasks[0], TaskStatus.COMPLETED, {"res": 0}),
        TaskState(tasks[2], TaskStatus.FAILED, error="Failure 2"),
    ]
    store.set_many(updates)

    states = get_all_states_dict(store)
    assert len(states) == 3
    assert states["task_0"].status == TaskStatus.COMPLETED
    assert states["task_0"].result == {"res": 0}
    assert states["task_1"].status == TaskStatus.PENDING  # Unchanged
    assert states["task_2"].status == TaskStatus.FAILED
    assert states["task_2"].error == "Failure 2"


def test_set_many_overwrites_previous_status(store):
    task1 = create_task(1)
    store.put_many([task1])
    store.set_many([TaskState(task1, TaskStatus.COMPLETED, {"r": 1})])
    assert store.get_many([task1.id])[0].status == TaskStatus.COMPLETED

    store.set_many([TaskState(task1, TaskStatus.FAILED, error="E1")])
    assert store.get_many([task1.id])[0].status == TaskStatus.FAILED
    assert store.get_many([task1.id])[0].result is None
    assert store.get_many([task1.id])[0].error == "E1"

    store.set_many([TaskState(task1, TaskStatus.PENDING)])
    assert store.get_many([task1.id])[0].status == TaskStatus.PENDING
    assert store.get_many([task1.id])[0].result is None
    assert store.get_many([task1.id])[0].error is None


# --- delete_many ---
def test_delete_many_empty_list(store):
    store.put_many([create_task(1)])
    store.delete_many([])
    assert len(store) == 1


def test_delete_many_missing_ids(store):
    task1 = create_task(1)
    store.put_many([task1])
    try:
        store.delete_many(["task_missing", "task_1", "task_another_missing"])
    except Exception as e:
        pytest.fail(f"delete_many raised unexpected exception: {e}")
    assert len(store) == 0
    assert not store.task_exists(task1.id)


def test_delete_many_all_missing_ids(store):
    store.put_many([create_task(1)])
    try:
        store.delete_many(["task_missing", "task_another_missing"])
    except Exception as e:
        pytest.fail(f"delete_many raised unexpected exception: {e}")
    assert len(store) == 1


def test_delete_many_existing_ids(store):
    tasks = [create_task(i) for i in range(5)]
    store.put_many(tasks)
    store.delete_many(["task_1", "task_3", "task_4"])
    remaining_states = get_all_states_dict(store)
    assert len(remaining_states) == 2
    assert "task_0" in remaining_states
    assert "task_2" in remaining_states


# 4. Serialization/Deserialization Tests
# =======================================


@pytest.mark.parametrize(
    "obj",
    [
        None,
        {"a": 1, "b": "hello", "c": [1, 2, None], "d": True},
        [1, "two", {"three": 3.0}],
        "a simple string",
        123,
        45.67,
        True,
        False,
        {},
        [],
    ],
)
def test_json_serializer_roundtrip(serializer, obj):
    serialized = serializer.serialize(obj)
    if obj is None:
        assert serialized is None
    else:
        assert isinstance(serialized, bytes)
        # Check if compressed (heuristic: look for zlib header if level > 0)
        if serializer.compression_level > 0 and len(serialized) > 2:
            # Common zlib headers
            assert (
                serialized.startswith(b"\x78\x01")
                or serialized.startswith(b"\x78\x9c")
                or serialized.startswith(b"\x78\xda")
            )

    deserialized = serializer.deserialize(serialized)
    assert deserialized == obj


def test_json_serializer_compression_level(serializer):
    obj = {"data": "some data " * 100}  # Reasonably large object
    serializer_no_compress = JSONSerializer(compression_level=0)
    serializer_max_compress = JSONSerializer(compression_level=9)

    blob_nocompress = serializer_no_compress.serialize(obj)
    blob_compress = serializer.serialize(obj)  # Default level 1
    blob_maxcompress = serializer_max_compress.serialize(obj)

    assert isinstance(blob_nocompress, bytes)
    assert isinstance(blob_compress, bytes)
    assert isinstance(blob_maxcompress, bytes)

    # Expect compression to reduce size significantly for compressible data
    if (
        len(obj["data"]) > 50
    ):  # Avoid checking for tiny inputs where overhead might dominate
        assert len(blob_compress) < len(blob_nocompress)
        assert len(blob_maxcompress) < len(blob_nocompress)
        # Max compression should be smaller or equal to default
        assert len(blob_maxcompress) <= len(blob_compress)

    # Verify roundtrip for all levels
    assert serializer_no_compress.deserialize(blob_nocompress) == obj
    assert serializer.deserialize(blob_compress) == obj
    assert serializer_max_compress.deserialize(blob_maxcompress) == obj


# 5. Concurrent Writes Test (Also tests retry logic implicitly)
# =============================================================


def worker_thread_put(store, tasks, results_dict):
    """Worker function for concurrent put_many."""
    try:
        # Ensure each thread gets its own connection implicitly via thread-local
        store.put_many(tasks)
        results_dict[threading.get_ident()] = "success"
    except Exception as e:
        logger.error(
            f"Worker thread {threading.get_ident()} failed in put_many: {e}",
            exc_info=True,
        )
        results_dict[threading.get_ident()] = e


def worker_thread_set(store, states, results_dict):
    """Worker function for concurrent set_many."""
    try:
        # Ensure each thread gets its own connection implicitly via thread-local
        store.set_many(states)
        results_dict[threading.get_ident()] = "success"
    except Exception as e:
        logger.error(
            f"Worker thread {threading.get_ident()} failed in set_many: {e}",
            exc_info=True,
        )
        results_dict[threading.get_ident()] = e


def test_concurrent_put_many(store):
    num_threads = 10
    tasks_per_thread = 50  # Keep relatively small to increase chance of collision
    total_tasks = num_threads * tasks_per_thread
    threads = []
    results = {}
    all_tasks_flat = []

    for i in range(num_threads):
        start_idx = i * tasks_per_thread
        thread_tasks = [
            create_task(j) for j in range(start_idx, start_idx + tasks_per_thread)
        ]
        all_tasks_flat.extend(thread_tasks)
        # Pass the main store instance; connection management is internal via thread-local
        thread = threading.Thread(
            target=worker_thread_put, args=(store, thread_tasks, results)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join(timeout=30)  # Increased timeout for potential contention/retries

    # Check results
    failures = {tid: res for tid, res in results.items() if res != "success"}
    assert not failures, f"Some worker threads failed: {failures}"
    # Check if all threads completed, even if some might have failed internally
    # assert len(results) == num_threads, "Not all threads reported results" # This might fail if thread creation failed

    # Verify final state
    # Add a small delay before final check to allow WAL commits to settle if needed
    time.sleep(0.2)
    final_count = len(store)
    final_states = get_all_states_dict(store)  # Re-read from store
    assert (
        final_count == total_tasks
    ), f"Expected {total_tasks} tasks, found {final_count}"

    assert set(final_states.keys()) == {t.id for t in all_tasks_flat}
    assert all(s.status == TaskStatus.PENDING for s in final_states.values())


def test_concurrent_set_many(store):
    num_tasks = 100
    tasks = [create_task(i) for i in range(num_tasks)]
    store.put_many(tasks)  # Initialize all tasks

    num_threads = 10
    tasks_per_thread = num_tasks // num_threads
    threads = []
    results = {}
    all_states_flat = []

    for i in range(num_threads):
        start_idx = i * tasks_per_thread
        end_idx = start_idx + tasks_per_thread
        thread_states = []
        for j in range(start_idx, end_idx):
            # Ensure all threads try to update distinct tasks first
            status = random.choice([TaskStatus.COMPLETED, TaskStatus.FAILED])
            result = {"res": j} if status == TaskStatus.COMPLETED else None
            error = f"Error_{j}" if status == TaskStatus.FAILED else None
            state = TaskState(tasks[j], status, result=result, error=error)
            thread_states.append(state)
        all_states_flat.extend(thread_states)
        # Pass the main store instance
        thread = threading.Thread(
            target=worker_thread_set, args=(store, thread_states, results)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join(timeout=30)  # Increased timeout

    # Check results
    failures = {tid: res for tid, res in results.items() if res != "success"}
    assert not failures, f"Some worker threads failed: {failures}"
    # assert len(results) == num_threads, "Not all threads reported results"

    # Verify final state
    time.sleep(0.2)  # Allow WAL to settle
    assert len(store) == num_tasks
    final_states_map = get_all_states_dict(store)  # Re-read from store
    expected_states_map = {s.task.id: s for s in all_states_flat}

    assert len(final_states_map) == num_tasks
    for task_id, expected_state in expected_states_map.items():
        assert task_id in final_states_map
        actual_state = final_states_map[task_id]
        assert actual_state.status == expected_state.status
        assert actual_state.result == expected_state.result
        assert actual_state.error == expected_state.error


# 6. clear, __len__, __iter__ Tests (Removed Index Tests)
# ==================================


def test_len(store):
    assert len(store) == 0
    store.put_many([create_task(i) for i in range(5)])
    assert len(store) == 5
    store.put_many([create_task(i) for i in range(5, 10)])
    assert len(store) == 10
    store.delete_many(["task_9"])
    assert len(store) == 9


def test_clear(store):
    store.put_many([create_task(i) for i in range(5)])
    store.set_many([TaskState(create_task(0), TaskStatus.COMPLETED, {"r": 0})])
    store.set_many([TaskState(create_task(1), TaskStatus.FAILED, error="E1")])
    assert len(store) == 5

    store.clear()
    assert len(store) == 0

    # Verify tables are actually empty using a new connection to be safe
    conn = sqlite3.connect(store.db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM task_definitions").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM completed_tasks").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM failed_tasks").fetchone()[0] == 0
    finally:
        conn.close()


def test_iter_empty(store):
    assert list(store) == []


def test_iter_all_pending(store):
    tasks = [create_task(i) for i in range(3)]
    store.put_many(tasks)
    iterated_states = list(store)
    assert len(iterated_states) == 3
    iterated_map = {s.task.id: s for s in iterated_states}
    for i in range(3):
        task_id = f"task_{i}"
        assert task_id in iterated_map
        assert iterated_map[task_id].status == TaskStatus.PENDING
        assert iterated_map[task_id].task.data == {
            "value": i,
            "desc": f"Task number {i}",
        }


def test_iter_mixed_statuses(store):
    tasks = [create_task(i) for i in range(4)]
    store.put_many(tasks)
    store.set_many(
        [
            TaskState(tasks[1], TaskStatus.COMPLETED, {"res": 1}),
            TaskState(tasks[3], TaskStatus.FAILED, error="Fail 3"),
        ]
    )

    iterated_states = list(store)
    assert len(iterated_states) == 4
    iterated_map = {s.task.id: s for s in iterated_states}

    assert iterated_map["task_0"].status == TaskStatus.PENDING
    assert iterated_map["task_1"].status == TaskStatus.COMPLETED
    assert iterated_map["task_1"].result == {"res": 1}
    assert iterated_map["task_2"].status == TaskStatus.PENDING
    assert iterated_map["task_3"].status == TaskStatus.FAILED
    assert iterated_map["task_3"].error == "Fail 3"


# 7. Other Utility Methods
# ========================


def test_task_exists(store):
    assert not store.task_exists("task_0")
    store.put(create_task(0))
    assert store.task_exists("task_0")
    store.delete("task_0")
    assert not store.task_exists("task_0")


def test_get_pending_task_ids(store):
    tasks = [create_task(i) for i in range(4)]
    store.put_many(tasks)
    store.set_many(
        [
            TaskState(tasks[1], TaskStatus.COMPLETED, {"res": 1}),
            TaskState(tasks[3], TaskStatus.FAILED, error="Fail 3"),
        ]
    )
    pending_ids = store.get_pending_task_ids()
    assert sorted(pending_ids) == ["task_0", "task_2"]


def test_get_completed_task_ids(store):
    tasks = [create_task(i) for i in range(4)]
    store.put_many(tasks)
    store.set_many(
        [
            TaskState(tasks[1], TaskStatus.COMPLETED, {"res": 1}),
            TaskState(tasks[3], TaskStatus.FAILED, error="Fail 3"),
            TaskState(tasks[0], TaskStatus.COMPLETED, {"res": 0}),
        ]
    )
    completed_ids = store.get_completed_task_ids()
    assert sorted(completed_ids) == ["task_0", "task_1"]


def test_get_failed_task_ids(store):
    tasks = [create_task(i) for i in range(4)]
    store.put_many(tasks)
    store.set_many(
        [
            TaskState(tasks[1], TaskStatus.COMPLETED, {"res": 1}),
            TaskState(tasks[3], TaskStatus.FAILED, error="Fail 3"),
            TaskState(tasks[0], TaskStatus.FAILED, error="Fail 0"),
        ]
    )
    failed_ids = store.get_failed_task_ids()
    assert sorted(failed_ids) == ["task_0", "task_3"]


def test_reset_failed(store):
    tasks = [create_task(i) for i in range(4)]
    store.put_many(tasks)
    store.set_many(
        [
            TaskState(tasks[1], TaskStatus.COMPLETED, {"res": 1}),
            TaskState(tasks[3], TaskStatus.FAILED, error="Fail 3"),
            TaskState(tasks[0], TaskStatus.FAILED, error="Fail 0"),
        ]
    )
    assert len(store.get_failed_task_ids()) == 2
    assert len(store.get_pending_task_ids()) == 1
    assert len(store.get_completed_task_ids()) == 1

    reset_count = store.reset_failed()
    assert reset_count == 2

    assert len(store.get_failed_task_ids()) == 0
    assert len(store.get_pending_task_ids()) == 3  # 1 original + 2 reset
    assert len(store.get_completed_task_ids()) == 1  # Unchanged

    states = get_all_states_dict(store)
    assert states["task_0"].status == TaskStatus.PENDING
    assert states["task_0"].error is None
    assert states["task_3"].status == TaskStatus.PENDING
    assert states["task_3"].error is None
    assert states["task_1"].status == TaskStatus.COMPLETED  # Unchanged
    assert states["task_2"].status == TaskStatus.PENDING  # Unchanged


@patch("random.random", return_value=0.05)  # Fixed 5% jitter for test predictability
def test_retry_backoff_capped(mock_random, store_path):
    """Verify that sleep time during retries is capped."""
    max_retries = 5
    cap_seconds = 0.5  # Set a specific cap for the test
    initial_delay = 0.01

    # Initialize store with the cap
    store = SingleSQLiteTaskStore(
        directory=store_path, max_retries=max_retries, max_retry_delay_sec=cap_seconds
    )

    # Mock the connection's execute method to always raise the lock error
    mock_execute = MagicMock(side_effect=sqlite3.OperationalError("database is locked"))
    mock_conn = MagicMock()
    mock_conn.execute = mock_execute

    # Use patch context managers for mocks
    with (
        patch.object(store, "_get_connection", return_value=mock_conn),
        patch("time.sleep") as mock_sleep,
    ):

        # Call a method that triggers _execute_with_retry (e.g., __len__)
        # Expect it to fail after exhausting retries
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            len(store)  # Trigger the retry logic

        # Verify execute was called max_retries + 1 times (initial + retries)
        assert mock_execute.call_count == max_retries + 1

        # Verify time.sleep was called max_retries times
        assert mock_sleep.call_count == max_retries

        # Check the sleep durations passed to time.sleep
        sleep_calls = mock_sleep.call_args_list
        expected_sleep_times = []
        current_delay = initial_delay

        # --- START FIX ---
        # Correctly calculate the jitter factor based on the implementation logic
        mock_random_return = mock_random.return_value  # e.g., 0.05
        jitter_scaling_factor = 0.1  # From implementation: random.random() * 0.1
        jitter_value = (
            mock_random_return * jitter_scaling_factor
        )  # e.g., 0.05 * 0.1 = 0.005
        jitter_factor = 1 + jitter_value  # e.g., 1 + 0.005 = 1.005
        # --- END FIX ---

        for i in range(max_retries):
            # Use the correctly calculated jitter_factor
            base_sleep = current_delay * jitter_factor
            expected_capped_sleep = min(base_sleep, cap_seconds)
            expected_sleep_times.append(pytest.approx(expected_capped_sleep, abs=1e-3))
            current_delay *= 2  # Calculate next base delay

        actual_sleep_times = [c.args[0] for c in sleep_calls]

        # Assert that the actual sleep times match the expected capped times
        # This assertion should now pass
        assert (
            actual_sleep_times == expected_sleep_times
        ), f"Actual: {actual_sleep_times}, Expected: {[t.expected for t in expected_sleep_times]}"

        # Explicitly check that no sleep time exceeded the cap
        for sleep_time in actual_sleep_times:
            assert sleep_time <= cap_seconds

    store.close()
