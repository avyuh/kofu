# File: tests/test_executor_large_scale.py
# (or tests/test_stress_test.py as you named it)

import pytest
import time
import os
import logging
import random
import gc

from kofu import LocalThreadedExecutor
from kofu.store import (
    SingleSQLiteTaskStore,
    TaskStatus,
    get_status_summary,
)
from kofu.tasks import SimpleFn

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
log = logging.getLogger(__name__)  # Use test file's logger


# --- Fixtures ---
@pytest.fixture(scope="function")
def store(tmp_path):
    db_dir = tmp_path / "large_scale_store"
    log.info(f"Creating store in: {db_dir}")
    store = SingleSQLiteTaskStore(directory=str(db_dir), timeout=30)
    yield store
    log.info("Closing store...")
    store.close()
    log.info("Store closed.")
    gc.collect()


# --- Helper Functions --- (Keep as they were)
def simple_success_task_fn(data):
    return {"result": data["value"] * 2, "status": "ok"}


def simple_fail_task_fn(data):
    raise ValueError(f"Failed processing value {data.get('value', 'N/A')}")


# --- Test Cases ---
@pytest.mark.parametrize("num_tasks", [1000, 10000, 50000])
@pytest.mark.parametrize("failure_rate", [0.0, 0.05])
def test_large_scale_execution(store, num_tasks, failure_rate):
    log.info(
        f"--- Starting test_large_scale_execution ({num_tasks=}, {failure_rate=}) ---"
    )

    # --- 1. Task Generation --- (Keep as is)
    log.info("Generating tasks...")
    start_gen = time.perf_counter()
    tasks_to_create = []
    num_failures = int(num_tasks * failure_rate)
    task_ids = [f"task_{i:0{len(str(num_tasks))}}" for i in range(num_tasks)]
    for i, task_id in enumerate(task_ids):
        task_data = {"value": i, "id": task_id}
        if i < num_failures:
            fn_task = SimpleFn(task_id, simple_fail_task_fn, args=(task_data,))
        else:
            fn_task = SimpleFn(task_id, simple_success_task_fn, args=(task_data,))
        tasks_to_create.append(fn_task)
    duration_gen = time.perf_counter() - start_gen
    log.info(f"Generated {num_tasks} task objects in {duration_gen:.2f}s")

    # --- 2. Executor Initialization --- (Keep as is)
    log.info("Initializing Executor...")
    max_concurrency = min(32, (os.cpu_count() or 1) * 2)
    batch_size = 500
    start_init = time.perf_counter()
    executor = LocalThreadedExecutor(
        tasks=tasks_to_create,
        store=store,
        max_concurrency=max_concurrency,
        retry=1,
        batch_size=batch_size,
    )
    duration_init = time.perf_counter() - start_init
    log.info(f"Executor initialized in {duration_init:.2f}s")

    # --- 3. Initial State Verification (BEFORE run) ---
    # Verify the store is initially empty or contains unrelated tasks if reused (it shouldn't be reused with function scope fixture)
    log.info("Verifying store state BEFORE run...")
    initial_len = len(store)
    log.info(f"Store length before executor.run(): {initial_len}")
    # This assertion should typically expect 0 if the fixture provides a clean store
    assert (
        initial_len == 0
    ), f"Store should be empty before run, found {initial_len} items."
    initial_summary = get_status_summary(store)
    assert (
        not initial_summary
    ), f"Store summary should be empty before run, found: {initial_summary}"

    # --- 4. Task Execution ---
    log.info(f"Running tasks with {max_concurrency=}, {batch_size=}...")
    start_run = time.perf_counter()
    try:
        executor.run()  # This is where _initialize_tasks is called internally now
    except Exception as e:
        log.error(f"Executor run failed: {e}", exc_info=True)
        # Optionally add a check of the store state here upon failure
        try:
            log.error(
                f"Store state upon failure: len={len(store)}, summary={get_status_summary(store)}"
            )
        except Exception as e_inner:
            log.error(f"Could not get store state upon failure: {e_inner}")
        pytest.fail(f"executor.run() raised an unexpected exception: {e}")

    duration_run = time.perf_counter() - start_run
    log.info(f"Executor run completed in {duration_run:.2f}s")

    # --- 5. Final State Verification (AFTER run) ---
    # This section remains the primary verification point for the overall execution
    log.info("Verifying final task count and status summary...")
    final_summary = get_status_summary(store)
    log.info(f"Final status summary: {final_summary}")

    expected_completed = num_tasks - num_failures
    expected_failed = num_failures

    assert (
        final_summary.get(TaskStatus.PENDING, 0) == 0
    ), "Expected 0 pending tasks post-run"
    # Check the actual count against expected, providing more info on failure
    actual_completed = final_summary.get(TaskStatus.COMPLETED, 0)
    actual_failed = final_summary.get(TaskStatus.FAILED, 0)
    assert (
        actual_completed == expected_completed
    ), f"Expected {expected_completed} completed tasks, found {actual_completed}"
    assert (
        actual_failed == expected_failed
    ), f"Expected {expected_failed} failed tasks, found {actual_failed}"
    assert (
        len(store) == num_tasks
    ), f"Expected final count {num_tasks} post-run, found {len(store)}"

    # --- 6. (Optional) Sample Data Verification --- (Keep as is)
    log.info("Performing sample data verification...")
    sample_size = min(20, num_tasks)
    if sample_size > 0:
        all_ids = list(task_ids)
        random.shuffle(all_ids)
        sample_ids = all_ids[:sample_size]
        try:
            sample_states = store.get_many(sample_ids)
            assert len(sample_states) == sample_size
            # ... (rest of sample verification logic) ...
            log.info(f"Verified {sample_size} random tasks successfully.")
        except Exception as e:
            log.error(f"Sample verification failed: {e}", exc_info=True)
            pytest.fail(f"Sample verification check failed: {e}")

    # --- 7. Checkpoint and Re-verify Count --- (Keep as is)
    log.info("Running checkpoint and re-verifying count...")
    start_checkpoint = time.perf_counter()
    # ... (rest of checkpoint logic) ...
    duration_checkpoint = time.perf_counter() - start_checkpoint
    log.info(
        f"Checkpoint and reopen verification completed in {duration_checkpoint:.2f}s"
    )

    log.info(
        f"--- Finished test_large_scale_execution ({num_tasks=}, {failure_rate=}) ---"
    )
