import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Set,
    TypedDict,
    cast,
)

from tqdm import tqdm

from .store import (
    TaskStore,
    TaskDefinition,
    TaskState,
    TaskStatus,
    SingleSQLiteTaskStore,
)
from .tasks import Task

# Configure logging
logger = logging.getLogger(__name__)


# Type definition for task data
class TaskData(TypedDict, total=False):
    fn_name: str
    args: tuple
    kwargs: dict
    task_type: str


class LocalThreadedExecutor:
    """Concurrent task executor with efficient state tracking.

    Optimized to use TaskStore implementations for state persistence and
    high-performance task execution.
    """

    def __init__(
        self,
        tasks: list[Task],
        store: Optional[TaskStore] = None,
        path: Optional[str] = None,
        max_concurrency: int = 4,
        stop_all_when: Optional[Callable[[], bool]] = None,
        retry: int = 1,
        batch_size: int = 50,
    ):
        """Initialize the executor.

        Args:
            tasks: List of task instances that conform to the Task protocol
            store: TaskStore for state persistence (default None, will create SingleSQLiteTaskStore)
            path: Path for store if none provided (required if store is None)
            max_concurrency: Maximum number of threads to run concurrently
            stop_all_when: Function returning True to stop execution (e.g., for rate limiting)
            retry: Number of retries for each task on failure
            batch_size: Number of tasks to process in a single batch update

        Raises:
            ValueError: If neither store nor path is provided
            TypeError: If any task does not conform to the Task protocol
        """
        logger.debug(
            "LocalThreadedExecutor.__init__: Starting initialization."
        )  # Added log

        # Validate all tasks conform to the Task protocol
        for i, task in enumerate(tasks):
            if not isinstance(task, Task):
                raise TypeError(
                    f"Task at index {i} does not conform to the Task protocol"
                )

        self.tasks = tasks
        self.path = path
        self.max_concurrency = max_concurrency
        self.stop_all_when = stop_all_when
        self._stopped = False
        self.retry = retry
        self.batch_size = batch_size

        # Task lookup for efficient access
        logger.debug("LocalThreadedExecutor.__init__: Creating task map.")  # Added log
        self._task_map = {task.id: task for task in tasks}
        logger.debug(
            f"LocalThreadedExecutor.__init__: Task map created with {len(self._task_map)} entries."
        )  # Added log

        # Initialize store
        if store is None:
            if path is None:
                raise ValueError("Either a store instance or a path must be provided")
                logger.debug(
                    f"LocalThreadedExecutor.__init__: Creating new SingleSQLiteTaskStore at path: {path}"
                )  # Added log
            self.store = SingleSQLiteTaskStore(directory=path)
        else:
            logger.debug(
                "LocalThreadedExecutor.__init__: Using provided store instance."
            )  # Added log
            self.store = store

    def status_summary(self) -> Dict[TaskStatus, int]:
        """Get a summary of task statuses.

        Returns:
            Dictionary mapping TaskStatus values to counts
        """
        # Use TaskStore utility functions
        from .store import get_status_summary

        summary = get_status_summary(self.store)

        # Display summary
        print(f"Pending tasks: {summary.get(TaskStatus.PENDING, 0)}")
        print(f"Completed tasks: {summary.get(TaskStatus.COMPLETED, 0)}")
        print(f"Failed tasks: {summary.get(TaskStatus.FAILED, 0)}")

        return summary

    def run(self) -> None:
        """Run tasks concurrently with optimized state handling.

        Processes all pending tasks in parallel using a thread pool.
        Tasks are batched for efficient database operations.
        Robust error handling ensures no task results are lost.
        """
        # Register all tasks (idempotent)
        logger.debug(
            "LocalThreadedExecutor.run: Calling _initialize_tasks."
        )  # Added log
        self._initialize_tasks()
        logger.debug("LocalThreadedExecutor.run: Finished initialization.")  # Added log

        # Get pending task IDs efficiently
        pending_task_ids = self.store.get_pending_task_ids()

        if not pending_task_ids:
            logger.info("All tasks are already completed.")
            return

        # Filter out tasks that aren't in our current task list
        tasks_to_run = [
            self._task_map[task_id]
            for task_id in pending_task_ids
            if task_id in self._task_map
        ]

        if not tasks_to_run:
            logger.info("No pending tasks found in current task list.")
            return

        logger.info(
            f"Running {len(tasks_to_run)} pending tasks out of {len(self.tasks)} total tasks"
        )

        # Task tracking
        all_task_count = len(self.tasks)
        completed_count = len(self.store.get_completed_task_ids())
        failed_count = len(self.store.get_failed_task_ids())

        # Ensure initial value doesn't exceed total for tqdm
        initial_progress = min(completed_count + failed_count, all_task_count)

        # Initialize progress bar with accurate counts
        with tqdm(
            total=all_task_count,
            desc="Task Progress",
            unit="task",
            initial=initial_progress,
        ) as pbar:
            # Process tasks using thread pool
            self._run_with_threadpool(tasks_to_run, pbar)

        # Final status summary
        self.status_summary()

    def _run_with_threadpool(self, tasks_to_run: list, pbar: tqdm) -> None:
        """Execute tasks using a thread pool with proper batching and error handling.

        Args:
            tasks_to_run: List of tasks to execute
            pbar: Progress bar to update
        """
        # Thread pool for execution
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            # Process tasks in batches for efficiency
            batches = [
                tasks_to_run[i : i + self.batch_size]
                for i in range(0, len(tasks_to_run), self.batch_size)
            ]

            for batch in batches:
                if self._check_stop_condition():
                    break

                # Submit all tasks in this batch
                future_to_task = {}
                for task in batch:
                    if self._check_stop_condition():
                        break
                    future = executor.submit(self._execute_task, task, self.retry)
                    future_to_task[future] = task

                # Track results for batch update
                completed_states: list[TaskState] = []
                failed_states: list[TaskState] = []

                # Process results as they complete
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    task_id = task.id

                    try:
                        # Get task result
                        result = future.result()

                        # Validate result is serializable
                        result = self._validate_result(result)

                        # Create task object matching our data model
                        task_obj = TaskDefinition(
                            id=task_id, data=self._get_task_data(task)
                        )

                        # Add to completed batch
                        completed_states.append(
                            TaskState(
                                task=task_obj,
                                status=TaskStatus.COMPLETED,
                                result=result,
                                error=None,
                            )
                        )
                    except Exception as e:
                        # Log the failure
                        logger.warning(f"Task {task_id} failed: {str(e)}")

                        # Create task object
                        task_obj = TaskDefinition(
                            id=task_id, data=self._get_task_data(task)
                        )

                        # Format error message
                        error_message = f"{type(e).__name__}: {str(e)}"

                        # Add to failed batch
                        failed_states.append(
                            TaskState(
                                task=task_obj,
                                status=TaskStatus.FAILED,
                                result=None,
                                error=error_message,
                            )
                        )

                    # Update progress bar
                    pbar.update(1)

                # Batch update the store with all completed/failed tasks
                if completed_states or failed_states:
                    self._update_task_states(completed_states, failed_states)

                # Check stop condition after batch processing
                if self._check_stop_condition():
                    break

    def _update_task_states(
        self, completed_states: list[TaskState], failed_states: list[TaskState]
    ) -> None:
        """Update task states with resilient batch operations.

        If a batch update fails, falls back to individual updates.

        Args:
            completed_states: List of completed task states
            failed_states: List of failed task states
        """
        # First try batch update for each status group
        if completed_states:
            try:
                with self.store.atomic():
                    self.store.set_many(completed_states)
            except Exception as e:
                logger.warning(
                    f"Batch update failed for completed tasks: {e}. Trying individual updates..."
                )
                self._update_individual_states(completed_states)

        if failed_states:
            try:
                with self.store.atomic():
                    self.store.set_many(failed_states)
            except Exception as e:
                logger.warning(
                    f"Batch update failed for failed tasks: {e}. Trying individual updates..."
                )
                self._update_individual_states(failed_states)

    def _update_individual_states(self, states: list[TaskState]) -> None:
        """Update task states individually when batch update fails.

        Args:
            states: List of task states to update
        """
        for state in states:
            try:
                self.store.set_many([state])
                logger.debug(f"Successfully updated task {state.task.id} individually")
            except Exception as e2:
                logger.error(f"Failed to save state for {state.task.id}: {e2}")

    def _check_stop_condition(self) -> bool:
        """Check if execution should stop based on stop condition or internal flag.

        Returns:
            True if execution should stop, False otherwise
        """
        if self._stopped:
            return True

        if self.stop_all_when and self.stop_all_when():
            logger.info("Stop condition met. Halting execution.")
            self._stopped = True
            return True

        return False

    def _execute_task(self, task: Any, retries_left: int) -> Any:
        """Execute task with retry logic.

        Args:
            task: Task to execute
            retries_left: Number of retries remaining

        Returns:
            Task result

        Raises:
            Exception: If task execution fails and no retries remain
        """
        if self._stopped:
            raise RuntimeError("Execution was stopped by an external condition")

        try:
            return task()
        except Exception:
            if retries_left >= 1:
                # Exponential backoff with jitter
                delay = 0.1 * (2 ** (self.retry - retries_left))
                jitter = random.random() * 0.1
                wait_time = delay + jitter

                logger.info(
                    f"Retrying task {task.id} in {wait_time:.2f}s... "
                    f"Attempts left: {retries_left-1}"
                )

                time.sleep(wait_time)
                return self._execute_task(task, retries_left - 1)
            else:
                # No more retries, propagate the exception
                raise

    def _validate_result(self, result: Any) -> Dict[str, Any]:
        """Validate and normalize task result to ensure it's serializable.

        Args:
            result: Raw task result

        Returns:
            Dictionary result suitable for storage
        """
        # Handle None result
        if result is None:
            return {}

        # Handle primitive results by wrapping
        if not isinstance(result, dict):
            return {"value": result}

        # Ensure we're working with a dict
        return result

    def _get_task_data(self, task: Task) -> TaskData:
        """Extract serializable data from a task.

        Args:
            task: Task object conforming to the Task protocol

        Returns:
            Dictionary with task metadata

        Raises:
            TypeError: If the task doesn't provide necessary attributes for metadata extraction
        """
        # Check for SimpleFn-like interface using duck typing
        if hasattr(task, "fn") and hasattr(task, "args") and hasattr(task, "kwargs"):
            # Special handling for SimpleFn-like tasks
            return {
                "fn_name": task.fn.__name__,
                "args": task.args,
                "kwargs": task.kwargs,
            }

        # For all other task types, collect basic information we can reliably extract
        task_info = {
            "task_type": type(task).__name__,
        }

        # Try to extract additional useful information if available
        if hasattr(task, "__dict__"):
            # Filter out callables and private attributes
            public_attrs = {
                k: v
                for k, v in task.__dict__.items()
                if not k.startswith("_") and not callable(v)
            }
            task_info.update(public_attrs)

        return task_info

    def _initialize_tasks(self) -> None:
        """Register all tasks with the store (idempotent).

        Uses batch operations for efficiency and resilient error handling.
        """
        logger.debug("--- Entering _initialize_tasks ---")  # Changed log alias

        # Gather existing tasks using efficient batch operations
        existing_ids: Set[str] = set()

        has_exists_method = hasattr(self.store, "task_exists") and callable(
            getattr(self.store, "task_exists")
        )  # Check callable
        task_ids = [task.id for task in self.tasks]
        logger.debug(
            f"_initialize_tasks: Processing {len(task_ids)} tasks provided to executor."
        )

        # Define check_batch_size here
        check_batch_size = 500

        if has_exists_method:
            logger.debug(
                "_initialize_tasks: Using task_exists method for existence check."
            )
            checked_count = 0
            try:
                for i in range(0, len(task_ids), check_batch_size):
                    batch_ids_to_check = task_ids[i : i + check_batch_size]
                    count_in_batch = 0
                    for task_id in batch_ids_to_check:
                        if cast(Any, self.store).task_exists(
                            task_id
                        ):  # Pass task_id directly
                            existing_ids.add(task_id)
                            count_in_batch += 1
                        checked_count += 1
                    logger.debug(
                        f"_initialize_tasks: Checked task_exists for batch {i // check_batch_size}, found {count_in_batch} existing in batch."
                    )
            except Exception as e:
                logger.warning(
                    f"_initialize_tasks: Error during task_exists check: {e}",
                    exc_info=True,
                )
            logger.debug(
                f"_initialize_tasks: Finished task_exists checks. Total checked: {checked_count}. Found {len(existing_ids)} existing."
            )

        else:
            # Efficient batch check with get_many
            logger.debug(
                "_initialize_tasks: Using get_many method for existence check."
            )
            try:
                for i in range(0, len(task_ids), check_batch_size):
                    batch_ids = task_ids[i : i + check_batch_size]
                    logger.debug(
                        f"_initialize_tasks: Calling get_many for {len(batch_ids)} IDs (batch {i // check_batch_size})."
                    )
                    # Ensure store is callable if using get_many
                    if callable(getattr(self.store, "get_many", None)):
                        states = self.store.get_many(batch_ids)
                        batch_existing_ids = {state.task.id for state in states}
                        existing_ids.update(batch_existing_ids)
                        logger.debug(
                            f"_initialize_tasks: get_many returned {len(states)} states for batch {i // check_batch_size}. Total existing now: {len(existing_ids)}"
                        )
                    else:
                        logger.warning(
                            "_initialize_tasks: store.get_many is not callable, skipping existence check."
                        )
                        break  # Cannot check existence this way

            except Exception as e:
                logger.warning(
                    f"_initialize_tasks: Error during bulk get_many for existence check: {e}",
                    exc_info=True,
                )

        logger.debug(
            f"_initialize_tasks: Total existing tasks identified via checks: {len(existing_ids)}"
        )

        # Create new tasks for anything not already in the store
        new_tasks: list[TaskDefinition] = []
        for task in self.tasks:
            task_id = task.id
            if task_id not in existing_ids:
                try:
                    task_data = self._get_task_data(task)
                    # Ensure task_data is serializable before adding
                    # (SimpleFn data should be, but good practice)
                    new_tasks.append(TaskDefinition(id=task_id, data=task_data))
                except Exception as e:
                    logger.error(
                        f"_initialize_tasks: Failed to create Task object for {task_id}: {e}",
                        exc_info=True,
                    )

        logger.debug(
            f"_initialize_tasks: Identified {len(new_tasks)} tasks to add/update in the store."
        )

        # Batch insert/update all new tasks
        if new_tasks:
            logger.info(
                f"_initialize_tasks: Registering/Updating {len(new_tasks)} tasks in store..."
            )

            # Process in batches
            # Use a batch size appropriate for put_many (might differ from check batch size)
            put_batch_size = min(
                500, self.batch_size * 5
            )  # Larger batches for initialization is often good
            logger.debug(
                f"_initialize_tasks: Using batch size {put_batch_size} for put_many."
            )

            for i in range(0, len(new_tasks), put_batch_size):
                batch = new_tasks[i : i + put_batch_size]
                logger.debug(
                    f"_initialize_tasks: Calling put_many for batch {i // put_batch_size + 1}/{ (len(new_tasks) + put_batch_size - 1) // put_batch_size } ({len(batch)} tasks)."
                )
                try:
                    # Make sure put_many call is actually happening
                    # logger.debug(f"put_many task IDs (sample): {[t.id for t in batch[:5]]}") # Can be verbose
                    self.store.put_many(batch)
                    logger.debug(
                        f"_initialize_tasks: put_many for batch {i // put_batch_size + 1} succeeded."
                    )
                except Exception as e:
                    logger.warning(
                        f"_initialize_tasks: Error registering batch {i // put_batch_size + 1} of tasks: {e}. Falling back to individual puts if possible.",
                        exc_info=True,  # Add traceback
                    )
                    # Fallback logic (optional, depends on desired robustness)
                    logger.info(
                        f"Attempting individual puts for failed batch {i // put_batch_size + 1}..."
                    )
                    success_count = 0
                    fail_count = 0
                    for task_to_put in batch:
                        try:
                            self.store.put_many(
                                [task_to_put]
                            )  # put_many expects a list
                            success_count += 1
                        except Exception as e2:
                            logger.error(
                                f"_initialize_tasks: Failed individual put for task {task_to_put.id}: {e2}"
                            )
                            fail_count += 1
                    logger.info(
                        f"Individual puts result for batch {i // put_batch_size + 1}: {success_count} succeeded, {fail_count} failed."
                    )

            logger.info("_initialize_tasks: Finished registering/updating tasks.")
        else:
            logger.info(
                "_initialize_tasks: No new tasks identified to register/update."
            )

        # Final check of store length
        try:
            current_len = len(self.store)
            logger.debug(
                f"_initialize_tasks: Store length after initialization attempt: {current_len}"
            )
        except Exception as e:
            logger.error(
                f"_initialize_tasks: Failed to get store length after init: {e}"
            )

        logger.debug("--- Exiting _initialize_tasks ---")
        """Register all tasks with the store (idempotent).

        Uses batch operations for efficiency and resilient error handling.
        """
        logger.debug("--- Entering _initialize_tasks ---")  # Changed log alias

        # Gather existing tasks using efficient batch operations
        existing_ids: Set[str] = set()

        # Use task_exists method if available (custom extension)
        has_exists_method = hasattr(self.store, "task_exists")

        task_ids = [task.id for task in self.tasks]
        logger.debug(
            f"_initialize_tasks: Processing {len(task_ids)} tasks provided to executor."
        )

        check_batch_size = 500
        if has_exists_method:
            logger.debug(
                "_initialize_tasks: Using task_exists method for existence check."
            )
            checked_count = 0
            for task_id in task_ids:
                try:
                    if cast(Any, self.store).task_exists(task_id):
                        existing_ids.add(task_id)
                except Exception as e:
                    logger.debug(f"Error checking if task {task_id} exists: {e}")
        else:
            # Efficient batch check with get_many
            try:
                # Get tasks in batches to avoid large operations
                for i in range(0, len(task_ids), 500):
                    batch_ids = task_ids[i : i + 500]
                    states = self.store.get_many(batch_ids)
                    existing_ids.update(state.task.id for state in states)
            except Exception as e:
                logger.warning(f"Error checking existing tasks: {e}")

        # Create new tasks for anything not already in the store
        new_tasks: list[TaskDefinition] = []
        for task in self.tasks:
            task_id = task.id
            if task_id not in existing_ids:
                task_data = self._get_task_data(task)
                new_tasks.append(TaskDefinition(id=task_id, data=task_data))

        # Batch insert all new tasks
        if new_tasks:
            logger.info(f"Registering {len(new_tasks)} new tasks")

            # Process in batches to avoid transaction timeouts
            batch_size = min(
                500, self.batch_size * 5
            )  # Larger batches for initialization
            for i in range(0, len(new_tasks), batch_size):
                batch = new_tasks[i : i + batch_size]
                try:
                    self.store.put_many(batch)
                except Exception as e:
                    logger.warning(
                        f"Error registering batch of tasks: {e}, falling back to individual"
                    )
                    # Try individual inserts as fallback
                    for task in batch:
                        try:
                            self.store.put_many([task])
                        except Exception as e2:
                            logger.error(f"Failed to register task {task.id}: {e2}")

    def reset_failed_tasks(self) -> int:
        """Reset all failed tasks to pending status.

        Returns:
            Number of tasks reset
        """
        return self.store.reset_failed()

    def get_results(self) -> Dict[str, Any]:
        """Get all completed task results.

        Returns:
            Dictionary mapping task IDs to results

        Raises:
            Exception: If store access fails
        """
        from .store import get_all_results

        try:
            return get_all_results(self.store)
        except Exception as e:
            logger.error(f"Error getting results: {e}")
            raise

    def get_errors(self) -> Dict[str, str]:
        """Get all failed task errors.

        Returns:
            Dictionary mapping task IDs to error messages

        Raises:
            Exception: If store access fails
        """
        from .store import get_errors

        try:
            return get_errors(self.store)
        except Exception as e:
            logger.error(f"Error getting errors: {e}")
            raise

    def close(self) -> None:
        """Clean up resources.

        Closes the store connection if supported.
        """
        if hasattr(self.store, "close"):
            try:
                cast(Any, self.store).close()
            except Exception as e:
                logger.warning(f"Error closing store: {e}")

    def __enter__(self) -> "LocalThreadedExecutor":
        """Context manager support.

        Returns:
            Self for use in with statement
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up resources when exiting context.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        self.close()
