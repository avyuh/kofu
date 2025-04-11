import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Optional, Set, TypedDict, cast

from tqdm import tqdm

from .store import TaskStore, Task, TaskState, TaskStatus, SingleSQLiteTaskStore
from .tasks import SimpleFn

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
        tasks: list,
        store: Optional[TaskStore] = None,
        path: Optional[str] = None,
        max_concurrency: int = 4,
        stop_all_when: Optional[Callable[[], bool]] = None,
        retry: int = 1,
        batch_size: int = 50,
    ):
        """Initialize the executor.

        Args:
            tasks: List of task instances that can be executed
            store: TaskStore for state persistence (default None, will create SingleSQLiteTaskStore)
            path: Path for store if none provided (required if store is None)
            max_concurrency: Maximum number of threads to run concurrently
            stop_all_when: Function returning True to stop execution (e.g., for rate limiting)
            retry: Number of retries for each task on failure
            batch_size: Number of tasks to process in a single batch update

        Raises:
            ValueError: If neither store nor path is provided
        """
        self.tasks = tasks
        self.path = path
        self.max_concurrency = max_concurrency
        self.stop_all_when = stop_all_when
        self._stopped = False
        self.retry = retry
        self.batch_size = batch_size

        # Task lookup for efficient access
        self._task_map = {task.get_id(): task for task in tasks}

        # Initialize store
        if store is None:
            if path is None:
                raise ValueError("Either a store instance or a path must be provided")
            self.store = SingleSQLiteTaskStore(directory=path)
        else:
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
        self._initialize_tasks()

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
                    task_id = task.get_id()

                    try:
                        # Get task result
                        result = future.result()

                        # Validate result is serializable
                        result = self._validate_result(result)

                        # Create task object matching our data model
                        task_obj = Task(id=task_id, data=self._get_task_data(task))

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
                        task_obj = Task(id=task_id, data=self._get_task_data(task))

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
                    f"Retrying task {task.get_id()} in {wait_time:.2f}s... "
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

    def _get_task_data(self, task: Any) -> TaskData:
        """Extract serializable data from a task.

        Args:
            task: Task object

        Returns:
            Dictionary with task metadata
        """
        if isinstance(task, SimpleFn):
            # Special handling for SimpleFn tasks
            return {
                "fn_name": task.fn.__name__,
                "args": task.args,
                "kwargs": task.kwargs,
            }
        else:
            # Generic task data
            return {"task_type": type(task).__name__}

    def _initialize_tasks(self) -> None:
        """Register all tasks with the store (idempotent).

        Uses batch operations for efficiency and resilient error handling.
        """
        # Gather existing tasks using efficient batch operations
        existing_ids: Set[str] = set()

        # Use task_exists method if available (custom extension)
        has_exists_method = hasattr(self.store, "task_exists")

        task_ids = [task.get_id() for task in self.tasks]

        if has_exists_method:
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
        new_tasks: list[Task] = []
        for task in self.tasks:
            task_id = task.get_id()
            if task_id not in existing_ids:
                task_data = self._get_task_data(task)
                new_tasks.append(Task(id=task_id, data=task_data))

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
