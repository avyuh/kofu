from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional
from tqdm import tqdm  # Import tqdm for progress bar
from kofu.memory.sqlite_memory import SQLiteMemory  # Import SQLiteMemory
from kofu.tasks import SimpleFn


class LocalThreadedExecutor:
    def __init__(
        self,
        tasks: List,
        memory=None,
        path: Optional[str] = None,
        max_concurrency: int = 4,
        stop_all_when: Optional[Callable] = None,
        retry: int = 1,
    ):
        """
        Initialize the LocalThreadedExecutor.

        :param tasks: List of task instances that can be executed.
        :param memory: Memory object (e.g., SQLiteMemory) to manage task states and results. If not provided, SQLiteMemory is used.
        :param path: Optional path for SQLiteMemory storage. Required if memory is not provided.
        :param max_concurrency: Maximum number of threads to run concurrently.
        :param stop_all_when: Function that returns True if execution should stop (e.g., rate limiting, API blocks).
        :param retry: Number of retries for each task in case of failure.
        """
        self.tasks = tasks
        self.memory = memory
        self.path = path
        self.max_concurrency = max_concurrency
        self.stop_all_when = stop_all_when
        self._stopped = False
        self.retry = retry  # Add retry parameter

        # Ensure memory is either provided or initialized
        if self.memory is None:
            if self.path is None:
                raise ValueError("Either a memory instance or a path must be provided")
            self.memory = SQLiteMemory(path=self.path)

    def status_summary(self):
        """
        Print a summary of task statuses: pending, completed, failed.
        """
        pending = self.memory.get_pending_tasks()
        completed = self.memory.get_completed_tasks()
        failed = self.memory.get_failed_tasks()

        print(f"Pending tasks: {len(pending)}")
        print(f"Completed tasks: {len(completed)}")
        print(f"Failed tasks: {len(failed)}")

    def run(self):
        """
        Run the tasks concurrently with a thread pool. Query memory for task status to skip completed tasks
        and stop execution if the stop_all_when condition is met.
        """
        # Ensure all tasks are stored in memory before execution (idempotent)
        self._initialize_tasks_in_memory()

        # Retrieve pending tasks from memory
        pending_task_ids = set(self.memory.get_pending_tasks())
        tasks_to_run = [
            task for task in self.tasks if task.get_id() in pending_task_ids
        ]

        if not tasks_to_run:
            print("All tasks are already completed.")
            return

        # Total task count for progress bar
        total_tasks = len(self.tasks)
        completed_tasks = len(self.memory.get_completed_tasks())
        failed_tasks = len(self.memory.get_failed_tasks())

        # Initialize progress bar
        with tqdm(
            total=total_tasks,
            desc="Task Progress",
            unit="task",
            initial=completed_tasks + failed_tasks,
        ) as pbar:
            # Thread pool execution
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
                future_to_task = {}
                # Submit tasks to the executor one by one, checking stop condition before submitting
                for task in tasks_to_run:
                    if self._stopped or (self.stop_all_when and self.stop_all_when()):
                        print("Stop condition met. Halting task submission.")
                        self._stopped = True
                        break

                    future = executor.submit(self._execute_task, task, self.retry)
                    future_to_task[future] = task

                # Collect results as tasks finish and update memory
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result = (
                            future.result()
                        )  # This will raise an exception if the task failed
                        self.memory.update_task_statuses(
                            [(task.get_id(), "completed", result, None)]
                        )
                        completed_tasks += 1
                    except Exception as e:
                        exception_type = type(e).__name__
                        exception_message = str(e)
                        error_info = f"{exception_type}: {exception_message}"
                        self.memory.update_task_statuses(
                            [(task.get_id(), "failed", None, error_info)]
                        )
                        failed_tasks += 1

                    # Update progress bar
                    pbar.update(1)

                    # Check the stop condition after each task is processed
                    if self.stop_all_when and self.stop_all_when():
                        print("Emergency stop condition met. Halting execution.")
                        self._stopped = True
                        break

        # Print status summary at the end
        self.status_summary()

    def _execute_task(self, task, retries_left):
        """
        Execute the given task and return its result.
        This function will be executed by each thread in the thread pool.
        """
        if self._stopped:
            raise RuntimeError("Execution was stopped by an external condition.")

        try:
            return task()  # Try to execute the task
        except Exception as e:
            if retries_left >= 1:
                print(
                    f"Retrying task {task.get_id()}... Attempts left: {retries_left-1}"
                )
                return self._execute_task(task, retries_left - 1)
            else:
                raise e  # Raise the error if no retries are left

    def _initialize_tasks_in_memory(self):
        """
        Ensure that all tasks are registered in the memory with a `pending` status if they are not already defined.
        This method is idempotent and will not overwrite existing tasks.
        """
        task_definitions = []

        for task in self.tasks:
            task_id = task.get_id()
            try:
                # Check if the task already exists in memory
                self.memory.get_task_status(task_id)
            except KeyError:
                # For SimpleFn tasks, we store function name and args; otherwise, store task type and other info.
                if isinstance(task, SimpleFn):
                    task_data = {
                        "fn_name": task.fn.__name__,
                        "args": task.args,
                        "kwargs": task.kwargs,
                    }
                else:
                    # Generic task data (optional, based on your task implementation)
                    task_data = {"task_type": type(task).__name__}

                # Store task definition with ID and data
                task_definitions.append((task_id, task_data))

        if task_definitions:
            self.memory.store_tasks(task_definitions)
