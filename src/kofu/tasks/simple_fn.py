from typing import Callable, Any, Tuple


class SimpleFn:
    def __init__(self, task_id: str, fn: Callable, args: Tuple = (), kwargs: dict = {}):
        """
        Initialize the SimpleFn task.

        :param task_id: Unique ID for the task
        :param fn: The function to be executed
        :param args: Positional arguments to pass to the function
        :param kwargs: Keyword arguments to pass to the function
        """
        self.task_id = task_id
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def get_id(self) -> str:
        """
        Return the unique task ID.
        """
        return self.task_id

    def __call__(self) -> Any:
        """
        Execute the function with the provided arguments and return the result.
        """
        return self.fn(*self.args, **self.kwargs)
