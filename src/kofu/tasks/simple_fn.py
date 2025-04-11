from dataclasses import dataclass
from typing import Callable, Any, Tuple, Dict


@dataclass
class SimpleFn:
    """A simple function-based task implementation.

    This class wraps a Python function with its arguments to create a task
    that can be executed by Kofu's execution framework.
    """

    task_id: str
    fn: Callable
    args: Tuple = ()
    kwargs: Dict = None

    def __post_init__(self):
        """Initialize after dataclass construction."""
        # Ensure kwargs is never None
        if self.kwargs is None:
            self.kwargs = {}

    @property
    def id(self) -> str:
        """Return the unique task ID."""
        return self.task_id

    def __call__(self) -> Any:
        """
        Execute the function with the provided arguments and return the result.
        """
        return self.fn(*self.args, **self.kwargs)
