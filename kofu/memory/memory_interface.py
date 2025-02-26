from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple

class Memory(ABC):
    @abstractmethod
    def store_tasks(self, tasks: List[Tuple[str, dict]]):
        """Store multiple tasks in memory. Each task is a tuple (task_id, task_data)."""
        pass
    
    @abstractmethod
    def update_task_statuses(self, statuses: List[Tuple[str, str, Optional[dict], Optional[str]]]):
        """
        Update the status of multiple tasks. Each status update is a tuple of:
        (task_id, status, result, error).
        """
        pass

    @abstractmethod
    def get_task_status(self, task_id: str) -> str:
        """Retrieve the status of a task by task_id."""
        pass

    @abstractmethod
    def get_pending_tasks(self) -> List[str]:
        """Retrieve all tasks that are pending."""
        pass
    
    @abstractmethod
    def get_completed_tasks(self) -> List[str]:
        """Retrieve all tasks that have been completed successfully."""
        pass

    @abstractmethod
    def get_failed_tasks(self) -> List[Tuple[str, str]]:
        """Retrieve all tasks that have failed, along with their errors."""
        pass

    @abstractmethod
    def get_task_result(self, task_id: str) -> Optional[dict]:
        """Retrieve the result of a completed task."""
        pass

    @abstractmethod
    def clear(self):
        """Clear all tasks in memory."""
        pass

    @abstractmethod
    def clear_tasks(self, task_ids: List[str]):
        """Clear specific tasks by task IDs."""
        pass

    @abstractmethod
    def dump_all(self) -> Dict[str, Dict[str, dict]]:
        """
        Dump all task data, including statuses, results, and errors, as a nested dictionary:
        {
            "task_definitions": {task_id: task_data, ...},
            "task_statuses": {task_id: status, ...},
            "task_results": {task_id: result, ...},
            "task_errors": {task_id: error, ...}
        }
        """
        pass
