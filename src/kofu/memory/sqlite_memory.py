import sqlite3
import json
import os
from typing import List, Dict, Optional, Tuple
from .memory_interface import Memory
from threading import Lock


class SQLiteMemory(Memory):
    def __init__(self, path: str):
        # Ensure the directory exists
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = Lock()  # Use a lock to ensure thread safety for write operations
        self._setup_db()

    def _setup_db(self):
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_definition (
                    task_id TEXT PRIMARY KEY,
                    task_data TEXT
                )
            """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_status (
                    task_id TEXT PRIMARY KEY,
                    status TEXT,
                    FOREIGN KEY(task_id) REFERENCES task_definition(task_id)
                )
            """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_result (
                    task_id TEXT PRIMARY KEY,
                    result TEXT,
                    FOREIGN KEY(task_id) REFERENCES task_definition(task_id)
                )
            """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_error (
                    task_id TEXT PRIMARY KEY,
                    error TEXT,
                    FOREIGN KEY(task_id) REFERENCES task_definition(task_id)
                )
            """
            )

    def store_tasks(self, tasks: List[Tuple[str, dict]]):
        serialized_tasks = []

        # Validate task data and prepare it for insertion
        for task_id, task_data in tasks:
            try:
                json_data = json.dumps(task_data)  # Ensure it's JSON serializable
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"Task data for task {task_id} is not JSON serializable: {e}"
                )

            serialized_tasks.append((task_id, json_data))

        # Use executemany for batch insertion
        with self.conn, self.lock:
            self.conn.executemany(
                "INSERT OR IGNORE INTO task_definition (task_id, task_data) VALUES (?, ?)",
                serialized_tasks,
            )
            self.conn.executemany(
                "INSERT OR IGNORE INTO task_status (task_id, status) VALUES (?, ?)",
                [(task_id, "pending") for task_id, _ in tasks],
            )

    def update_task_statuses(
        self, statuses: List[Tuple[str, str, Optional[dict], Optional[str]]]
    ):
        with self.conn, self.lock:
            # Extract task_ids from the statuses list
            task_ids = [task_id for task_id, _, _, _ in statuses]

            # Check if all task IDs exist in a single query
            cursor = self.conn.execute(
                "SELECT task_id FROM task_status WHERE task_id IN ({})".format(
                    ",".join("?" * len(task_ids))
                ),
                task_ids,
            )
            existing_task_ids = {row[0] for row in cursor.fetchall()}

            # Raise KeyError if any task is missing
            for task_id in task_ids:
                if task_id not in existing_task_ids:
                    raise KeyError(f"Task {task_id} does not exist")

            self.conn.executemany(
                """
                UPDATE task_status SET status = ? WHERE task_id = ?
            """,
                [(status, task_id) for task_id, status, _, _ in statuses],
            )

            # TODO: Make this more optimized
            for task_id, _, result, error in statuses:
                if result is not None:
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO task_result (task_id, result) VALUES (?, ?)
                    """,
                        (task_id, json.dumps(result)),
                    )
                if error:
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO task_error (task_id, error) VALUES (?, ?)
                    """,
                        (task_id, json.dumps(error)),
                    )

    def get_task_status(self, task_id: str) -> str:
        cursor = self.conn.execute(
            "SELECT status FROM task_status WHERE task_id = ?", (task_id,)
        )
        result = cursor.fetchone()
        if result is None:
            raise KeyError(f"Task with ID {task_id} not found in the database.")
        return result[0]

    def get_pending_tasks(self) -> List[str]:
        cursor = self.conn.execute(
            "SELECT task_id FROM task_status WHERE status = ?", ("pending",)
        )
        return [row[0] for row in cursor.fetchall()]

    def get_completed_tasks(self) -> List[str]:
        cursor = self.conn.execute(
            "SELECT task_id FROM task_status WHERE status = ?", ("completed",)
        )
        return [row[0] for row in cursor.fetchall()]

    def get_failed_tasks(self) -> List[Tuple[str, str]]:
        cursor = self.conn.execute(
            """
            SELECT ts.task_id, te.error 
            FROM task_status ts 
            JOIN task_error te ON ts.task_id = te.task_id 
            WHERE ts.status = 'failed'
        """
        )
        return [(row[0], json.loads(row[1])) for row in cursor.fetchall()]

    def get_task_result(self, task_id: str) -> Optional[dict]:
        cursor = self.conn.execute(
            "SELECT result FROM task_result WHERE task_id = ?", (task_id,)
        )
        result = cursor.fetchone()
        return json.loads(result[0]) if result else None

    def clear(self):
        with self.conn, self.lock:
            self.conn.execute("DELETE FROM task_definition")
            self.conn.execute("DELETE FROM task_status")
            self.conn.execute("DELETE FROM task_result")
            self.conn.execute("DELETE FROM task_error")

    def clear_tasks(self, task_ids: List[str]):
        with self.conn:
            for task_id in task_ids:
                self.conn.execute(
                    "DELETE FROM task_definition WHERE task_id = ?", (task_id,)
                )
                self.conn.execute(
                    "DELETE FROM task_status WHERE task_id = ?", (task_id,)
                )
                self.conn.execute(
                    "DELETE FROM task_result WHERE task_id = ?", (task_id,)
                )
                self.conn.execute(
                    "DELETE FROM task_error WHERE task_id = ?", (task_id,)
                )

    def dump_all(self) -> Dict[str, Dict[str, dict]]:
        tasks = {}
        statuses = {}
        results = {}
        errors = {}

        cursor = self.conn.execute("SELECT task_id, task_data FROM task_definition")
        tasks = {row[0]: json.loads(row[1]) for row in cursor.fetchall()}

        cursor = self.conn.execute("SELECT task_id, status FROM task_status")
        statuses = {row[0]: row[1] for row in cursor.fetchall()}

        cursor = self.conn.execute("SELECT task_id, result FROM task_result")
        results = {row[0]: json.loads(row[1]) for row in cursor.fetchall()}

        cursor = self.conn.execute("SELECT task_id, error FROM task_error")
        errors = {row[0]: json.loads(row[1]) for row in cursor.fetchall()}

        return {
            "task_definitions": tasks,
            "task_statuses": statuses,
            "task_results": results,
            "task_errors": errors,
        }
