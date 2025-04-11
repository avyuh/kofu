import os
import sqlite3
import json
import zlib
import threading
import time
import logging
import random  # Fixed missing import
from contextlib import contextmanager
from typing import Dict, Iterator, Optional, Any, Union

from .task_store import TaskStore
from .task_state import TaskDefinition, TaskState, TaskStatus

from packaging.version import Version, InvalidVersion

# Configure logging
logger = logging.getLogger(__name__)

KOFU_DB_BRAND = "kofu-a9e82f39-4262-425c-aee1-6a0432601c9f"

SUPPORTED_SCHEMA_VERSIONS = [Version("1.0")]
CURRENT_SCHEMA_VERSION = SUPPORTED_SCHEMA_VERSIONS[-1]

SQLITE_PARAM_LIMIT = 999


class DatabaseSchemaError(sqlite3.Error):
    """Raised when schema version is missing or incompatible."""


# SQLite optimal configuration
DEFAULT_PRAGMAS = {
    "journal_mode": "WAL",  # Write-Ahead Logging for concurrency
    "synchronous": "NORMAL",  # Balance between safety and speed
    "cache_size": 8000,  # 8MB page cache
    "mmap_size": 67108864,  # 64MB memory mapping
    "auto_vacuum": "INCREMENTAL",  # Prevent file bloat
    "busy_timeout": 5000,  # 5-second timeout
    "temp_store": "MEMORY",  # Store temp tables in memory
    "foreign_keys": "ON",  # Enforce referential integrity
}

# Schema creation with optimized ordering (BLOBs at end)
CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS task_definitions (
    task_id TEXT PRIMARY KEY,
    task_data BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS completed_tasks (
    task_id TEXT PRIMARY KEY,
    result BLOB,
    FOREIGN KEY(task_id) REFERENCES task_definitions(task_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS failed_tasks (
    task_id TEXT PRIMARY KEY,
    error TEXT,
    FOREIGN KEY(task_id) REFERENCES task_definitions(task_id) ON DELETE CASCADE
);
"""

CREATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_completed_tasks ON completed_tasks(task_id);
CREATE INDEX IF NOT EXISTS idx_failed_tasks ON failed_tasks(task_id);
"""


class Serializer:
    """Interface for data serialization."""

    def serialize(self, obj) -> Optional[bytes]:
        """Convert object to bytes."""
        raise NotImplementedError

    def deserialize(self, data: Optional[bytes]) -> Any:
        """Convert bytes back to object."""
        raise NotImplementedError


class JSONSerializer(Serializer):
    """JSON serializer with compression for efficiency."""

    def __init__(self, compression_level: int = 1):
        """Initialize with optional compression level.

        Args:
            compression_level: Zlib compression level (0-9, 0=none, 9=max)
        """
        self.compression_level = compression_level

    def serialize(self, obj) -> Optional[bytes]:
        """Serialize and compress object.

        Returns None if obj is None, otherwise compressed JSON bytes.
        """
        if obj is None:
            return None
        json_str = json.dumps(obj)
        return zlib.compress(json_str.encode("utf-8"), self.compression_level)

    def deserialize(self, data: Optional[bytes]) -> Any:
        """Decompress and deserialize data.

        Returns None if data is None, otherwise deserialized object.
        """
        if data is None:
            return None
        json_str = zlib.decompress(data).decode("utf-8")
        return json.loads(json_str)


class SingleSQLiteTaskStore(TaskStore):
    """High-performance SQLite-based task store with normalized tables."""

    def __init__(
        self,
        directory: str,
        serializer: Optional[Serializer] = None,
        timeout: float = 60,
        pragmas: Optional[Dict[str, Union[str, int]]] = None,
        max_retries: int = 3,
        max_retry_delay_sec: float = 3.0,
    ):
        """Initialize SQLite task store.

        Args:
            directory: Path to store SQLite database
            serializer: Custom serializer (default: JSONSerializer)
            timeout: SQLite connection timeout in seconds
            pragmas: Dict of SQLite PRAGMA settings to override defaults
            max_retries: Maximum retries for locked database operations
            max_retry_delay_sec: Maximum sleep time (in seconds) between retries to prevent very long waits.
        """
        self.directory = os.path.abspath(directory)
        os.makedirs(self.directory, exist_ok=True)

        self.db_path = os.path.join(self.directory, "tasks.db")
        self.serializer = serializer or JSONSerializer()
        self.timeout = timeout
        self._pragmas = {**DEFAULT_PRAGMAS, **(pragmas or {})}
        self._local = threading.local()
        self._max_retries = max_retries
        self._max_retry_delay_sec = max_retry_delay_sec

        # Initialize database
        self._setup_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local SQLite connection with retry.

        Returns:
            sqlite3.Connection: Thread-local database connection
        """
        conn = getattr(self._local, "connection", None)

        # Check if connection is valid
        if conn is not None:
            try:
                # Quick test query to verify connection
                conn.execute("SELECT 1").fetchone()
                return conn
            except sqlite3.Error:
                # Connection is bad, close it
                try:
                    conn.close()
                except Exception:
                    pass
                self._local.connection = None

        # Create new connection
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            isolation_level=None,  # Manual transaction management
            check_same_thread=False,  # Thread-local conn = thread safety
        )

        # Enable optimized row factory
        conn.row_factory = sqlite3.Row

        # Apply all pragmas
        for pragma, value in self._pragmas.items():
            conn.execute(f"PRAGMA {pragma} = {value}")

        # Store connection
        self._local.connection = conn
        self._local.statements = {}  # Reset prepared statements
        return conn

    def _execute_with_retry(self, func, *args, **kwargs):
        """Execute a function with retry logic for database locks.

        Applies exponential backoff with jitter, capped by max_retry_delay_sec.

        Args:
            func: Function to execute
            *args, **kwargs: Arguments for the function

        Returns:
            Result of the function

        Raises:
            sqlite3.Error: If retries are exhausted or a non-lock error occurs.
        """
        retry_count = 0
        retry_delay = 0.01  # Initial delay in seconds

        while True:
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and retry_count < self._max_retries:
                    retry_count += 1

                    # Calculate exponential backoff with jitter
                    jitter = random.random() * 0.1  # 0-10% jitter
                    base_sleep_time = retry_delay * (1 + jitter)

                    # Apply the maximum sleep time cap
                    sleep_time = min(base_sleep_time, self._max_retry_delay_sec)

                    # Log retry attempt with actual sleep time
                    logger.debug(
                        f"Database locked, retrying ({retry_count}/{self._max_retries}) "
                        f"after sleeping for {sleep_time:.3f}s (base delay: {retry_delay:.3f}s)"
                    )

                    time.sleep(sleep_time)

                    retry_delay *= 2  # Exponential backoff
                else:
                    # Log the final error before re-raising
                    if "database is locked" in str(e):
                        logger.error(
                            f"SQLite 'database is locked' error persisted after "
                            f"{self._max_retries} retries. Giving up."
                        )
                    else:
                        logger.error(f"SQLite operational error: {e}")
                    raise  # Re-raise the original or a new informative error
            except sqlite3.Error as e:
                # Catch other SQLite errors separately if needed, or just log and re-raise
                logger.error(f"Unhandled SQLite error during execution: {e}")
                raise

    def _setup_database(self) -> None:
        conn = self._get_connection()

        # Always attempt to create tables/indexes (idempotent)
        conn.executescript(CREATE_TABLES_SQL)
        conn.executescript(CREATE_INDEXES_SQL)

        with conn:
            # Check for 'brand' in the meta table
            brand_row = conn.execute(
                "SELECT value FROM meta WHERE key = 'brand'"
            ).fetchone()

            if brand_row:
                # Brand exists — verify it's the correct one
                if brand_row[0] != KOFU_DB_BRAND:
                    raise DatabaseSchemaError(
                        f"Database brand mismatch: expected '{KOFU_DB_BRAND}', found '{brand_row[0]}'"
                    )
            else:
                # No brand — is this a new DB, or a corrupted/legacy one?
                meta_keys = conn.execute("SELECT key FROM meta").fetchall()

                if meta_keys:
                    raise DatabaseSchemaError(
                        "Refusing to initialize existing database: missing 'brand' entry in 'meta' table.\n"
                        f"This file likely wasn't created by KOFU, or it may be corrupted, or from an older version.\n"
                        f"To fix: delete the database at '{self.db_path}' and let KOFU recreate it, or migrate it manually if needed."
                    )

                # Brand new database — safe to initialize
                conn.execute(
                    "INSERT INTO meta (key, value) VALUES (?, ?)",
                    ("brand", KOFU_DB_BRAND),
                )
                conn.execute(
                    "INSERT INTO meta (key, value) VALUES (?, ?)",
                    ("schema_version", str(CURRENT_SCHEMA_VERSION)),
                )
                logger.info("Initialized brand new KOFU database.")

            # Validate version (fresh or existing)
            version = self._get_schema_version(conn)

            if version.major != CURRENT_SCHEMA_VERSION.major:
                raise DatabaseSchemaError(
                    f"Incompatible schema version {version}. "
                    f"Expected major version {CURRENT_SCHEMA_VERSION.major}."
                )
            elif version not in SUPPORTED_SCHEMA_VERSIONS:
                logger.warning(
                    f"Schema version {version} not explicitly supported. Proceeding cautiously."
                )
            else:
                logger.info(f"Schema version {version} is supported.")

    def _get_schema_version(self, conn: sqlite3.Connection) -> Version:
        try:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            if not row:
                raise DatabaseSchemaError(
                    "Missing 'schema_version' entry in meta table."
                )
            return Version(row[0])
        except InvalidVersion as e:
            raise DatabaseSchemaError(f"Invalid schema version format: {e}")
        except Exception as e:
            raise DatabaseSchemaError(f"Error reading schema version: {e}")

    @contextmanager
    def atomic(self) -> Iterator[None]:
        """Context manager for transaction handling with retry and savepoints.

        Provides nested transaction support through savepoints.
        """
        conn = self._get_connection()

        # Track if we're in a nested transaction
        in_transaction = (
            hasattr(self._local, "transaction_depth")
            and self._local.transaction_depth > 0
        )

        if not hasattr(self._local, "transaction_depth"):
            self._local.transaction_depth = 0

        savepoint_id = None

        try:
            if in_transaction:
                # Create a savepoint for nested transaction
                savepoint_id = f"sp_{threading.get_ident()}_{time.time_ns()}"
                self._execute_with_retry(conn.execute, f"SAVEPOINT {savepoint_id}")
                self._local.transaction_depth += 1
            else:
                # Start a new transaction with retry logic
                self._execute_with_retry(conn.execute, "BEGIN IMMEDIATE")
                self._local.transaction_depth = 1

            # Yield control to the context block
            yield

            # Commit changes
            if self._local.transaction_depth == 1:
                self._execute_with_retry(conn.execute, "COMMIT")
                self._local.transaction_depth = 0
            elif savepoint_id:
                self._execute_with_retry(conn.execute, f"RELEASE {savepoint_id}")
                self._local.transaction_depth -= 1

        except Exception:
            # Rollback on error
            if self._local.transaction_depth == 1:
                self._execute_with_retry(conn.execute, "ROLLBACK")
                self._local.transaction_depth = 0
            elif savepoint_id:
                self._execute_with_retry(conn.execute, f"ROLLBACK TO {savepoint_id}")
                self._local.transaction_depth -= 1
            raise

    def close(self) -> None:
        """Close database connection and clean up resources."""
        conn = getattr(self._local, "connection", None)
        if conn:
            try:
                # Perform a WAL checkpoint to keep file size in check
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.close()
            except Exception as e:
                logger.warning(f"Error closing SQLite connection: {e}")
            finally:
                self._local.connection = None
                if hasattr(self._local, "statements"):
                    delattr(self._local, "statements")
                if hasattr(self._local, "transaction_depth"):
                    delattr(self._local, "transaction_depth")

    def __enter__(self) -> "SingleSQLiteTaskStore":
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close connection on context exit."""
        self.close()

    def __del__(self) -> None:
        """Ensure connection is closed on deletion."""
        self.close()

    # TaskStore interface implementation

    def get_many(self, task_ids: list[str]) -> list[TaskState]:
        """Retrieve multiple tasks by ID.

        Args:
            task_ids: List of task IDs to retrieve

        Returns:
            List of TaskState objects for existing task IDs
        """
        if not task_ids:
            return []

        results = []
        conn = self._get_connection()

        for chunk in _chunked(task_ids, SQLITE_PARAM_LIMIT):
            placeholders = ",".join("?" for _ in chunk)
            query = f"""
                SELECT d.task_id, d.task_data, 
                    c.result IS NOT NULL as is_completed,
                    c.result, f.error
                FROM task_definitions d
                LEFT JOIN completed_tasks c ON d.task_id = c.task_id
                LEFT JOIN failed_tasks f ON d.task_id = f.task_id
                WHERE d.task_id IN ({placeholders})
                """
            cursor = self._execute_with_retry(conn.execute, query, chunk)
            for row in cursor:
                task_id, task_data_blob, is_completed, result_blob, error = row
                task_data = self.serializer.deserialize(task_data_blob)
                task = TaskDefinition(id=task_id, data=task_data)
                if is_completed:
                    status = TaskStatus.COMPLETED
                    result = self.serializer.deserialize(result_blob)
                    error = None
                elif error is not None:
                    status = TaskStatus.FAILED
                    result = None
                else:
                    status = TaskStatus.PENDING
                    result = None
                results.append(
                    TaskState(task=task, status=status, result=result, error=error)
                )
        return results

    def put_many(self, tasks: list[TaskDefinition]) -> None:
        """Create multiple tasks with PENDING status.

        Args:
            tasks: List of Task objects to store
        """
        if not tasks:
            return

        task_data = [(task.id, self.serializer.serialize(task.data)) for task in tasks]
        task_ids = [task.id for task in tasks]

        with self.atomic():
            conn = self._get_connection()
            for chunk in _chunked(task_data, SQLITE_PARAM_LIMIT // 2):
                self._execute_with_retry(
                    conn.executemany,
                    "INSERT OR REPLACE INTO task_definitions (task_id, task_data) VALUES (?, ?)",
                    chunk,
                )
            for chunk in _chunked(task_ids, SQLITE_PARAM_LIMIT):
                placeholders = ",".join("?" for _ in chunk)
                self._execute_with_retry(
                    conn.execute,
                    f"DELETE FROM completed_tasks WHERE task_id IN ({placeholders})",
                    chunk,
                )
                self._execute_with_retry(
                    conn.execute,
                    f"DELETE FROM failed_tasks WHERE task_id IN ({placeholders})",
                    chunk,
                )

    def set_many(self, states: list[TaskState]) -> None:
        """Update states for multiple tasks.

        Args:
            states: List of TaskState objects to update
        """
        if not states:
            return

        completed = []
        failed = []
        pending = []
        all_task_ids = set()

        for state in states:
            all_task_ids.add(state.task.id)
            if state.status == TaskStatus.COMPLETED:
                completed.append(
                    (state.task.id, self.serializer.serialize(state.result))
                )
            elif state.status == TaskStatus.FAILED:
                failed.append((state.task.id, state.error))
            else:
                pending.append(state.task.id)

        with self.atomic():
            conn = self._get_connection()

            # Ensure all task definitions exist
            missing_task_ids = set()
            for chunk in _chunked(list(all_task_ids), SQLITE_PARAM_LIMIT):
                placeholders = ",".join("?" for _ in chunk)
                cursor = self._execute_with_retry(
                    conn.execute,
                    f"SELECT task_id FROM task_definitions WHERE task_id IN ({placeholders})",
                    chunk,
                )
                existing = {row[0] for row in cursor}
                missing = set(chunk) - existing
                missing_task_ids.update(missing)

            if missing_task_ids:
                missing_tasks = [
                    (state.task.id, self.serializer.serialize(state.task.data))
                    for state in states
                    if state.task.id in missing_task_ids
                ]
                for chunk in _chunked(missing_tasks, SQLITE_PARAM_LIMIT // 2):
                    self._execute_with_retry(
                        conn.executemany,
                        "INSERT INTO task_definitions (task_id, task_data) VALUES (?, ?)",
                        chunk,
                    )

            # Completed tasks
            if completed:
                completed_ids = [task_id for task_id, _ in completed]
                for chunk in _chunked(completed_ids, SQLITE_PARAM_LIMIT):
                    placeholders = ",".join("?" for _ in chunk)
                    self._execute_with_retry(
                        conn.execute,
                        f"DELETE FROM failed_tasks WHERE task_id IN ({placeholders})",
                        chunk,
                    )
                for chunk in _chunked(completed, SQLITE_PARAM_LIMIT // 2):
                    self._execute_with_retry(
                        conn.executemany,
                        "INSERT OR REPLACE INTO completed_tasks (task_id, result) VALUES (?, ?)",
                        chunk,
                    )

            # Failed tasks
            if failed:
                failed_ids = [task_id for task_id, _ in failed]
                for chunk in _chunked(failed_ids, SQLITE_PARAM_LIMIT):
                    placeholders = ",".join("?" for _ in chunk)
                    self._execute_with_retry(
                        conn.execute,
                        f"DELETE FROM completed_tasks WHERE task_id IN ({placeholders})",
                        chunk,
                    )
                for chunk in _chunked(failed, SQLITE_PARAM_LIMIT // 2):
                    self._execute_with_retry(
                        conn.executemany,
                        "INSERT OR REPLACE INTO failed_tasks (task_id, error) VALUES (?, ?)",
                        chunk,
                    )

            # Pending tasks
            if pending:
                for chunk in _chunked(pending, SQLITE_PARAM_LIMIT):
                    placeholders = ",".join("?" for _ in chunk)
                    self._execute_with_retry(
                        conn.execute,
                        f"DELETE FROM completed_tasks WHERE task_id IN ({placeholders})",
                        chunk,
                    )
                    self._execute_with_retry(
                        conn.execute,
                        f"DELETE FROM failed_tasks WHERE task_id IN ({placeholders})",
                        chunk,
                    )

    def delete_many(self, task_ids: list[str]) -> None:
        """Delete multiple tasks by ID.

        Args:
            task_ids: List of task IDs to delete
        """
        if not task_ids:
            return

        with self.atomic():
            conn = self._get_connection()
            for chunk in _chunked(task_ids, SQLITE_PARAM_LIMIT):
                placeholders = ",".join("?" for _ in chunk)
                self._execute_with_retry(
                    conn.execute,
                    f"DELETE FROM task_definitions WHERE task_id IN ({placeholders})",
                    chunk,
                )

    def __iter__(self) -> Iterator[TaskState]:
        """Iterate through all tasks.

        Returns:
            Iterator yielding TaskState objects
        """
        # This query efficiently combines tables with optimal LEFT JOINs
        query = """
        SELECT d.task_id, d.task_data, 
               c.result IS NOT NULL as is_completed,
               c.result, f.error
        FROM task_definitions d
        LEFT JOIN completed_tasks c ON d.task_id = c.task_id
        LEFT JOIN failed_tasks f ON d.task_id = f.task_id
        """

        conn = self._get_connection()

        # Use server-side cursor with fetchmany for memory efficiency
        cursor = self._execute_with_retry(conn.execute, query)
        batch_size = 100

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            for row in rows:
                task_id, task_data_blob, is_completed, result_blob, error = row

                # Deserialize task data
                task_data = self.serializer.deserialize(task_data_blob)
                task = TaskDefinition(id=task_id, data=task_data)

                # Determine status and results
                if is_completed:
                    status = TaskStatus.COMPLETED
                    result = self.serializer.deserialize(result_blob)
                    error = None
                elif error is not None:
                    status = TaskStatus.FAILED
                    result = None
                else:
                    status = TaskStatus.PENDING
                    result = None

                yield TaskState(task=task, status=status, result=result, error=error)

    def __len__(self) -> int:
        """Return the number of tasks.

        Returns:
            Total count of tasks
        """
        conn = self._get_connection()
        cursor = self._execute_with_retry(
            conn.execute, "SELECT COUNT(*) FROM task_definitions"
        )
        return cursor.fetchone()[0]

    def clear(self) -> None:
        """Delete all tasks."""
        with self.atomic():
            conn = self._get_connection()
            # Delete from definitions first (cascades to status tables)
            self._execute_with_retry(conn.execute, "DELETE FROM task_definitions")

    def query(self, status: Optional[TaskStatus] = None) -> Iterator[TaskState]:
        """Query tasks by status.

        Args:
            status: Optional TaskStatus to filter by

        Returns:
            Iterator yielding TaskState objects matching the query
        """
        conn = self._get_connection()

        if status is None:
            # Return all tasks
            yield from self.__iter__()
            return

        elif status == TaskStatus.COMPLETED:
            # Optimized query for completed tasks with direct JOIN
            query = """
            SELECT d.task_id, d.task_data, c.result
            FROM task_definitions d
            JOIN completed_tasks c ON d.task_id = c.task_id
            """
            cursor = self._execute_with_retry(conn.execute, query)

            for row in cursor:
                task_id, task_data_blob, result_blob = row
                task_data = self.serializer.deserialize(task_data_blob)
                result = self.serializer.deserialize(result_blob)

                task = TaskDefinition(id=task_id, data=task_data)
                yield TaskState(
                    task=task, status=TaskStatus.COMPLETED, result=result, error=None
                )

        elif status == TaskStatus.FAILED:
            # Optimized query for failed tasks with direct JOIN
            query = """
            SELECT d.task_id, d.task_data, f.error
            FROM task_definitions d
            JOIN failed_tasks f ON d.task_id = f.task_id
            """
            cursor = self._execute_with_retry(conn.execute, query)

            for row in cursor:
                task_id, task_data_blob, error = row
                task_data = self.serializer.deserialize(task_data_blob)

                task = TaskDefinition(id=task_id, data=task_data)
                yield TaskState(
                    task=task, status=TaskStatus.FAILED, result=None, error=error
                )

        else:  # TaskStatus.PENDING
            # Optimized query for pending tasks (using NOT EXISTS)
            query = """
            SELECT d.task_id, d.task_data
            FROM task_definitions d
            WHERE NOT EXISTS (SELECT 1 FROM completed_tasks c WHERE c.task_id = d.task_id)
              AND NOT EXISTS (SELECT 1 FROM failed_tasks f WHERE f.task_id = d.task_id)
            """
            cursor = self._execute_with_retry(conn.execute, query)

            for row in cursor:
                task_id, task_data_blob = row
                task_data = self.serializer.deserialize(task_data_blob)

                task = TaskDefinition(id=task_id, data=task_data)
                yield TaskState(
                    task=task, status=TaskStatus.PENDING, result=None, error=None
                )

    # TaskStore interface for resetting failed tasks

    def reset_failed(self) -> int:
        """Reset all failed tasks to pending status.

        Returns:
            Number of tasks reset
        """
        conn = self._get_connection()

        with self.atomic():
            # Get count first
            cursor = self._execute_with_retry(
                conn.execute, "SELECT COUNT(*) FROM failed_tasks"
            )
            count = cursor.fetchone()[0]

            # Delete all entries from failed_tasks
            self._execute_with_retry(conn.execute, "DELETE FROM failed_tasks")

        return count

    # Enhanced functionality beyond the base interface

    def vacuum(self) -> None:
        """Optimize database storage."""
        conn = self._get_connection()
        self._execute_with_retry(conn.execute, "VACUUM")

    def integrity_check(self) -> list[str]:
        """Run SQLite integrity check.

        Returns:
            List of integrity check messages ("ok" if all is well)
        """
        conn = self._get_connection()
        cursor = self._execute_with_retry(conn.execute, "PRAGMA integrity_check")
        return [row[0] for row in cursor]

    def get_pending_task_ids(self) -> list[str]:
        """Get IDs of all pending tasks.

        Returns:
            List of task IDs with PENDING status
        """
        # Optimized query that avoids creating TaskState objects
        query = """
        SELECT d.task_id
        FROM task_definitions d
        WHERE NOT EXISTS (SELECT 1 FROM completed_tasks c WHERE c.task_id = d.task_id)
          AND NOT EXISTS (SELECT 1 FROM failed_tasks f WHERE f.task_id = d.task_id)
        """

        conn = self._get_connection()
        cursor = self._execute_with_retry(conn.execute, query)
        return [row[0] for row in cursor]

    def get_completed_task_ids(self) -> list[str]:
        """Get IDs of all completed tasks.

        Returns:
            List of task IDs with COMPLETED status
        """
        conn = self._get_connection()
        cursor = self._execute_with_retry(
            conn.execute, "SELECT task_id FROM completed_tasks"
        )
        return [row[0] for row in cursor]

    def get_failed_task_ids(self) -> list[str]:
        """Get IDs of all failed tasks.

        Returns:
            List of task IDs with FAILED status
        """
        conn = self._get_connection()
        cursor = self._execute_with_retry(
            conn.execute, "SELECT task_id FROM failed_tasks"
        )
        return [row[0] for row in cursor]

    def task_exists(self, task_id: str) -> bool:
        """Check if a task exists in the store.

        Args:
            task_id: Task ID to check

        Returns:
            True if task exists, False otherwise
        """
        conn = self._get_connection()
        cursor = self._execute_with_retry(
            conn.execute,
            "SELECT 1 FROM task_definitions WHERE task_id = ? LIMIT 1",
            (task_id,),
        )
        return cursor.fetchone() is not None

    def checkpoint(self) -> None:
        """Force a WAL checkpoint to keep database size in check."""
        conn = self._get_connection()
        self._execute_with_retry(conn.execute, "PRAGMA wal_checkpoint(FULL)")


def _chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]
