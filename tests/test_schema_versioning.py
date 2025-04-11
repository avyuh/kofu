import sqlite3
import pytest
from packaging.version import Version
from kofu.store.sqlite_store import (
    SingleSQLiteTaskStore,
    DatabaseSchemaError,
    KOFU_DB_BRAND,
    CURRENT_SCHEMA_VERSION,
)


def test_fresh_store_initializes_schema(tmp_path):
    store = SingleSQLiteTaskStore(directory=str(tmp_path))
    conn = sqlite3.connect(tmp_path / "tasks.db")
    brand = conn.execute("SELECT value FROM meta WHERE key = 'brand'").fetchone()
    version = conn.execute(
        "SELECT value FROM meta WHERE key = 'schema_version'"
    ).fetchone()

    assert brand[0] == KOFU_DB_BRAND
    assert Version(version[0]) == CURRENT_SCHEMA_VERSION
    store.close()


def test_rejects_db_with_missing_brand(tmp_path):
    # meta table exists with version but no brand
    db_path = tmp_path / "tasks.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta (key, value) VALUES ('schema_version', '1.0');
    """
    )
    conn.commit()
    conn.close()

    with pytest.raises(DatabaseSchemaError, match="missing 'brand'"):
        SingleSQLiteTaskStore(directory=str(tmp_path))


def test_rejects_db_with_wrong_brand(tmp_path):
    db_path = tmp_path / "tasks.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta (key, value) VALUES ('brand', 'not-kofu');
        INSERT INTO meta (key, value) VALUES ('schema_version', '1.0');
    """
    )
    conn.commit()
    conn.close()

    with pytest.raises(DatabaseSchemaError, match="brand mismatch"):
        SingleSQLiteTaskStore(directory=str(tmp_path))


def test_warns_on_minor_version_mismatch(tmp_path, caplog):
    db_path = tmp_path / "tasks.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        f"""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta (key, value) VALUES ('brand', '{KOFU_DB_BRAND}');
        INSERT INTO meta (key, value) VALUES ('schema_version', '1.9');
        CREATE TABLE task_definitions (task_id TEXT PRIMARY KEY, task_data BLOB NOT NULL);
        CREATE TABLE completed_tasks (task_id TEXT PRIMARY KEY, result BLOB);
        CREATE TABLE failed_tasks (task_id TEXT PRIMARY KEY, error TEXT);
    """
    )
    conn.commit()
    conn.close()

    with caplog.at_level("WARNING"):
        store = SingleSQLiteTaskStore(directory=str(tmp_path))
        store.close()

    assert "not explicitly supported" in caplog.text


def test_rejects_incompatible_major_version(tmp_path):
    db_path = tmp_path / "tasks.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        f"""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta (key, value) VALUES ('brand', '{KOFU_DB_BRAND}');
        INSERT INTO meta (key, value) VALUES ('schema_version', '2.0');
        CREATE TABLE task_definitions (task_id TEXT PRIMARY KEY, task_data BLOB NOT NULL);
        CREATE TABLE completed_tasks (task_id TEXT PRIMARY KEY, result BLOB);
        CREATE TABLE failed_tasks (task_id TEXT PRIMARY KEY, error TEXT);
    """
    )
    conn.commit()
    conn.close()

    with pytest.raises(DatabaseSchemaError, match="Incompatible schema version"):
        SingleSQLiteTaskStore(directory=str(tmp_path))
