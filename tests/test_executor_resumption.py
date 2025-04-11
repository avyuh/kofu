import pytest
from kofu import LocalThreadedExecutor
from kofu.store import SingleSQLiteTaskStore, TaskDefinition, TaskState, TaskStatus


class ExampleTask:
    def __init__(self, task_id, url):
        self.task_id = task_id
        self.url = url

    @property
    def id(self):
        return self.task_id

    def __call__(self):
        return f"Processed {self.url}"


@pytest.fixture
def store(tmp_path):
    s = SingleSQLiteTaskStore(directory=str(tmp_path))
    yield s
    s.close()


def test_resumption_after_some_tasks_completed(store):
    tasks = [
        ExampleTask("task_1", "http://example.com"),
        ExampleTask("task_2", "http://example.org"),
        ExampleTask("task_3", "http://example.net"),
    ]
    store.put_many([TaskDefinition(id=t.id, data={"url": t.url}) for t in tasks])

    # Mark task_1 and task_2 as completed
    store.set_many(
        [
            TaskState(
                task=TaskDefinition(id="task_1", data={"url": "http://example.com"}),
                status=TaskStatus.COMPLETED,
                result={"html": "<html>Processed</html>"},
            ),
            TaskState(
                task=TaskDefinition(id="task_2", data={"url": "http://example.org"}),
                status=TaskStatus.COMPLETED,
                result={"html": "<html>Processed</html>"},
            ),
        ]
    )

    executor = LocalThreadedExecutor(tasks=tasks, store=store, max_concurrency=2)
    executor.run()

    assert store["task_1"].status == TaskStatus.COMPLETED
    assert store["task_2"].status == TaskStatus.COMPLETED
    assert store["task_3"].status == TaskStatus.COMPLETED
    assert store["task_3"].result["value"] == "Processed http://example.net"


def test_resumption_after_incomplete_execution(store):
    tasks = [
        ExampleTask("task_1", "http://example.com"),
        ExampleTask("task_2", "http://example.org"),
        ExampleTask("task_3", "http://example.net"),
    ]
    store.put_many([TaskDefinition(id=t.id, data={"url": t.url}) for t in tasks])

    store.set_many(
        [
            TaskState(
                task=TaskDefinition(id="task_1", data={"url": "http://example.com"}),
                status=TaskStatus.COMPLETED,
                result={"html": "<html>Processed</html>"},
            )
        ]
    )

    executor = LocalThreadedExecutor(tasks=tasks, store=store, max_concurrency=2)
    executor.run()

    assert store["task_1"].status == TaskStatus.COMPLETED
    assert store["task_2"].status == TaskStatus.COMPLETED
    assert store["task_3"].status == TaskStatus.COMPLETED


def test_memory_persistence_for_task_statuses(store):
    tasks = [
        ExampleTask("task_1", "http://example.com"),
        ExampleTask("task_2", "http://example.org"),
    ]
    store.put_many([TaskDefinition(id=t.id, data={"url": t.url}) for t in tasks])

    executor = LocalThreadedExecutor(tasks=tasks, store=store, max_concurrency=2)
    executor.run()

    assert store["task_1"].status == TaskStatus.COMPLETED
    assert store["task_2"].status == TaskStatus.COMPLETED

    # Mark task_1 as failed manually
    store.set_many(
        [
            TaskState(
                task=TaskDefinition(id="task_1", data={"url": "http://example.com"}),
                status=TaskStatus.FAILED,
                error="TimeoutError",
            )
        ]
    )

    assert store["task_1"].status == TaskStatus.FAILED
    assert store["task_1"].error == "TimeoutError"
