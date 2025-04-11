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


def always_false():
    return False


@pytest.fixture
def store(tmp_path):
    s = SingleSQLiteTaskStore(directory=str(tmp_path))
    yield s
    s.close()


def get_status(store, task_id):
    return store[task_id].status


def get_result(store, task_id):
    return store[task_id].result


def test_single_task_execution(store):
    task = ExampleTask("task_1", "http://example.com")
    task_obj = TaskDefinition(id="task_1", data={"url": "http://example.com"})
    store.put_many([task_obj])

    executor = LocalThreadedExecutor(
        tasks=[task], store=store, max_concurrency=1, stop_all_when=always_false
    )

    assert get_status(store, "task_1") == TaskStatus.PENDING

    executor.run()

    assert get_status(store, "task_1") == TaskStatus.COMPLETED
    assert get_result(store, "task_1") == {"value": "Processed http://example.com"}


def test_multiple_task_execution(store):
    task1 = ExampleTask("task_1", "http://example.com")
    task2 = ExampleTask("task_2", "http://example.org")
    tasks = [task1, task2]

    store.put_many(
        [
            TaskDefinition(id="task_1", data={"url": "http://example.com"}),
            TaskDefinition(id="task_2", data={"url": "http://example.org"}),
        ]
    )

    executor = LocalThreadedExecutor(
        tasks=tasks, store=store, max_concurrency=2, stop_all_when=always_false
    )

    executor.run()

    assert get_status(store, "task_1") == TaskStatus.COMPLETED
    assert get_status(store, "task_2") == TaskStatus.COMPLETED
    assert get_result(store, "task_1") == {"value": "Processed http://example.com"}
    assert get_result(store, "task_2") == {"value": "Processed http://example.org"}


def test_skip_completed_tasks(store):
    task1 = ExampleTask("task_1", "http://example.com")
    task2 = ExampleTask("task_2", "http://example.org")
    tasks = [task1, task2]

    store.put_many(
        [
            TaskDefinition(id="task_1", data={"url": "http://example.com"}),
            TaskDefinition(id="task_2", data={"url": "http://example.org"}),
        ]
    )
    store.set_many(
        [
            TaskState(
                task=TaskDefinition(id="task_1", data={"url": "http://example.com"}),
                status=TaskStatus.COMPLETED,
                result={"html": "<html>Processed</html>"},
            )
        ]
    )

    executor = LocalThreadedExecutor(
        tasks=tasks, store=store, max_concurrency=2, stop_all_when=always_false
    )

    executor.run()

    assert get_status(store, "task_1") == TaskStatus.COMPLETED
    assert get_status(store, "task_2") == TaskStatus.COMPLETED
    assert get_result(store, "task_2") == {"value": "Processed http://example.org"}


def test_put_and_get_many_above_sqlite_limit(store):
    num_tasks = 1050  # just above the 999 SQLite param limit
    tasks = [TaskDefinition(id=f"task-{i}", data={"x": i}) for i in range(num_tasks)]

    # Should not raise or truncate
    store.put_many(tasks)

    # Verify all tasks are inserted
    fetched = store.get_many([t.id for t in tasks])
    assert len(fetched) == num_tasks
    assert set(t.task.id for t in fetched) == set(t.id for t in tasks)


def test_set_many_completed_chunked_above_sqlite_limit(store):
    num_tasks = 1050
    tasks = [TaskDefinition(id=f"task-{i}", data={"x": i}) for i in range(num_tasks)]
    store.put_many(tasks)

    states = [
        TaskState(task=t, status=TaskStatus.COMPLETED, result={"done": True})
        for t in tasks
    ]
    store.set_many(states)

    completed_ids = set(store.get_completed_task_ids())
    assert len(completed_ids) == num_tasks
    assert completed_ids == set(t.id for t in tasks)


def test_delete_many_above_sqlite_limit(store):
    num_tasks = 1050
    tasks = [TaskDefinition(id=f"task-{i}", data={"x": i}) for i in range(num_tasks)]
    store.put_many(tasks)

    to_delete = [t.id for t in tasks[:500]]
    store.delete_many(to_delete)

    remaining_ids = set(t.task.id for t in store)
    assert set(to_delete).isdisjoint(remaining_ids)
    assert len(remaining_ids) == num_tasks - 500
