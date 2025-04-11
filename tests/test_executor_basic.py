import pytest
from kofu import LocalThreadedExecutor
from kofu.store import SingleSQLiteTaskStore, Task, TaskState, TaskStatus


class ExampleTask:
    def __init__(self, task_id, url):
        self.task_id = task_id
        self.url = url

    def get_id(self):
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
    task_obj = Task(id="task_1", data={"url": "http://example.com"})
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
            Task(id="task_1", data={"url": "http://example.com"}),
            Task(id="task_2", data={"url": "http://example.org"}),
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
            Task(id="task_1", data={"url": "http://example.com"}),
            Task(id="task_2", data={"url": "http://example.org"}),
        ]
    )
    store.set_many(
        [
            TaskState(
                task=Task(id="task_1", data={"url": "http://example.com"}),
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
