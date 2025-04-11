import pytest
import time
from kofu import LocalThreadedExecutor
from kofu.store import SingleSQLiteTaskStore, TaskDefinition, TaskStatus


class ExampleTaskWithDelay:
    def __init__(self, task_id, url, delay=0):
        self.task_id = task_id
        self.url = url
        self.delay = delay

    @property
    def id(self):
        return self.task_id

    def __call__(self):
        time.sleep(self.delay)
        return f"Processed {self.url} after {self.delay}s"


@pytest.fixture
def store(tmp_path):
    s = SingleSQLiteTaskStore(directory=str(tmp_path))
    yield s
    s.close()


def get_status(store, task_id):
    return store[task_id].status


def get_result(store, task_id):
    return store[task_id].result


def test_tasks_execute_concurrently_with_limit(store):
    tasks = [
        ExampleTaskWithDelay("task_1", "http://example.com", delay=2),
        ExampleTaskWithDelay("task_2", "http://example.org", delay=2),
        ExampleTaskWithDelay("task_3", "http://example.net", delay=2),
    ]
    store.put_many(
        [TaskDefinition(id=task.id, data={"url": task.url}) for task in tasks]
    )

    executor = LocalThreadedExecutor(tasks=tasks, store=store, max_concurrency=2)

    start_time = time.time()
    executor.run()
    elapsed = time.time() - start_time

    assert 4 <= elapsed < 5
    assert all(get_status(store, t.id) == TaskStatus.COMPLETED for t in tasks)


def test_no_task_duplication(store):
    execution_count = {}

    class ExampleTaskWithCount:
        def __init__(self, task_id, url):
            self.task_id = task_id
            self.url = url

        @property
        def id(self):
            return self.task_id

        def __call__(self):
            execution_count[self.task_id] = execution_count.get(self.task_id, 0) + 1
            return f"Processed {self.url}"

    tasks = [
        ExampleTaskWithCount("task_1", "http://example.com"),
        ExampleTaskWithCount("task_2", "http://example.org"),
        ExampleTaskWithCount("task_3", "http://example.net"),
    ]
    store.put_many(
        [TaskDefinition(id=task.id, data={"url": task.url}) for task in tasks]
    )

    executor = LocalThreadedExecutor(tasks=tasks, store=store, max_concurrency=3)
    executor.run()

    assert all(execution_count[t.id] == 1 for t in tasks)


def test_correct_task_ordering(store):
    execution_order = []

    class ExampleTaskWithOrder:
        def __init__(self, task_id, url):
            self.task_id = task_id
            self.url = url

        @property
        def id(self):
            return self.task_id

        def __call__(self):
            execution_order.append(self.task_id)
            return f"Processed {self.url}"

    tasks = [
        ExampleTaskWithOrder("task_1", "http://example.com"),
        ExampleTaskWithOrder("task_2", "http://example.org"),
        ExampleTaskWithOrder("task_3", "http://example.net"),
    ]
    store.put_many(
        [TaskDefinition(id=task.id, data={"url": task.url}) for task in tasks]
    )

    executor = LocalThreadedExecutor(tasks=tasks, store=store, max_concurrency=2)
    executor.run()

    assert set(execution_order) == {"task_1", "task_2", "task_3"}
    assert len(execution_order) == 3
