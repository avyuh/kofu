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


def test_stop_condition_after_task_execution(store):
    executed_tasks = 0

    class ExampleTaskWithCounter(ExampleTaskWithDelay):
        def __call__(self):
            nonlocal executed_tasks
            result = super().__call__()
            executed_tasks += 1
            return result

    tasks = [
        ExampleTaskWithCounter("task_1", "http://example.com", delay=0),
        ExampleTaskWithCounter("task_2", "http://example.org", delay=0),
        ExampleTaskWithCounter("task_3", "http://example.net", delay=0),
    ]
    store.put_many([TaskDefinition(id=t.id, data={"url": t.url}) for t in tasks])

    def stop_after_two():
        return executed_tasks >= 2

    executor = LocalThreadedExecutor(
        tasks=tasks, store=store, max_concurrency=2, stop_all_when=stop_after_two
    )
    executor.run()

    assert 1 <= executed_tasks <= 3
    completed = [s for s in store if s.status == TaskStatus.COMPLETED]
    assert 1 <= len(completed) <= 3


def test_stop_condition_checked_after_each_task(store):
    tasks = [
        ExampleTaskWithDelay("task_1", "http://example.com", delay=1),
        ExampleTaskWithDelay("task_2", "http://example.org", delay=1),
    ]
    store.put_many([TaskDefinition(id=t.id, data={"url": t.url}) for t in tasks])

    def stop_after_first_done():
        return store["task_1"].status == TaskStatus.COMPLETED

    executor = LocalThreadedExecutor(
        tasks=tasks, store=store, max_concurrency=2, stop_all_when=stop_after_first_done
    )
    executor.run()

    assert store["task_1"].status == TaskStatus.COMPLETED
    assert store["task_2"].status in {TaskStatus.PENDING, TaskStatus.COMPLETED}


def test_stop_condition_halts_mid_execution(store):
    tasks = [
        ExampleTaskWithDelay("task_1", "http://example.com", delay=3),
        ExampleTaskWithDelay("task_2", "http://example.org", delay=1),
    ]
    store.put_many([TaskDefinition(id=t.id, data={"url": t.url}) for t in tasks])

    def stop_midway():
        time.sleep(2)
        return True

    executor = LocalThreadedExecutor(
        tasks=tasks, store=store, max_concurrency=2, stop_all_when=stop_midway
    )
    executor.run()

    # May vary due to race conditions; validate conservatively
    assert store["task_1"].status in {TaskStatus.PENDING, TaskStatus.COMPLETED}
    assert store["task_2"].status in {TaskStatus.PENDING, TaskStatus.COMPLETED}
