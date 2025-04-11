import pytest
from kofu import LocalThreadedExecutor
from kofu.store import SingleSQLiteTaskStore, TaskDefinition, TaskStatus


class ExampleTaskWithException:
    def __init__(self, task_id, url, should_fail=False):
        self.task_id = task_id
        self.url = url
        self.should_fail = should_fail

    @property
    def id(self):
        return self.task_id

    def __call__(self):
        if self.should_fail:
            raise Exception(f"Task {self.task_id} failed")
        return f"Processed {self.url}"


@pytest.fixture
def store(tmp_path):
    s = SingleSQLiteTaskStore(directory=str(tmp_path))
    yield s
    s.close()


def test_task_execution_with_exceptions(store):
    tasks = [
        ExampleTaskWithException("task_1", "http://example.com", should_fail=True),
        ExampleTaskWithException("task_2", "http://example.org", should_fail=False),
    ]
    store.put_many([TaskDefinition(id=t.id, data={"url": t.url}) for t in tasks])

    executor = LocalThreadedExecutor(tasks=tasks, store=store, max_concurrency=2)
    executor.run()

    assert store["task_1"].status == TaskStatus.FAILED
    assert store["task_2"].status == TaskStatus.COMPLETED
    assert store["task_2"].result["value"] == "Processed http://example.org"
    assert "Task task_1 failed" in store["task_1"].error


def test_status_summary_after_execution(store, capsys):
    tasks = [
        ExampleTaskWithException("task_1", "http://example.com", should_fail=False),
        ExampleTaskWithException("task_2", "http://example.org", should_fail=True),
        ExampleTaskWithException("task_3", "http://example.net", should_fail=False),
    ]
    store.put_many([TaskDefinition(id=t.id, data={"url": t.url}) for t in tasks])

    executor = LocalThreadedExecutor(tasks=tasks, store=store, max_concurrency=2)
    executor.run()

    captured = capsys.readouterr()
    assert "Pending tasks: 0" in captured.out
    assert "Completed tasks: 2" in captured.out
    assert "Failed tasks: 1" in captured.out


def test_failed_tasks_are_retried(store):
    execution_count = {}

    class ExampleTaskWithRetry(ExampleTaskWithException):
        def __call__(self):
            execution_count[self.task_id] = execution_count.get(self.task_id, 0) + 1
            if execution_count[self.task_id] == 1:
                raise Exception(f"Task {self.task_id} failed on first attempt")
            return f"Processed {self.url} on retry"

    tasks = [ExampleTaskWithRetry("task_1", "http://example.com", should_fail=True)]
    store.put_many([TaskDefinition(id="task_1", data={"url": "http://example.com"})])

    executor = LocalThreadedExecutor(tasks=tasks, store=store, max_concurrency=1)
    executor.run()

    assert execution_count["task_1"] == 2
    assert store["task_1"].status == TaskStatus.COMPLETED
    assert store["task_1"].result["value"] == "Processed http://example.com on retry"
