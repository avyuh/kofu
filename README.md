# Kofu

[![PyPI version](https://badge.fury.io/py/kofu.svg)](https://badge.fury.io/py/kofu)
[![Python Versions](https://img.shields.io/pypi/pyversions/kofu)](https://pypi.org/project/kofu/)

**Kofu** (Japanese for "Miner") is a robust task execution framework with persistence, designed for I/O-heavy workloads like web scraping and LLM synthetic data generation on a single machine. It focuses on local concurrent execution, not distributed cluster computing.

## Features

- **Persistent Execution**: Survive restarts/crashes with SQLite-backed state
- **Concurrent Processing**: Thread-based parallelism with configurable workers
- **Atomic Operations**: Batch updates with transaction safety
- **Automatic Retries**: Configurable retry logic with exponential backoff
- **Progress Tracking**: Built-in tqdm integration for execution monitoring

## Installation

```bash
uv add kofu
```

## Quick Start

```python
from kofu import LocalThreadedExecutor, Task
from kofu.store import SingleSQLiteTaskStore
from kofu.tasks import SimpleFn

def fetch_url(url: str) -> dict:
    import requests
    response = requests.get(url)
    return {"status": response.status_code, "content": response.text[:100]}

tasks = [
    SimpleFn("example", fetch_url, args=("https://example.com",)),
    SimpleFn("python", fetch_url, args=("https://python.org",))
]

store = SingleSQLiteTaskStore(directory="./tasks_db")
executor = LocalThreadedExecutor(tasks=tasks, store=store, max_concurrency=2)
executor.run()

results = executor.get_results()
```

## Core Concepts

### Tasks
Implement the `Task` protocol or use `SimpleFn`:

```python
from dataclasses import dataclass
from kofu import Task

@dataclass
class AnalysisTask:
    input_data: str
    _task_id: str = None
    
    def __post_init__(self):
        if self._task_id is None:
            self._task_id = f"analysis_{hash(self.input_data)}"

    @property
    def id(self) -> str:
        return self._task_id

    def __call__(self) -> dict:
        return {"result": len(self.input_data)}
```

### Stores
Persistent storage backends:

```python
store = SingleSQLiteTaskStore(
    directory="./data",
    serializer=JSONSerializer(compression_level=1)
)
```

### Executors
Configure execution parameters:

```python
executor = LocalThreadedExecutor(
    tasks=[task1, task2],
    store=store,
    max_concurrency=4,
    retry=3,
    batch_size=50
)
```

## Advanced Usage

### Resume Execution

```python
# After crash/restart
store = SingleSQLiteTaskStore(directory="./tasks_db")
pending = store.get_pending_task_ids()

# Recreate tasks (or load from somewhere)
executor = LocalThreadedExecutor(
    tasks=[recreated_tasks[i] for i in pending],
    store=store
)
executor.run()
```

### LLM Processing

```python
from openai import OpenAI

client = OpenAI()

def llm_task(prompt: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return {"content": response.choices[0].message.content}

tasks = [SimpleFn(f"prompt_{i}", llm_task, args=(p,)) for i, p in enumerate(prompts)]
executor = LocalThreadedExecutor(tasks, store=SingleSQLiteTaskStore("./llm_tasks"), max_concurrency=5)
executor.run()
```

### Custom Serialization

```python
from kofu.store import Serializer

class BSONSerializer(Serializer):
    def serialize(self, obj) -> bytes:
        import bson
        return bson.dumps(obj)
    
    def deserialize(self, data: bytes) -> Any:
        import bson
        return bson.loads(data)

store = SingleSQLiteTaskStore(directory="./data", serializer=BSONSerializer())
```

## API Overview

| Component           | Description                                    |
|---------------------|------------------------------------------------|
| `Task`              | Protocol defining task interface               |
| `SimpleFn`          | Ready-to-use task wrapper for functions        |
| `SingleSQLiteTaskStore`| Production-ready SQLite persistence         |
| `LocalThreadedExecutor`| Thread-pool based task executor             |
| `TaskStatus`        | Enum (PENDING/COMPLETED/FAILED)                |

## Contributing

Contributions welcome! Please open an issue first to discuss proposed changes.

```bash
git clone https://github.com/avyuh/kofu
uv add -e .[dev]
pytest tests/
```

## License

MIT License. See `LICENSE` for details.