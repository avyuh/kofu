# Kofu

[![PyPI version](https://img.shields.io/pypi/v/kofu.svg)](https://pypi.org/project/kofu)
[![Python Versions](https://img.shields.io/pypi/pyversions/kofu.svg)](https://pypi.org/project/kofu/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Kofu** (Japanese for "Miner") is a Python framework for managing and executing concurrent tasks with built-in persistence. It's designed for single-computer environments, particularly Colab notebooks, focusing on I/O-heavy operations such as web scraping and LLM-based workflows.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Colab Usage](#colab-usage)
- [LLM Prompting Example](#llm-prompting-example)
- [Advanced Features](#advanced-features)
- [Performance Considerations](#performance-considerations)
- [Contributing](#contributing)
- [Testing](#testing)
- [TODO](#todo)
- [License](#license)
- [Contact](#contact)

## Features

- **Concurrent Execution**: Run tasks in parallel using Python threads.
- **Persistence and Resumption**: 
  - Automatically resume pending or failed tasks after interruptions.
  - Store task statuses and results in SQLite (easily adaptable for Google Drive in Colab).
- **Idempotency**: Re-running the same set of tasks will only process incomplete ones.
- **Robust Error Handling**: 
  - Capture and store execution errors.
  - Configurable automatic retry for failed tasks.
- **Execution Control**:
  - Custom stop conditions for graceful termination.
  - Configurable concurrency levels.
- **Colab Compatibility**: Designed to work seamlessly in Colab notebooks, handling interruptions and restarts.

## Installation

Install from PyPI (recommended):

```bash
pip install kofu
```

For the latest development version, install from GitHub:

```bash
pip install git+https://github.com/avyuh/kofu.git
```

Kofu supports Python 3.7+.

## Quick Start

Here's a simple example to get you started with Kofu:

```python
from kofu import LocalThreadedExecutor, SQLiteMemory, SimpleFn

# Define tasks using SimpleFn
tasks = [
    SimpleFn(task_id="Python_(programming_language)", fn=download_wikipedia_page, args=("Python_(programming_language)",)),
    SimpleFn(task_id="Web_scraping", fn=download_wikipedia_page, args=("Web_scraping",)),
    SimpleFn(task_id="Concurrency_(computer_science)", fn=download_wikipedia_page, args=("Concurrency_(computer_science)",))
]

def download_wikipedia_page(page_name: str) -> str:
    import requests
    url = f"https://en.wikipedia.org/wiki/{page_name}"
    response = requests.get(url)
    return response.text[:100]  # Simulate task by returning first 100 characters of the page

# Set up memory for task persistence
memory = SQLiteMemory(path="tasks.db")

# Run the executor
executor = LocalThreadedExecutor(tasks=tasks, memory=memory, max_concurrency=2)
executor.run()

# Check task statuses
print(executor.status_summary())
```

### Explanation:
- **Tasks**: You can use `SimpleFn` to wrap simple functions as tasks. Here, we're downloading Wikipedia pages using the `download_wikipedia_page` function.
- **SQLiteMemory**: This stores the task status and results, allowing tasks to resume on failures or interruptions.
- **Executor**: The `LocalThreadedExecutor` runs tasks concurrently, with the option to set `max_concurrency` to control how many tasks run in parallel.

### Key Concepts:

1. **SimpleFn**: 
   - `SimpleFn` allows you to easily create tasks from functions without needing to define a custom class.
   - It requires a `task_id` (to uniquely identify the task) and a function (`fn`) along with its arguments (`args`).
   - In this example, `download_wikipedia_page` is the function that fetches a Wikipedia page, and `task_id` is the name of the page.

2. **SQLiteMemory**:
   - `SQLiteMemory` provides persistence by storing task statuses and results in an SQLite database (`tasks.db` in this case).
   - This ensures that you can resume or retry tasks after a failure or interruption without losing progress.

3. **LocalThreadedExecutor**:
   - This executor manages the execution of tasks concurrently using threads.
   - You can control the level of concurrency by adjusting the `max_concurrency` parameter.
   - It also ensures idempotency: running the same set of tasks will only process incomplete ones, and completed tasks are skipped.

4. **Task Status Summary**:
   - After running tasks, you can check their statuses (`completed`, `pending`, or `failed`) using `executor.status_summary()`.

This example shows how simple it is to use Kofu for running concurrent tasks with built-in persistence. For more complex tasks, you can define custom classes with a `get_id()` method and a `__call__()` method.

## Detailed Usage

### Defining Tasks

Tasks in Kofu must implement two methods:

1. `get_id()`: Returns a unique identifier for the task.
2. `__call__()`: Contains the main logic of the task.

```python
class MyTask(Task):
    def __init__(self, task_id, data):
        self.task_id = task_id
        self.data = data

    def get_id(self):
        return self.task_id

    def __call__(self):
        # Your task logic here
        result = process_data(self.data)
        return result
```

### Memory Backends

Kofu supports different memory backends for task persistence. The `SQLiteMemory` is provided out of the box:

```python
from kofu import SQLiteMemory

memory = SQLiteMemory(path="my_tasks.db")
```

### Executor Configuration

The `LocalThreadedExecutor` supports various configuration options:

```python
executor = LocalThreadedExecutor(
    tasks=tasks,
    memory=memory,
    max_concurrency=4,  # Maximum number of concurrent tasks
    retry=3,  # Number of retry attempts for failed tasks
    stop_all_when=custom_stop_condition  # Function returning True to stop execution
)
```

### Error Handling

Kofu automatically captures and stores task execution errors. Failed tasks are retried based on the `retry` parameter:

```python
# Retrieve failed tasks
failed_tasks = memory.get_failed_tasks()

# Inspect errors
for task in failed_tasks:
    print(f"Task {task.get_id()} failed with error: {task.error}")
```

## Colab Usage

Kofu is particularly useful in Colab environments. Here's an example of how to use kofu in a Colab notebook for web scraping tasks with persistent storage on Google Drive:

```python
!pip install git+https://github.com/avyuh/kofu.git

# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import requests
from bs4 import BeautifulSoup
from typing import Optional
from kofu import LocalThreadedExecutor, SQLiteMemory
import os

# Define a path inside Google Drive to store SQLite database
sqlite_path = '/content/drive/MyDrive/kofu_example/data.db'
os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)

def fetch_url(url: str) -> Optional[str]:
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"An error occurred while fetching the URL: {e}")
        return None

def extract_content(html_content: str) -> dict:
    soup = BeautifulSoup(html_content, 'html.parser')
    # Example extraction, modify as needed
    title = soup.find('h1').text if soup.find('h1') else "No title found"
    paragraphs = [p.text for p in soup.find_all('p')]
    return {"title": title, "paragraphs": paragraphs}

class WebScrapingTask:
    def __init__(self, url, task_id):
        self.url = url
        self.task_id = task_id

    def get_id(self):
        return self.task_id

    def __call__(self):
        content = fetch_url(self.url)
        if content:
            return extract_content(content)
        else:
            raise Exception(f"Failed to download {self.url}")

# Example list of URLs to scrape
urls_to_scrape = [
    "http://example.com/page1",
    "http://example.com/page2",
    "http://example.com/page3",
]

scraping_tasks = [WebScrapingTask(url, f"task_{i}") for i, url in enumerate(urls_to_scrape)]

memory = SQLiteMemory(path=sqlite_path)

# First run
print("First run:")
executor = LocalThreadedExecutor(tasks=scraping_tasks, memory=memory, max_concurrency=3)
executor.run()
print(executor.status_summary())

# Simulating an interruption (in real scenario, this would be the notebook restarting)
print("\nSimulating interruption and restarting...")

# Second run - will only process incomplete tasks
print("Second run (resuming):")
executor = LocalThreadedExecutor(tasks=scraping_tasks, memory=memory, max_concurrency=3)
executor.run()
print(executor.status_summary())
```

This example demonstrates persistence, idempotency, easy resumption, and concurrency control in a Colab environment.

## LLM Prompting Example

Kofu can also be used for managing and executing LLM prompting tasks. Here's an example:

```python
!pip install git+https://github.com/avyuh/kofu.git
!pip install openai

from kofu import LocalThreadedExecutor, SQLiteMemory
import openai
import os

# Set up OpenAI API (make sure to keep your API key secure)
openai.api_key = 'your-api-key-here'

class LLMPromptTask:
    def __init__(self, prompt, task_id):
        self.prompt = prompt
        self.task_id = task_id

    def get_id(self):
        return self.task_id

    def __call__(self):
        try:
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=self.prompt,
                max_tokens=150
            )
            return response.choices[0].text.strip()
        except Exception as e:
            raise Exception(f"Failed to get LLM response: {str(e)}")

# Example prompts
prompts = [
    "Explain the concept of machine learning in simple terms.",
    "What are the main differences between Python and JavaScript?",
    "Describe the process of photosynthesis briefly.",
]

llm_tasks = [LLMPromptTask(prompt, f"task_{i}") for i, prompt in enumerate(prompts)]

# Set up SQLite memory (adjust path as needed)
sqlite_path = 'llm_tasks.db'
memory = SQLiteMemory(path=sqlite_path)

# Run tasks
executor = LocalThreadedExecutor(tasks=llm_tasks, memory=memory, max_concurrency=2)
executor.run()
print(executor.status_summary())

# Retrieve and print results
for task in llm_tasks:
    result = memory.get_task_result(task.get_id())
    if result:
        print(f"Task {task.get_id()}:")
        print(result)
        print("---")
```

This example showcases how Kofu can manage LLM prompting tasks with concurrency and persistence.

## Advanced Features

### Custom Stop Conditions

Implement custom stop conditions to halt execution based on specific criteria:

```python
def rate_limit_reached():
    # Your logic to check if rate limit is reached
    return requests_made > MAX_REQUESTS

executor = LocalThreadedExecutor(tasks=tasks, memory=memory, stop_all_when=rate_limit_reached)
```

### Custom Memory Backends

Create custom memory backends for specific storage needs:

```python
class MyCustomMemory(Memory):
    def store_tasks(self, tasks):
        # Custom storage logic

    def update_task_statuses(self, statuses):
        # Custom status update logic

    def get_task_status(self, task_id):
        # Retrieve task status

    def get_completed_tasks(self):
        # Retrieve completed tasks

custom_memory = MyCustomMemory()
executor = LocalThreadedExecutor(tasks=tasks, memory=custom_memory)
```

## Performance Considerations

- Kofu is designed for single-computer use, with a focus on Colab notebooks.
- Current implementation allows only one thread to write to SQLite at a time, which may limit concurrency for write-heavy workloads.
- Typical concurrency of 5-10 tasks can be achieved on a Colab notebook, but this may vary based on the specific tasks and available resources.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please make sure to update tests as appropriate and adhere to the [Code of Conduct](https://github.com/avyuh/kofu/blob/main/CODE_OF_CONDUCT.md).

## Testing

Run the test suite using pytest:

```bash
pip install pytest
pytest
```

## TODO

- Implement thorough performance benchmarking, especially on Colab environments
- Optimize SQLite write operations for better concurrency
- Explore options for distributed computing in future versions
- Expand documentation with more real-world examples
- Implement additional memory backends (e.g., Redis, MongoDB)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/avyuh/kofu/blob/main/LICENSE) file for details.

## Contact

avyuh - [@avyuh](https://github.com/avyuh)

Project Link: [https://github.com/avyuh/kofu](https://github.com/avyuh/kofu)

---

Happy mining with Kofu! ⛏️
