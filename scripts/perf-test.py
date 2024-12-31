import time
import json
import random
import string
import statistics
import asyncio
import aiohttp
import threading
import resource
from typing import List, Dict, Optional
from dataclasses import dataclass
from kofu import LocalThreadedExecutor, SQLiteMemory
import os
import psutil

# Increase the soft limit of open files
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))

# Shared session for all tasks
import aiohttp.connector

aiohttp.connector.DEFAULT_FORCE_CLOSE = True


class EventLoopThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loop = None
        self.loop_ready = threading.Event()

    def run(self):
        # Create event loop in the thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop_ready.set()
        self.loop.run_forever()

    def stop(self):
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.join()
            self.loop.close()


class ThreadSafeEventLoopPool:
    def __init__(self, max_workers=10):
        self.max_workers = max_workers
        self.workers = []
        self.worker_index = 0
        self.lock = threading.Lock()

        # Start workers
        for _ in range(max_workers):
            worker = EventLoopThread()
            worker.start()
            worker.loop_ready.wait()  # Wait for loop to be ready
            self.workers.append(worker)

    def get_loop(self):
        with self.lock:
            worker = self.workers[self.worker_index]
            self.worker_index = (self.worker_index + 1) % self.max_workers
            return worker.loop

    def shutdown(self):
        for worker in self.workers:
            worker.stop()


# Global event loop pool
event_loop_pool = None


def setup_event_loop_pool(max_workers=10):
    global event_loop_pool
    event_loop_pool = ThreadSafeEventLoopPool(max_workers)


def cleanup_event_loop_pool():
    global event_loop_pool
    if event_loop_pool:
        event_loop_pool.shutdown()
        event_loop_pool = None


async def mock_llm_api(prompt: str, delay_range: tuple = (0.1, 0.5)) -> str:
    """Simulate an API call with realistic network delay and response generation"""
    await asyncio.sleep(random.uniform(*delay_range))
    response_length = len(prompt) // 5
    response_data = {
        "id": f"resp_{time.time()}",
        "model": "gpt-4",
        "response": {"text": "mocked response"},
        "usage": {
            "prompt_tokens": len(prompt) // 4,
            "completion_tokens": response_length // 4,
            "total_tokens": (len(prompt) + response_length) // 4,
        },
    }
    return json.dumps(response_data)


class LLMTask:
    def __init__(self, task_id: str, prompt_size: int, response_size: int):
        self.task_id = task_id
        # Generate a simpler prompt to reduce memory usage
        self.prompt = json.dumps(
            {
                "text": "".join(random.choices(string.ascii_letters, k=prompt_size)),
                "metadata": {"size": prompt_size},
            }
        )

    def get_id(self):
        return self.task_id

    def __call__(self):
        if not event_loop_pool:
            raise RuntimeError("Event loop pool not initialized")

        loop = event_loop_pool.get_loop()
        future = asyncio.run_coroutine_threadsafe(self._async_call(), loop)
        return future.result()

    async def _async_call(self):
        response = await mock_llm_api(self.prompt)
        response_data = json.loads(response)

        return {
            "task_id": self.task_id,
            "prompt_length": len(self.prompt),
            "response_length": len(response),
            "timestamp": time.time(),
        }


def run_performance_test(size_category: str, concurrency: int, base_dir="./test_data"):
    """Run a performance test with given parameters"""
    os.makedirs(base_dir, exist_ok=True)
    sqlite_path = os.path.join(base_dir, f"perf_test_{size_category}_{concurrency}.db")

    # Initialize memory
    memory = SQLiteMemory(sqlite_path)

    try:
        # Create tasks
        config = {
            "small": {"prompt_size": 1000, "response_size": 500, "num_tasks": 100},
            "medium": {"prompt_size": 10000, "response_size": 2000, "num_tasks": 100},
            "large": {"prompt_size": 30000, "response_size": 5000, "num_tasks": 100},
        }[size_category]

        tasks = [
            LLMTask(
                f"task_{size_category}_{i}",
                config["prompt_size"],
                config["response_size"],
            )
            for i in range(config["num_tasks"])
        ]

        # Initialize event loop pool
        setup_event_loop_pool(max_workers=min(10, concurrency))

        # Track process stats
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024

        # Run the executor
        executor = LocalThreadedExecutor(
            tasks=tasks, memory=memory, max_concurrency=concurrency
        )

        start_time = time.time()
        executor.run()
        end_time = time.time()

        # Calculate metrics
        completed = len(memory.get_completed_tasks())
        failed = len(memory.get_failed_tasks())
        throughput = completed / (end_time - start_time)

        print(f"\nResults for {size_category} size, concurrency {concurrency}:")
        print(f"Completed tasks: {completed}")
        print(f"Failed tasks: {failed}")
        print(f"Throughput: {throughput:.2f} tasks/s")
        print(f"Memory usage: {process.memory_info().rss/1024/1024:.1f} MB")

    finally:
        # Cleanup
        cleanup_event_loop_pool()
        memory.conn.close()
        try:
            if os.path.exists(sqlite_path):
                os.remove(sqlite_path)
        except:
            pass


def run_benchmark():
    """Run the complete benchmark suite"""
    sizes = ["small", "medium", "large"]
    concurrency_levels = [1, 5, 10, 25, 50]  # Removed 100 as it's too high

    for size in sizes:
        for concurrency in concurrency_levels:
            try:
                run_performance_test(size, concurrency)
            except Exception as e:
                print(f"Error running test {size} with concurrency {concurrency}: {e}")


if __name__ == "__main__":
    run_benchmark()
