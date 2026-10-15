"""Bounded local scheduling and an explicit Redis/RQ production adapter."""
import os
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore


class QueueFull(RuntimeError):
    pass


class QueueUnavailable(RuntimeError):
    pass


class BoundedExecutor:
    def __init__(self, workers=1, pending=4):
        if workers < 1 or pending < 0:
            raise ValueError("Invalid task concurrency")
        self.capacity = BoundedSemaphore(workers + pending)
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bim-task")

    def submit(self, function, *args):
        if not self.capacity.acquire(blocking=False):
            raise QueueFull("Task queue is full")

        def run():
            try:
                return function(*args)
            finally:
                self.capacity.release()

        try:
            future = self.executor.submit(run)
        except BaseException:
            self.capacity.release()
            raise
        future.add_done_callback(lambda task: self.capacity.release() if task.cancelled() else None)
        return future

    def shutdown(self):
        self.executor.shutdown(wait=True)


class SimulationDispatcher:
    def __init__(self, environ=None):
        env = os.environ if environ is None else environ
        self.backend = env.get("TASK_BACKEND", "local")
        if self.backend not in {"local", "rq"}:
            raise ValueError("TASK_BACKEND must be local or rq")
        self.workers = int(env.get("TASK_WORKERS", "1"))
        self.pending = int(env.get("TASK_PENDING", "4"))
        if not 1 <= self.workers <= 16 or not 0 <= self.pending <= 100:
            raise ValueError("TASK_WORKERS must be 1–16 and TASK_PENDING 0–100")
        self.redis_url = env.get("REDIS_URL", "redis://127.0.0.1:6379/0")
        self.local = BoundedExecutor(self.workers, self.pending) if self.backend == "local" else None

    def submit(self, kind, job_id, data):
        if kind not in {"simple", "energyplus"}:
            raise ValueError("Unsupported task kind")
        if self.local is not None:
            from worker_tasks import run_simulation
            return self.local.submit(run_simulation, kind, job_id, data)
        from redis import Redis
        from redis.exceptions import RedisError
        from rq import Queue
        from rq.job import Callback
        from rq.serializers import JSONSerializer
        connection = Redis.from_url(self.redis_url, socket_connect_timeout=2, socket_timeout=3)
        queue = Queue("bim-simulation", connection=connection, serializer=JSONSerializer)
        try:
            # Bound the shared backlog across web workers, not each process.
            with connection.lock("bim-simulation-enqueue", timeout=10, blocking_timeout=2):
                if queue.count >= max(1, self.pending):
                    raise QueueFull("Task queue is full")
                return queue.enqueue("worker_tasks.run_simulation", kind, job_id, data,
                                     job_id=job_id, job_timeout=300, result_ttl=3600,
                                     failure_ttl=86400, on_failure=Callback("worker_tasks.mark_failed", timeout=30))
        except RedisError as error:
            raise QueueUnavailable("Task service unavailable") from error
        finally:
            connection.close()
