import threading
import unittest


class TaskDispatchTests(unittest.TestCase):
    def test_local_executor_rejects_overflow_and_releases_capacity(self):
        from task_dispatch import BoundedExecutor, QueueFull
        executor = BoundedExecutor(workers=1, pending=0)
        self.addCleanup(executor.shutdown)
        started = threading.Event()
        finish = threading.Event()
        self.addCleanup(finish.set)

        def task():
            started.set()
            finish.wait(5)
            return "finished"

        future = executor.submit(task)
        self.assertTrue(started.wait(2))
        with self.assertRaises(QueueFull):
            executor.submit(lambda: "overflow")
        finish.set()
        self.assertEqual(future.result(timeout=3), "finished")
        self.assertEqual(executor.submit(lambda: "next").result(timeout=3), "next")

    def test_invalid_dispatch_backend_is_rejected(self):
        from task_dispatch import SimulationDispatcher
        with self.assertRaises(ValueError):
            SimulationDispatcher({"TASK_BACKEND": "typo"})

    def test_task_kind_must_be_allowlisted(self):
        from task_dispatch import SimulationDispatcher
        dispatcher = SimulationDispatcher({})
        with self.assertRaises(ValueError):
            dispatcher.submit("arbitrary-function", "test", {})
