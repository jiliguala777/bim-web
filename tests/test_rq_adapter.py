import importlib.util
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("fakeredis"), "Optional fakeredis adapter integration test")
class RqAdapterTests(unittest.TestCase):
    def setUp(self):
        import fakeredis
        from rq import Queue
        from rq.serializers import JSONSerializer
        self.connection = fakeredis.FakeRedis()
        self.queue = Queue("bim-simulation", connection=self.connection, serializer=JSONSerializer)

    def test_enqueue_preserves_owner_payload_with_json_serializer(self):
        from task_dispatch import SimulationDispatcher, QueueFull
        dispatcher = SimulationDispatcher({"TASK_BACKEND": "rq", "TASK_PENDING": "1"})
        payload = {"owner_username": "alice", "report_number": "CASE-ONE"}
        with patch("redis.Redis.from_url", return_value=self.connection):
            job = dispatcher.submit("simple", "one", payload)
            with self.assertRaises(QueueFull):
                dispatcher.submit("simple", "two", payload)
        stored = self.queue.fetch_job("one")
        self.assertEqual(stored.func_name, "worker_tasks.run_simulation")
        self.assertEqual(stored.args, ["simple", "one", payload])
        self.assertEqual(stored.timeout, 300)
        self.assertEqual(stored.failure_callback.__name__, "mark_failed")

    def test_redis_failure_becomes_queue_unavailable(self):
        from redis.exceptions import ConnectionError
        from task_dispatch import SimulationDispatcher, QueueUnavailable
        dispatcher = SimulationDispatcher({"TASK_BACKEND": "rq"})
        with patch("redis.Redis.from_url", return_value=self.connection), \
                patch.object(self.connection, "lock", side_effect=ConnectionError("offline")):
            with self.assertRaises(QueueUnavailable):
                dispatcher.submit("simple", "one", {})
