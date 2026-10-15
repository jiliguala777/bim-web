import tempfile
import importlib.util
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class WorkerFailureTests(unittest.TestCase):
    def test_killed_workhorse_marks_persistent_job_and_report_failed(self):
        from tests.server_fixture import server
        import worker_tasks
        with ExitStack() as stack:
            root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            stack.enter_context(patch.object(server, "DB_PATH", str(root / "users.db")))
            stack.enter_context(patch.object(server, "JOBS_DIR", str(root / "jobs")))
            server.init_db()
            server._upsert_report_status("alice", "ONE", "calculating")
            server.simulation_jobs.set("job-one", {"status": "processing"})
            job = SimpleNamespace(args=["simple", "job-one", {"owner_username": "alice", "report_number": "ONE"}])
            for callback in ("killed", "exception"):
                with self.subTest(callback=callback):
                    server._upsert_report_status("alice", "ONE", "calculating")
                    server.simulation_jobs.set("job-one", {"status": "processing"})
                    if callback == "killed":
                        worker_tasks.mark_killed(job, 123, 9, None)
                    else:
                        worker_tasks.mark_failed(job, None, RuntimeError, RuntimeError("failure"), None)
                    self.assertEqual(server.simulation_jobs.get("job-one")["status"], "failed")
                    self.assertEqual(server._report_row("alice", "ONE")["status"], "failed")

    @unittest.skipUnless(importlib.util.find_spec("fakeredis"), "Optional fakeredis worker wiring test")
    def test_production_worker_wires_the_killed_workhorse_handler(self):
        import fakeredis
        import simulation_worker
        import worker_tasks
        from rq import Queue
        from rq.serializers import JSONSerializer
        connection = fakeredis.FakeRedis()
        with patch.dict("os.environ", {"BIM_WEB_ENV": "development"}):
            worker = simulation_worker.create_worker(connection=connection)
        queue = Queue("bim-simulation", connection=connection, serializer=JSONSerializer)
        job = queue.enqueue("worker_tasks.run_simulation", "simple", "job-one", {})
        # Exercise RQ's real dispatch hook rather than only inspecting a mock.
        with patch.object(worker_tasks, "_persist_failure") as persist:
            worker.handle_work_horse_killed(job, 123, 9, None)
        persist.assert_called_once_with(job)
