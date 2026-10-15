"""Importable task entrypoints shared by local scheduling and RQ workers."""


def run_simulation(kind, job_id, data):
    import web_server_server as runtime
    if kind == "simple":
        runtime.simple_simulation_task(job_id, data)
    elif kind == "energyplus" and runtime.energyplus_engine is not None:
        runtime.background_simulation_task(job_id, data)
    else:
        raise ValueError("Simulation engine not available")


def mark_failed(job, connection, exception_type, exception, traceback):
    _persist_failure(job)


def mark_killed(job, retpid, ret_val, rusage):
    """RQ calls this for OOM/SIGKILL/hard-timeout workhorse failures."""
    _persist_failure(job)


def _persist_failure(job):
    import web_server_server as runtime
    _, job_id, data = job.args
    runtime.simulation_jobs.update(job_id, status="failed", error="Simulation worker failed")
    runtime._update_report_status_if_exists(data["owner_username"], data["report_number"], "failed")
