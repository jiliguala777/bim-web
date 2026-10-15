"""Linux production RQ worker with ordinary and killed-workhorse status handling."""
import os

from runtime_config import application_config


def create_worker(connection=None):
    application_config()  # Reject missing/weak production keys before consuming jobs.
    from redis import Redis
    from rq import Queue, Worker
    from rq.serializers import JSONSerializer
    from worker_tasks import mark_killed
    if connection is None:
        connection = Redis.from_url(os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0"),
                                    socket_connect_timeout=3)
    queue = Queue("bim-simulation", connection=connection, serializer=JSONSerializer)
    return Worker([queue], connection=connection, serializer=JSONSerializer,
                  work_horse_killed_handler=mark_killed)


def main():
    create_worker().work()


if __name__ == "__main__":
    main()
