"""Atomic JSON task status with cross-process update locking."""
import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

_THREAD_LOCK = RLock()


class JobStore:
    def __init__(self, directory):
        self.directory = directory

    def _path(self, job_id, create=False):
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
            raise ValueError("Invalid job identifier")
        root = Path(self.directory() if callable(self.directory) else self.directory)
        if create:
            root.mkdir(parents=True, exist_ok=True)
        return root / (job_id + ".json")

    @contextmanager
    def _lock(self, path):
        with _THREAD_LOCK, path.with_suffix(".lock").open("a+b") as stream:
            if os.name == "nt":
                import msvcrt
                stream.seek(0, os.SEEK_END)
                if not stream.tell():
                    stream.write(b"\0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _read(self, path):
        try:
            with path.open(encoding="utf-8") as stream:
                return json.load(stream)
        except FileNotFoundError:
            return None

    def _write(self, path, data):
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def get(self, job_id):
        path = self._path(job_id)
        if not path.is_file():
            return None
        with self._lock(path):
            return self._read(path)

    def set(self, job_id, data):
        path = self._path(job_id, create=True)
        with self._lock(path):
            self._write(path, data)

    def update(self, job_id, **kwargs):
        path = self._path(job_id)
        if not path.is_file():
            return
        with self._lock(path):
            job = self._read(path)
            if job is not None:
                job.update(kwargs)
                self._write(path, job)
