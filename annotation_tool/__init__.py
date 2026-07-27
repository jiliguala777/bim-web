"""Local PDF annotation package."""

from .config import AnnotationConfig
from .storage import AnnotationStore


def create_app(config=None):
    from .app import create_app as app_factory

    return app_factory(config)


__all__ = ["AnnotationConfig", "AnnotationStore", "create_app"]
