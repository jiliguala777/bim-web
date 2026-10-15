"""Explicit integration points for engines not distributed with this repository."""
import importlib
import os


def load_optional_engine(variable, required_exports):
    name = os.environ.get(variable, "").strip()
    if not name:
        return None
    try:
        engine = importlib.import_module(name)
    except ImportError as error:
        raise RuntimeError(f"Configured optional engine {variable} cannot be imported") from error
    missing = [export for export in required_exports if not hasattr(engine, export)]
    if missing:
        raise RuntimeError(f"Configured engine {variable} lacks exports: {', '.join(missing)}")
    return engine
