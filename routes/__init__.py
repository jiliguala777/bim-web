"""Blueprint registration; compatibility exports keep service integrations stable."""
from . import auth, energy, ops, pdf, recognition, reports


def register_routes(app, runtime):
    views = {}
    for module in (auth, energy, ops, pdf, recognition, reports):
        views.update(module.register(app, runtime))
    return views
