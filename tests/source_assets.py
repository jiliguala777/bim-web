"""Read assembled assets for existing source-contract regression tests."""
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]


def energy_page_source():
    html = (ROOT / "templates/energy.html").read_text(encoding="utf-8")
    script = (ROOT / "static/energy/app.js").read_text(encoding="utf-8")
    return html.replace('<script src="/static/energy/app.js"></script>', '<script>\n' + script + '</script>')


def server_source():
    files = [ROOT / "web_server_server.py", *sorted((ROOT / "routes").glob("*.py"))]
    # Factory dependencies have an explicit namespace; original assertions still
    # inspect the underlying calculation expressions and response contracts.
    sources = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        if "def register(app, runtime):\n" in source:
            source = dedent(source.split("def register(app, runtime):\n", 1)[1])
        sources.append(source.replace("runtime.", ""))
    return "\n".join(sources)
