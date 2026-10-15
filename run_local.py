import os

os.environ.setdefault("BIM_WEB_ENV", "development")

import web_server_server

web_server_server.app.config["TEMPLATES_AUTO_RELOAD"] = True
web_server_server.app.jinja_env.auto_reload = True


if __name__ == "__main__":
    web_server_server.app.run(**web_server_server.local_run_options())
