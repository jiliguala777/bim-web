import os

os.environ.setdefault("SECRET_KEY", "bim-web-local-dev-secret")

import web_server_server

web_server_server.app.config["TEMPLATES_AUTO_RELOAD"] = True
web_server_server.app.jinja_env.auto_reload = True


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    web_server_server.app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
