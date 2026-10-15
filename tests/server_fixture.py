"""Import the server using explicit test settings, never production secrets."""
import os
from unittest.mock import patch

with patch.dict(os.environ, {
    "BIM_WEB_ENV": "development",
    "SECRET_KEY": "bim-web-tests-only-not-for-deployment",
}):
    import web_server_server as server

server.app.config.update(
    SESSION_COOKIE_SECURE=False,
    WTF_CSRF_ENABLED=False,
    AUTH_RATE_LIMIT_ENABLED=False,
)
