import unittest


class RuntimeConfigTests(unittest.TestCase):
    def test_defaults_are_local_and_not_debugging(self):
        from runtime_config import local_run_options
        self.assertEqual(local_run_options({}), {
            "host": "127.0.0.1", "port": 5000, "debug": False, "threaded": True,
        })

    def test_debug_requires_explicit_development_mode(self):
        from runtime_config import local_run_options
        with self.assertRaises(ValueError):
            local_run_options({"BIM_WEB_DEBUG": "1"})
        self.assertTrue(local_run_options({
            "BIM_WEB_ENV": "development", "BIM_WEB_DEBUG": "1",
        })["debug"])

    def test_production_requires_a_long_fixed_secret(self):
        from runtime_config import application_config
        for env in ({}, {"SECRET_KEY": "short"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                application_config(env)
        cfg = application_config({"SECRET_KEY": "a" * 64})
        self.assertEqual(cfg["SECRET_KEY"], "a" * 64)
        self.assertTrue(cfg["SESSION_COOKIE_SECURE"])
        self.assertTrue(cfg["SESSION_COOKIE_HTTPONLY"])
        self.assertEqual(cfg["SESSION_COOKIE_SAMESITE"], "Lax")

    def test_development_generates_a_secret_without_secure_cookie(self):
        from runtime_config import application_config
        one = application_config({"BIM_WEB_ENV": "development"})
        two = application_config({"BIM_WEB_ENV": "development"})
        self.assertNotEqual(one["SECRET_KEY"], two["SECRET_KEY"])
        self.assertFalse(one["SESSION_COOKIE_SECURE"])

    def test_invalid_environment_or_limits_are_rejected(self):
        from runtime_config import application_config
        for env in (
            {"BIM_WEB_ENV": "typo"},
            {"BIM_WEB_ENV": "development", "MAX_UPLOAD_MB": "0"},
            {"BIM_WEB_ENV": "development", "MAX_UPLOAD_MB": "abc"},
        ):
            with self.subTest(env=env), self.assertRaises(ValueError):
                application_config(env)
