import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from annotation_tool.config import resolve_poppler_path


class AnnotationPopplerConfigTests(unittest.TestCase):
    def test_discovers_the_codex_runtime_poppler_without_an_environment_variable(self):
        with tempfile.TemporaryDirectory() as temporary:
            fake_home = Path(temporary)
            poppler_bin = (
                fake_home
                / ".cache"
                / "codex-runtimes"
                / "codex-primary-runtime"
                / "dependencies"
                / "native"
                / "poppler"
                / "Library"
                / "bin"
            )
            poppler_bin.mkdir(parents=True)
            (poppler_bin / "pdfinfo.exe").write_bytes(b"test")
            (poppler_bin / "pdftoppm.exe").write_bytes(b"test")

            with (
                patch.dict(os.environ, {"POPPLER_PATH": ""}),
                patch("annotation_tool.config.Path.home", return_value=fake_home),
                patch("annotation_tool.config.shutil.which", return_value=None),
            ):
                resolved = resolve_poppler_path()

        self.assertEqual(resolved, str(poppler_bin.resolve()))


if __name__ == "__main__":
    unittest.main()
