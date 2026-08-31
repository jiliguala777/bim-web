import ast
import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVER = ROOT / "web_server_server.py"


def _is_upload_folder_config(node):
    """Return whether *node* is app.config['UPLOAD_FOLDER'] in AST form."""

    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "config"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "app"
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "UPLOAD_FOLDER"
    )


def _path_parts(node):
    """Flatten direct path-building expressions into their AST components."""

    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add)):
        return _path_parts(node.left) + _path_parts(node.right)
    if isinstance(node, ast.Call) and (
        (isinstance(node.func, ast.Name) and node.func.id == "Path")
        or (isinstance(node.func, ast.Attribute) and node.func.attr == "Path")
    ):
        return [part for arg in node.args for part in _path_parts(arg)]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"join", "joinpath"}
    ):
        receiver = _path_parts(node.func.value) if node.func.attr == "joinpath" else []
        return receiver + [part for arg in node.args for part in _path_parts(arg)]
    return [node]


def _is_path_constructor(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add)):
        return True
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "Path":
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr == "Path":
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"join", "joinpath"}:
            return True
    return False


class _DirectEnergyPathVisitor(ast.NodeVisitor):
    """Find direct UPLOAD_FOLDER/energy construction outside the resolver."""

    def __init__(self):
        self._violations_by_line = {}

    @property
    def violations(self):
        return [self._violations_by_line[line] for line in sorted(self._violations_by_line)]

    def visit_Call(self, node):
        self._inspect(node)
        self.generic_visit(node)

    def visit_BinOp(self, node):
        self._inspect(node)
        self.generic_visit(node)

    def _inspect(self, node):
        if not _is_path_constructor(node):
            return
        parts = _path_parts(node)
        has_upload_root = any(_is_upload_folder_config(part) for part in parts)
        has_energy_segment = any(
            isinstance(part, ast.Constant) and part.value == "energy" for part in parts
        )
        if has_upload_root and has_energy_segment:
            violation = (node.lineno, ast.unparse(node))
            prior = self._violations_by_line.get(node.lineno)
            if prior is None or len(violation[1]) > len(prior[1]):
                self._violations_by_line[node.lineno] = violation


class EnergyReportPathAuditTests(unittest.TestCase):
    def source_tree(self):
        return ast.parse(SERVER.read_text(encoding="utf-8"))

    def test_server_has_no_direct_flat_energy_path_construction(self):
        visitor = _DirectEnergyPathVisitor()
        visitor.visit(self.source_tree())
        self.assertEqual(
            [],
            visitor.violations,
            "user reports must use resolve_energy_report_context; direct "
            "UPLOAD_FOLDER/energy paths found: "
            f"{visitor.violations}",
        )

    def test_semantic_guard_rejects_path_constructor_variants_but_allows_ops(self):
        tree = ast.parse(
            """
legacy_join = os.path.join(app.config["UPLOAD_FOLDER"], "energy", report_id)
legacy_path = Path(app.config["UPLOAD_FOLDER"]) / "energy" / report_number
legacy_pathlib = pathlib.Path(app.config["UPLOAD_FOLDER"]).joinpath("energy", report_number)
operations = os.path.join(app.config["UPLOAD_FOLDER"], "ops", "bestest")
"""
        )
        visitor = _DirectEnergyPathVisitor()
        visitor.visit(tree)
        self.assertEqual(
            [
                (2, "os.path.join(app.config['UPLOAD_FOLDER'], 'energy', report_id)"),
                (3, "Path(app.config['UPLOAD_FOLDER']) / 'energy' / report_number"),
                (4, "pathlib.Path(app.config['UPLOAD_FOLDER']).joinpath('energy', report_number)"),
            ],
            visitor.violations,
        )

    def test_server_report_number_predicates_always_scope_by_username(self):
        unsafe_queries = []
        for node in ast.walk(self.source_tree()):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            query = re.sub(r"\s+", " ", node.value).lower()
            if (
                "reports" in query
                and "where" in query
                and "report_number" in query
                and "username" not in query
            ):
                unsafe_queries.append((node.lineno, node.value.strip()))
        self.assertEqual(
            [],
            unsafe_queries,
            "report reads and updates must use the (username, report_number) identity: "
            f"{unsafe_queries}",
        )

    def test_legacy_spellings_remain_explicit_regression_guards(self):
        source = SERVER.read_text(encoding="utf-8")
        forbidden = (
            "os.path.join(app.config['UPLOAD_FOLDER'], 'energy', report_number)",
            "Path(app.config['UPLOAD_FOLDER']) / 'energy' / report_number",
            "WHERE report_number = ?",
        )
        for fragment in forbidden:
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, source)

    def test_deployment_guides_describe_safe_user_scoped_rollout(self):
        for relative_path in ("README.md", "deploy/README.md"):
            guide = (ROOT / relative_path).read_text(encoding="utf-8")
            for expected in (
                "/var/lib/bim-web/uploads/energy/<安全用户名目录>/<报告编号>/",
                "systemctl stop bim-web",
                "/var/lib/bim-web/users.db",
                "/var/lib/bim-web/uploads",
                "不自动移动或删除",
                "事务",
                "HTTP 403",
                "管理员",
                "回滚",
            ):
                with self.subTest(path=relative_path, expected=expected):
                    self.assertIn(expected, guide)


if __name__ == "__main__":
    unittest.main()
