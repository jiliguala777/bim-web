import ast
import pathlib
import re
import sqlite3
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVER = ROOT / "web_server_server.py"


def _markdown_section(document, heading):
    start = document.index(heading)
    next_heading = document.find("\n## ", start + len(heading))
    return document[start:] if next_heading == -1 else document[start:next_heading]


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
    if isinstance(node, ast.JoinedStr):
        return [
            part
            for value in node.values
            for part in _path_parts(value.value if isinstance(value, ast.FormattedValue) else value)
        ]
    return [node]


def _is_path_constructor(node):
    if isinstance(node, (ast.BinOp, ast.JoinedStr)) and (
        not isinstance(node, ast.BinOp) or isinstance(node.op, (ast.Div, ast.Add))
    ):
        return True
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "Path":
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr == "Path":
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"join", "joinpath"}:
            return True
    return False


def _render_sql_expression(node, names):
    """Render simple SQL constants; mark interpolation instead of trusting it."""

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _render_sql_expression(node.left, names)
        right = _render_sql_expression(node.right, names)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        rendered = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                rendered.append(value.value)
            else:
                rendered.append("<dynamic>")
        return "".join(rendered)
    return None


def _is_unsafe_report_sql(query):
    normalized = re.sub(r"\s+", " ", query).lower()
    if not re.search(r"\b(?:from|update|delete\s+from)\s+reports\b", normalized):
        return False
    if "where" not in normalized:
        return False
    where_clause = normalized.split("where", 1)[1]
    if not re.search(r"\breport_number\b\s*=", where_clause):
        return False
    return not re.search(r"\busername\b\s*=", where_clause)


def _unsafe_report_sql_literals(tree):
    """Find report SQL whose WHERE clause lacks the owner predicate.

    This covers direct literals, aliases assigned from constant concatenation, and
    f-strings. More dynamic construction is intentionally not accepted by this
    audit once it visibly targets a report-number predicate.
    """

    names = {}
    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            rendered = _render_sql_expression(value, names)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if rendered is not None:
                for target in targets:
                    if isinstance(target, ast.Name):
                        names[target.id] = rendered
                candidates.append((node.lineno, rendered))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "execute":
            if node.args:
                rendered = _render_sql_expression(node.args[0], names)
                if rendered is not None:
                    candidates.append((node.lineno, rendered))

    unsafe = []
    seen = set()
    for line, query in candidates:
        item = (line, query.strip())
        if _is_unsafe_report_sql(query) and item not in seen:
            unsafe.append(item)
            seen.add(item)
    return unsafe


class _DirectEnergyPathVisitor(ast.NodeVisitor):
    """Find direct UPLOAD_FOLDER/energy construction outside the resolver."""

    def __init__(self):
        self._violations_by_line = {}
        self._upload_root_aliases = set()
        self._join_aliases = set()

    @property
    def violations(self):
        return [self._violations_by_line[line] for line in sorted(self._violations_by_line)]

    def visit_Call(self, node):
        self._inspect(node)
        self.generic_visit(node)

    def visit_BinOp(self, node):
        self._inspect(node)
        self.generic_visit(node)

    def visit_JoinedStr(self, node):
        self._inspect(node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module == "os.path":
            for imported in node.names:
                if imported.name == "join":
                    self._join_aliases.add(imported.asname or imported.name)
        self.generic_visit(node)

    def visit_Assign(self, node):
        self._remember_assignment_aliases(node.value, node.targets)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self._remember_assignment_aliases(node.value, [node.target])
        self.generic_visit(node)

    def _remember_assignment_aliases(self, value, targets):
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        if _is_upload_folder_config(value) or (
            isinstance(value, ast.Name) and value.id in self._upload_root_aliases
        ):
            self._upload_root_aliases.update(names)
        if isinstance(value, ast.Name) and value.id in self._join_aliases:
            self._join_aliases.update(names)

    def _inspect(self, node):
        is_join_alias = isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (
            node.func.id in self._join_aliases
        )
        if not _is_path_constructor(node) and not is_join_alias:
            return
        parts = (
            [part for arg in node.args for part in _path_parts(arg)]
            if is_join_alias
            else _path_parts(node)
        )
        has_upload_root = any(
            _is_upload_folder_config(part)
            or (isinstance(part, ast.Name) and part.id in self._upload_root_aliases)
            for part in parts
        )
        has_energy_segment = any(
            isinstance(part, ast.Constant)
            and isinstance(part.value, str)
            and part.value.strip("/\\") == "energy"
            for part in parts
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
root = app.config["UPLOAD_FOLDER"]
root_alias = root
legacy_root_alias = os.path.join(root_alias, "energy", report_number)
from os.path import join as imported_join
legacy_imported_join = imported_join(root, "energy", report_number)
assigned_join = imported_join
legacy_assigned_join = assigned_join(root, "energy", report_number)
legacy_f_string = f"{root}/energy/{report_number}"
legacy_concatenation = root + "/energy/" + report_number
annotated_root: str = app.config["UPLOAD_FOLDER"]
legacy_annotated_root = os.path.join(annotated_root, "energy", report_number)
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
                (7, "os.path.join(root_alias, 'energy', report_number)"),
                (9, "imported_join(root, 'energy', report_number)"),
                (11, "assigned_join(root, 'energy', report_number)"),
                (12, "f'{root}/energy/{report_number}'"),
                (13, "root + '/energy/' + report_number"),
                (15, "os.path.join(annotated_root, 'energy', report_number)"),
            ],
            visitor.violations,
        )

    def test_semantic_sql_guard_requires_username_in_the_where_predicate(self):
        tree = ast.parse(
            """
unsafe_select = "SELECT username FROM reports WHERE report_number = ?"
unsafe_update = "UPDATE reports SET username = ? WHERE report_number = ?"
unsafe_parts = "SELECT * FROM " + "reports WHERE report_number = ?"
unsafe_dynamic = f"SELECT * FROM reports WHERE report_number = {report_number}"
safe = "SELECT username FROM reports WHERE username = ? AND report_number = ?"
"""
        )
        self.assertEqual(
            [
                (2, "SELECT username FROM reports WHERE report_number = ?"),
                (3, "UPDATE reports SET username = ? WHERE report_number = ?"),
                (4, "SELECT * FROM reports WHERE report_number = ?"),
                (5, "SELECT * FROM reports WHERE report_number = <dynamic>"),
            ],
            _unsafe_report_sql_literals(tree),
        )

    def test_read_only_preflight_accepts_only_persisted_owner_keys(self):
        from tools.user_report_storage_preflight import StorageLayoutError, validate_layout

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            db_path = root / "users.db"
            uploads = root / "uploads"
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE reports (username TEXT, report_number TEXT)")
            conn.execute("INSERT INTO reports VALUES (?, ?)", ("alice", "BIM-1"))
            conn.commit()
            conn.close()

            valid_report = uploads / "energy" / "alice-2bd806c9" / "BIM-1"
            valid_report.mkdir(parents=True)
            validate_layout(db_path, uploads)

            legacy = uploads / "energy" / "BIM-deadbeef"
            legacy.mkdir()
            with self.assertRaisesRegex(StorageLayoutError, "BIM-deadbeef"):
                validate_layout(db_path, uploads)

    def test_read_only_preflight_rejects_unknown_and_empty_persisted_owners(self):
        from tools.user_report_storage_preflight import StorageLayoutError, validate_layout

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            db_path = root / "users.db"
            uploads = root / "uploads"
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE reports (username TEXT, report_number TEXT)")
            conn.execute("INSERT INTO reports VALUES (?, ?)", ("alice", "BIM-1"))
            conn.commit()
            conn.close()
            (uploads / "energy" / "unknown-deadbeef").mkdir(parents=True)
            with self.assertRaisesRegex(StorageLayoutError, "unknown-deadbeef"):
                validate_layout(db_path, uploads)

            (uploads / "energy" / "unknown-deadbeef").rmdir()
            conn = sqlite3.connect(db_path)
            conn.execute("INSERT INTO reports VALUES (?, ?)", ("", "BIM-EMPTY"))
            conn.commit()
            conn.close()
            with self.assertRaisesRegex(StorageLayoutError, "empty username"):
                validate_layout(db_path, uploads)

    def test_preflight_does_not_mutate_a_rejected_layout(self):
        from tools.user_report_storage_preflight import StorageLayoutError, validate_layout

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            db_path = root / "users.db"
            uploads = root / "uploads"
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE reports (username TEXT, report_number TEXT)")
            conn.execute("INSERT INTO reports VALUES (?, ?)", ("alice", "BIM-1"))
            conn.commit()
            conn.close()
            legacy = uploads / "energy" / "BIM-deadbeef"
            legacy.mkdir(parents=True)
            before = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
            with self.assertRaises(StorageLayoutError):
                validate_layout(db_path, uploads)
            after = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
            self.assertEqual(before, after)

    def test_preflight_cli_runs_read_only_outside_the_project_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            db_path = root / "users.db"
            uploads = root / "uploads"
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE reports (username TEXT, report_number TEXT)")
            conn.execute("INSERT INTO reports VALUES (?, ?)", ("alice", "BIM-1"))
            conn.commit()
            conn.close()
            (uploads / "energy" / "alice-2bd806c9" / "BIM-1").mkdir(parents=True)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/user_report_storage_preflight.py"),
                    "--db",
                    str(db_path),
                    "--uploads",
                    str(uploads),
                ],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)

    def test_server_report_number_predicates_always_scope_by_username(self):
        unsafe_queries = _unsafe_report_sql_literals(self.source_tree())
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

    def test_deployment_guide_orders_audit_and_preflight_before_service_start(self):
        guide = (ROOT / "deploy/README.md").read_text(encoding="utf-8")
        rollout = _markdown_section(guide, "### 12.1 用户隔离报告存储发布（必须停服并执行）")
        ordered_steps = (
            "systemctl stop bim-web",
            "cp -a /var/lib/bim-web/users.db",
            "git -C /opt/bim-web/app pull --ff-only origin main",
            "tests.test_energy_report_path_audit",
            "tools/user_report_storage_preflight.py",
            "systemctl start bim-web",
        )
        positions = [rollout.index(step) for step in ordered_steps]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("完成两点比例尺标定", rollout)
        self.assertNotIn("必要时两点比例尺标定", rollout)

    def test_deployment_guide_requires_validated_fail_fast_rollback(self):
        guide = (ROOT / "deploy/README.md").read_text(encoding="utf-8")
        rollback = _markdown_section(guide, "### 12.2 用户隔离报告存储回滚")
        for required in (
            "set -euo pipefail",
            "realpath -e",
            "sha256sum -c",
            "[ ! -e \"$stash_root\" ]",
            "[ -f \"$snapshot_dir/users.db\" ]",
            "[ -d \"$snapshot_dir/uploads\" ]",
        ):
            with self.subTest(required=required):
                self.assertIn(required, rollback)
        self.assertLess(rollback.index("systemctl stop bim-web"), rollback.index("mv /var/lib/bim-web/users.db"))

    def test_generic_update_and_rollback_sections_route_to_safe_procedure(self):
        deploy_guide = (ROOT / "deploy/README.md").read_text(encoding="utf-8")
        root_guide = (ROOT / "README.md").read_text(encoding="utf-8")
        deploy_update = _markdown_section(deploy_guide, "## 12. 后续更新")
        deploy_rollback = _markdown_section(deploy_guide, "## 13. 回退")
        root_update = _markdown_section(root_guide, "## 13. 服务器更新网站")
        root_rollback = _markdown_section(root_guide, "## 17. 代码回退")
        for section in (deploy_update, deploy_rollback, root_update, root_rollback):
            self.assertIn("12.1", section)
            self.assertIn("12.2", section)
        self.assertNotIn("systemctl restart bim-web", deploy_update)
        self.assertNotIn("systemctl restart bim-web", deploy_rollback)
        self.assertNotIn("systemctl restart bim-web", root_update)
        self.assertNotIn("systemctl restart bim-web", root_rollback)


if __name__ == "__main__":
    unittest.main()
