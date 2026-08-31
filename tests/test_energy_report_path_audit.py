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
        return names.get(node.id, "<dynamic>")
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
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
        template = _render_sql_expression(node.func.value, names)
        values = [_render_sql_expression(arg, names) for arg in node.args]
        if template is not None and all(value is not None for value in values):
            try:
                return template.format(*values)
            except (IndexError, KeyError, ValueError):
                return None
    return None


def _is_unsafe_report_sql(query):
    normalized = re.sub(r"\s+", " ", query).lower()
    if not re.search(r"\b(?:from|update|delete\s+from)\s+(?:reports\b|<dynamic>)", normalized):
        return False
    if "where" not in normalized:
        return False
    where_clause = normalized.split("where", 1)[1]
    if not re.search(r"\breport_number\b\s*(?:=|in\b|is\b)", where_clause):
        return False
    return not re.search(r"\busername\b\s*(?:=|in\b|is\b)", where_clause)


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
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in {"execute", "executemany"}:
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


def _target_names(target):
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    if isinstance(target, (ast.List, ast.Tuple)):
        return {name for item in target.elts for name in _target_names(item)}
    return set()


def _function_outer_expressions(node):
    yield from node.decorator_list
    yield from node.args.defaults
    yield from (default for default in node.args.kw_defaults if default is not None)
    arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
    if node.args.vararg is not None:
        arguments.append(node.args.vararg)
    if node.args.kwarg is not None:
        arguments.append(node.args.kwarg)
    yield from (argument.annotation for argument in arguments if argument.annotation is not None)
    if node.returns is not None:
        yield node.returns
    yield from getattr(node, "type_params", ())


class _FunctionLocalBindingCollector(ast.NodeVisitor):
    """Collect names that Python binds across an entire function scope."""

    def __init__(self, node):
        self.bound = {
            argument.arg
            for argument in node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        }
        if node.args.vararg is not None:
            self.bound.add(node.args.vararg.arg)
        if node.args.kwarg is not None:
            self.bound.add(node.args.kwarg.arg)
        self.global_names = set()
        self.nonlocal_names = set()
        for statement in node.body:
            self.visit(statement)

    @property
    def local_names(self):
        return self.bound - self.global_names - self.nonlocal_names

    def _bind_target(self, target):
        self.bound.update(_target_names(target))

    def _visit_nested_function_header(self, node):
        for expression in _function_outer_expressions(node):
            self.visit(expression)

    def visit_FunctionDef(self, node):
        self.bound.add(node.name)
        self._visit_nested_function_header(node)

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        self.bound.add(node.name)
        for expression in (*node.decorator_list, *node.bases):
            self.visit(expression)
        for keyword in node.keywords:
            self.visit(keyword.value)
        for type_param in getattr(node, "type_params", ()):
            self.visit(type_param)

    def visit_Lambda(self, node):
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)

    def visit_Global(self, node):
        self.global_names.update(node.names)

    def visit_Nonlocal(self, node):
        self.nonlocal_names.update(node.names)

    def visit_Assign(self, node):
        for target in node.targets:
            self._bind_target(target)
        self.visit(node.value)

    def visit_AnnAssign(self, node):
        self._bind_target(node.target)
        self.visit(node.annotation)
        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node):
        self._bind_target(node.target)
        self.visit(node.value)

    def visit_NamedExpr(self, node):
        self._bind_target(node.target)
        self.visit(node.value)

    def visit_For(self, node):
        self._bind_target(node.target)
        self.visit(node.iter)
        for statement in (*node.body, *node.orelse):
            self.visit(statement)

    def visit_AsyncFor(self, node):
        self.visit_For(node)

    def visit_With(self, node):
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._bind_target(item.optional_vars)
        for statement in node.body:
            self.visit(statement)

    def visit_AsyncWith(self, node):
        self.visit_With(node)

    def visit_ExceptHandler(self, node):
        if node.name is not None:
            self.bound.add(node.name)
        if node.type is not None:
            self.visit(node.type)
        for statement in node.body:
            self.visit(statement)

    def visit_Import(self, node):
        self.bound.update(imported.asname or imported.name.split(".", 1)[0] for imported in node.names)

    def visit_ImportFrom(self, node):
        self.bound.update(
            imported.asname or imported.name
            for imported in node.names
            if imported.name != "*"
        )

    def visit_Delete(self, node):
        for target in node.targets:
            self._bind_target(target)

    def visit_MatchAs(self, node):
        if node.name is not None:
            self.bound.add(node.name)
        if node.pattern is not None:
            self.visit(node.pattern)

    def visit_MatchStar(self, node):
        if node.name is not None:
            self.bound.add(node.name)

    def visit_MatchMapping(self, node):
        if node.rest is not None:
            self.bound.add(node.rest)
        self.generic_visit(node)


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
        bound_names = {
            imported.asname or imported.name
            for imported in node.names
            if imported.name != "*"
        }
        self._forget_aliases(bound_names)
        if node.module == "os.path":
            for imported in node.names:
                if imported.name == "join":
                    self._join_aliases.add(imported.asname or imported.name)

    def visit_Import(self, node):
        self._forget_aliases(
            {imported.asname or imported.name.split(".", 1)[0] for imported in node.names}
        )

    def visit_FunctionDef(self, node):
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node):
        self._visit_function(node)

    def _visit_function(self, node):
        for expression in _function_outer_expressions(node):
            self.visit(expression)

        saved_roots, saved_joins = self._upload_root_aliases, self._join_aliases
        local_names = _FunctionLocalBindingCollector(node).local_names
        self._upload_root_aliases = set(saved_roots) - local_names - {node.name}
        self._join_aliases = set(saved_joins) - local_names - {node.name}
        for statement in node.body:
            self.visit(statement)
        self._upload_root_aliases, self._join_aliases = saved_roots, saved_joins
        self._forget_aliases({node.name})

    def visit_Assign(self, node):
        self._remember_assignment_aliases(node.value, node.targets)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self._remember_assignment_aliases(node.value, [node.target])
        self.generic_visit(node)

    def _remember_assignment_aliases(self, value, targets):
        bound_names = {name for target in targets for name in _target_names(target)}
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        is_path_upload_root = (
            isinstance(value, ast.Call)
            and ((isinstance(value.func, ast.Name) and value.func.id == "Path") or (isinstance(value.func, ast.Attribute) and value.func.attr == "Path"))
            and value.args
            and _is_upload_folder_config(value.args[0])
        )
        is_upload_root = _is_upload_folder_config(value) or is_path_upload_root or (
            isinstance(value, ast.Name) and value.id in self._upload_root_aliases
        )
        is_join = (
            isinstance(value, ast.Name) and value.id in self._join_aliases
        ) or (isinstance(value, ast.Attribute) and value.attr == "join")
        self._forget_aliases(bound_names)
        if is_upload_root:
            self._upload_root_aliases.update(names)
        if is_join:
            self._join_aliases.update(names)

    def _forget_aliases(self, names):
        self._upload_root_aliases.difference_update(names)
        self._join_aliases.difference_update(names)

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
joiner = os.path.join
legacy_joiner = joiner(app.config["UPLOAD_FOLDER"], "energy", report_number)
path_root = Path(app.config["UPLOAD_FOLDER"])
legacy_path_root = path_root / "energy" / report_number
module_root = app.config["UPLOAD_FOLDER"]
def route():
    return os.path.join(module_root, "energy", report_number)
def outer():
    enclosing_root = app.config["UPLOAD_FOLDER"]
    def inner():
        return os.path.join(enclosing_root, "energy", report_number)
pathlib_root = pathlib.Path(app.config["UPLOAD_FOLDER"])
legacy_pathlib_root = pathlib_root / "energy" / report_number
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
                (17, "joiner(app.config['UPLOAD_FOLDER'], 'energy', report_number)"),
                (19, "path_root / 'energy' / report_number"),
                (22, "os.path.join(module_root, 'energy', report_number)"),
                (26, "os.path.join(enclosing_root, 'energy', report_number)"),
                (28, "pathlib_root / 'energy' / report_number"),
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
unsafe_in = "SELECT * FROM reports WHERE report_number IN (?)"
unsafe_is = "SELECT * FROM reports WHERE report_number IS ?"
unsafe_format = "SELECT * FROM {} WHERE report_number = ?".format("reports")
unsafe_dynamic_format = "SELECT * FROM reports WHERE report_number = {}".format(value)
unsafe_dynamic_table = "SELECT * FROM {} WHERE report_number = ?".format(table)
safe_in = "SELECT * FROM reports WHERE username IN (?) AND report_number = ?"
safe = "SELECT username FROM reports WHERE username = ? AND report_number = ?"
"""
        )
        self.assertEqual(
            [
                (2, "SELECT username FROM reports WHERE report_number = ?"),
                (3, "UPDATE reports SET username = ? WHERE report_number = ?"),
                (4, "SELECT * FROM reports WHERE report_number = ?"),
                (5, "SELECT * FROM reports WHERE report_number = <dynamic>"),
                (6, "SELECT * FROM reports WHERE report_number IN (?)"),
                (7, "SELECT * FROM reports WHERE report_number IS ?"),
                (8, "SELECT * FROM reports WHERE report_number = ?"),
                (9, "SELECT * FROM reports WHERE report_number = <dynamic>"),
                (10, "SELECT * FROM <dynamic> WHERE report_number = ?"),
            ],
            _unsafe_report_sql_literals(tree),
        )

    def test_scope_aliases_do_not_leak_between_functions(self):
        tree = ast.parse("""
def first():
    root = app.config["UPLOAD_FOLDER"]
    return root
def second():
    return os.path.join(root, "energy", report_number)
""")
        visitor = _DirectEnergyPathVisitor()
        visitor.visit(tree)
        self.assertEqual([], visitor.violations)

    def test_function_parameter_shadows_module_upload_root_alias(self):
        tree = ast.parse("""
root = app.config["UPLOAD_FOLDER"]
def route(root):
    return os.path.join(root, "energy", report_number)
""")
        visitor = _DirectEnergyPathVisitor()
        visitor.visit(tree)
        self.assertEqual([], visitor.violations)

    def test_function_local_non_upload_bindings_shadow_module_aliases(self):
        tree = ast.parse("""
root = app.config["UPLOAD_FOLDER"]
def assigned_route():
    root = "unrelated"
    return os.path.join(root, "energy", report_number)
def assigned_after_use_route():
    result = os.path.join(root, "energy", report_number)
    root = "unrelated"
    return result
def imported_route():
    from unrelated import root
    return os.path.join(root, "energy", report_number)
""")
        visitor = _DirectEnergyPathVisitor()
        visitor.visit(tree)
        self.assertEqual([], visitor.violations)

    def test_async_function_aliases_do_not_leak_between_siblings(self):
        tree = ast.parse("""
async def first():
    root = app.config["UPLOAD_FOLDER"]
    return root
async def second():
    return os.path.join(root, "energy", report_number)
""")
        visitor = _DirectEnergyPathVisitor()
        visitor.visit(tree)
        self.assertEqual([], visitor.violations)

    def test_unshadowed_enclosing_and_async_local_upload_aliases_are_detected(self):
        tree = ast.parse("""
module_root = app.config["UPLOAD_FOLDER"]
async def module_route():
    return os.path.join(module_root, "energy", report_number)
def outer():
    enclosing_root = app.config["UPLOAD_FOLDER"]
    async def inner():
        return os.path.join(enclosing_root, "energy", report_number)
async def local_route():
    local_root = app.config["UPLOAD_FOLDER"]
    return os.path.join(local_root, "energy", report_number)
""")
        visitor = _DirectEnergyPathVisitor()
        visitor.visit(tree)
        self.assertEqual(
            [
                (4, "os.path.join(module_root, 'energy', report_number)"),
                (8, "os.path.join(enclosing_root, 'energy', report_number)"),
                (11, "os.path.join(local_root, 'energy', report_number)"),
            ],
            visitor.violations,
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
            validate_layout(db_path, uploads, runtime_root=root)

            legacy = uploads / "energy" / "BIM-deadbeef"
            legacy.mkdir()
            with self.assertRaisesRegex(StorageLayoutError, "BIM-deadbeef"):
                validate_layout(db_path, uploads, runtime_root=root)

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
                validate_layout(db_path, uploads, runtime_root=root)

            (uploads / "energy" / "unknown-deadbeef").rmdir()
            conn = sqlite3.connect(db_path)
            conn.execute("INSERT INTO reports VALUES (?, ?)", ("", "BIM-EMPTY"))
            conn.commit()
            conn.close()
            with self.assertRaisesRegex(StorageLayoutError, "empty username"):
                validate_layout(db_path, uploads, runtime_root=root)

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
                validate_layout(db_path, uploads, runtime_root=root)
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
                    "--runtime-root",
                    str(root),
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
        self.assertLess(rollout.index('[ ! -e "$backup_dir" ]'), rollout.index('install -d -m 0700 "$backup_dir"'))
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
            "[ -f \"$snapshot_dir/release-before\" ]",
            "rollback_commit=$(sed -n",
        ):
            with self.subTest(required=required):
                self.assertIn(required, rollback)
        self.assertLess(rollback.index("systemctl stop bim-web"), rollback.index("mv /var/lib/bim-web/users.db"))
        self.assertLess(rollback.index("release-before"), rollback.index("systemctl stop bim-web"))
        self.assertNotIn("REPLACE_WITH_VERIFIED_COMMIT", rollback)

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
