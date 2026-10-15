import os
from pathlib import Path
import secrets
import json
import copy
import hashlib
import hmac
import math
import shutil
import tempfile
import zipfile
import threading
import uuid
import time
from runtime_config import application_config, local_run_options
from auth_security import configure_csrf, consume_auth_attempt, valid_password
from job_store import JobStore
from optional_engines import load_optional_engine
from task_dispatch import SimulationDispatcher, QueueFull, QueueUnavailable

_APPLICATION_CONFIG = application_config()
from flask import Flask, after_this_request, request, render_template, jsonify, send_file, send_from_directory, session, redirect, url_for
from itsdangerous import BadSignature, URLSafeSerializer
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from functools import wraps
import logging
import numpy as np
import psutil
import subprocess
from contextlib import contextmanager

from energy_report_storage import (
    colliding_user_storage_keys,
    ensure_unique_user_storage_keys,
    EnergyReportContext,
    InvalidReportPath,
    ReportAccessDenied,
    resolve_report_child_destination,
    resolve_report_child_file,
    resolve_report_subdirectory,
    create_private_report_subdirectory,
    open_private_report_lock_file,
    resolve_energy_report_context,
    resolve_report_owner,
    StorageIdentityCollision,
    user_storage_key,
    validate_canonical_username,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
WEATHER_DATA_DIR = os.path.join(BASE_DIR, 'weather_data')
FLOORPLAN_MODEL_NAME = 'M2_UNet_ResNet34_DA'
ONNX_MODEL_PATH = os.environ.get('ONNX_MODEL_PATH', os.path.join(MODELS_DIR, 'M2_pub_plus_user.onnx'))
FLOORPLAN_MODEL_VERSION = os.path.basename(ONNX_MODEL_PATH)
FLOORPLAN_MODEL_MIOU = 0.787
FLOORPLAN_IMAGE_MODEL_NAME = 'DualHeadUNet_dataset_5000'
IMAGE_ONNX_MODEL_PATH = os.environ.get(
    'IMAGE_ONNX_MODEL_PATH', os.path.join(MODELS_DIR, 'floorplan_image_dataset_5000.onnx'),
)
FLOORPLAN_IMAGE_MODEL_VERSION = os.path.basename(IMAGE_ONNX_MODEL_PATH)
RECOGNITION_SCHEMA_VERSION = 1
LEGACY_ONNX_BACKEND = 'legacy_onnx'
VECTOR_PYTORCH_BACKEND = 'vector_pytorch'
IMAGE_ONNX_BACKEND = 'image_onnx'

# --- IFC / Benchmark 模块 ---
_ifc_engine = load_optional_engine('BIM_WEB_IFC_MODULE', ('IFCParser', 'HAS_IFC'))
IFCParser = _ifc_engine.IFCParser if _ifc_engine else None
HAS_IFC = bool(_ifc_engine and _ifc_engine.HAS_IFC)
_benchmark_engine = load_optional_engine('BIM_WEB_BENCHMARK_MODULE', (
    'run_benchmark', 'run_simple_model', 'generate_test_dxf', 'generate_test_ifc', 'BESTEST_BENCHMARKS',
))
HAS_BENCHMARK = _benchmark_engine is not None
run_benchmark = _benchmark_engine.run_benchmark if _benchmark_engine else None
run_simple_model = _benchmark_engine.run_simple_model if _benchmark_engine else None
generate_test_dxf = _benchmark_engine.generate_test_dxf if _benchmark_engine else None
generate_test_ifc = _benchmark_engine.generate_test_ifc if _benchmark_engine else None
BESTEST_BENCHMARKS = _benchmark_engine.BESTEST_BENCHMARKS if _benchmark_engine else {}
energyplus_engine = load_optional_engine('BIM_WEB_ENERGYPLUS_MODULE', ('generate_idf', 'run_eplus', 'parse_results'))

# --- AI 图纸识别模块 ---
try:
    from floorplan_onnx import get_segmenter
    from floorplan_page_pipeline import _write_image, prepare_pdf_page
    from floorplan_rooms import apply_scale_to_room_topology
    _floorplan_segmenter = get_segmenter(ONNX_MODEL_PATH)
    HAS_FLOORPLAN_AI = True
except Exception as e:
    HAS_FLOORPLAN_AI = False
    print(f"FloorPlan AI not available: {e}")

try:
    from floorplan_image_onnx import get_image_segmenter
    _floorplan_image_segmenter = get_image_segmenter(IMAGE_ONNX_MODEL_PATH)
    HAS_IMAGE_FLOORPLAN_AI = True
except Exception as e:
    _floorplan_image_segmenter = None
    HAS_IMAGE_FLOORPLAN_AI = False
    print(f"Image FloorPlan AI not available: {e}")

try:
    from vector_platform import VECTOR_BACKEND, VectorPlatformAdapter, VectorPlatformConfig
    _vector_platform_config = VectorPlatformConfig.from_environment()
    if _vector_platform_config is None:
        _vector_platform_adapter = None
        HAS_VECTOR_FLOORPLAN_AI = False
    else:
        _vector_platform_config.require_available()
        _vector_platform_adapter = VectorPlatformAdapter(_vector_platform_config)
        HAS_VECTOR_FLOORPLAN_AI = True
except Exception as e:
    _vector_platform_adapter = None
    HAS_VECTOR_FLOORPLAN_AI = False
    print(f"Vector FloorPlan AI not available: {e}")

try:
    from vector_pdf_fusion_pipeline import analyze_vector_pdf_page
    from vector_pdf_model import VectorModelConfig as VectorPdfModelConfig
    _vector_pdf_fusion_config = VectorPdfModelConfig.from_environment()
    if _vector_pdf_fusion_config is None:
        HAS_VECTOR_PDF_FUSION = False
    else:
        _vector_pdf_fusion_config.require_available()
        HAS_VECTOR_PDF_FUSION = True
except Exception as e:
    _vector_pdf_fusion_config = None
    HAS_VECTOR_PDF_FUSION = False
    print(f"Vector PDF fusion not available: {e}")

try:
    from vector_pdf_scale import (
        build_dimension_annotation_mask,
        build_nonstructural_vector_mask,
        build_structural_vector_mask,
        calibrate_from_overall_dimensions,
        detect_building_roi,
        detect_dimension_candidates,
        extract_vector_page,
        remove_dimension_annotations,
    )
    from floorplan_ocr import extract_numeric_text_spans
    HAS_VECTOR_PDF_SCALE = True
except Exception as e:
    HAS_VECTOR_PDF_SCALE = False
    print(f"Vector PDF scale calibration not available: {e}")

# --- 建筑能耗计算引擎 ---
try:
    import sys
    sys.path.insert(0, os.path.join(BASE_DIR, 'deploy'))
    sys.path.insert(0, BASE_DIR)
    import energy_calc
    HAS_ENERGY_CALC = True
except ImportError as e:
    HAS_ENERGY_CALC = False
    print(f"Energy Calc Import Error: {e}")

try:
    import design_load_calc
    HAS_DESIGN_LOAD_CALC = True
except ImportError as e:
    HAS_DESIGN_LOAD_CALC = False
    print(f"Design Load Calc Import Error: {e}")

try:
    from energy_library import list_envelope_parameters, list_materials, list_wall_assemblies
    HAS_BUILDING_LIBRARY = True
except ImportError as e:
    HAS_BUILDING_LIBRARY = False
    print(f"Building library import warning: {e}")

# --- 依赖检查 ---
try:
    import ezdxf
    from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString
    from shapely.ops import unary_union
    import pvlib
    HAS_ENERGY_DEPS = True

    # 尝试加载高级识别模块
    try:
        import sys
        _cad_engine = load_optional_engine('BIM_WEB_CAD_MODULE', ('AdvancedCADRecognition',))
        AdvancedCADRecognition = _cad_engine.AdvancedCADRecognition if _cad_engine else None
    except ImportError:
        AdvancedCADRecognition = None
except ImportError as e:
    HAS_ENERGY_DEPS = False
    print(f"Import Error: {e}")

# --- Configuration ---
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'csv', 'json', 'dxf', 'epw', 'png', 'jpg', 'jpeg'}
ALLOWED_DXF = {'dxf'}
ALLOWED_IFC = {'ifc'}
ALLOWED_EPW = {'epw'}
ALLOWED_RASTER = {'pdf', 'png', 'jpg', 'jpeg'}

# --- App Initialization ---
app = Flask(__name__, template_folder='templates', static_folder=os.path.join(BASE_DIR, 'static'), static_url_path='/static')
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', UPLOAD_FOLDER)
app.config.update(_APPLICATION_CONFIG)
if os.environ.get('BIM_WEB_TRUST_PROXY') == '1':
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
app.config['AUTH_RATE_LIMIT_ENABLED'] = True
configure_csrf(app)


@app.errorhandler(413)
def upload_too_large(error):
    return jsonify({'error': 'Upload exceeds the configured size limit'}), 413


def _resolve_poppler_path(candidate_dirs=None):
    """Return a directory containing native Poppler executables, if one is known."""
    configured = os.environ.get('POPPLER_PATH', '').strip()
    if configured:
        return configured

    if candidate_dirs is None:
        user_profile = os.environ.get('USERPROFILE', '')
        candidate_dirs = [
            os.path.join(BASE_DIR, 'tools', 'poppler', 'Library', 'bin'),
            os.path.join(BASE_DIR, 'tools', 'poppler', 'bin'),
        ]
        if user_profile:
            candidate_dirs.append(os.path.join(
                user_profile,
                '.cache',
                'codex-runtimes',
                'codex-primary-runtime',
                'dependencies',
                'native',
                'poppler',
                'Library',
                'bin',
            ))

    for candidate in candidate_dirs:
        candidate = os.fspath(candidate)
        if (
            os.path.isfile(os.path.join(candidate, 'pdfinfo.exe'))
            and os.path.isfile(os.path.join(candidate, 'pdftoppm.exe'))
        ):
            return candidate
    return None


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局任务池 (用于异步处理解决卡死问题) -> 改为基于文件持久化
JOBS_DIR = os.environ.get('JOBS_DIR', os.path.join(BASE_DIR, 'jobs'))
os.makedirs(JOBS_DIR, exist_ok=True)

simulation_jobs = JobStore(lambda: JOBS_DIR)
task_dispatcher = SimulationDispatcher()


class ReportOwnerNotFound(FileNotFoundError):
    """Raised when an administrator explicitly targets an unknown owner."""


class ReportNotFound(FileNotFoundError):
    """Raised when an authorized report directory does not exist."""


_REPORT_STORAGE_EXCEPTIONS = (
    ReportAccessDenied,
    InvalidReportPath,
    ReportOwnerNotFound,
    ReportNotFound,
)

# --- Authentication Decorator ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('logged_in') is not True or not session.get('username'):
            return redirect(url_for('auth.login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """Require a complete admin session before running an operations API."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('logged_in') is not True or not session.get('username'):
            return jsonify({'error': 'Authentication required'}), 401
        if session.get('is_admin') is not True:
            return jsonify({'error': 'Administrator access required'}), 403
        return f(*args, **kwargs)
    return decorated_function


def allowed_file(filename, allowed_set=None):
    if allowed_set is None:
        allowed_set = ALLOWED_EXTENSIONS
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set


import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

# --- Database Setup ---
DB_PATH = os.environ.get('DB_PATH', os.path.join(BASE_DIR, 'users.db'))


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _migrate_reports_schema(conn):
    """Upgrade report records to the user-scoped composite identity."""
    savepoint = "reports_schema_migration"
    conn.execute(f"SAVEPOINT {savepoint}")
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(reports)")}
        if "status" not in columns:
            conn.execute("ALTER TABLE reports ADD COLUMN status TEXT")
        if "updated_at" not in columns:
            conn.execute("ALTER TABLE reports ADD COLUMN updated_at DATETIME")

        conn.execute(
            """
            DELETE FROM reports
            WHERE id NOT IN (
                SELECT id FROM (
                    SELECT id,
                           ROW_NUMBER() OVER (
                               PARTITION BY username, report_number
                               ORDER BY created_at DESC, id DESC
                           ) AS row_number
                    FROM reports
                )
                WHERE row_number = 1
            )
            """
        )
        _remove_legacy_report_number_uniqueness(conn)
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS reports_username_report_number_uq
            ON reports(username, report_number)
            """
        )
    except Exception:
        conn.execute(f"ROLLBACK TO {savepoint}")
        conn.execute(f"RELEASE {savepoint}")
        raise
    else:
        conn.execute(f"RELEASE {savepoint}")


def _quote_sqlite_identifier(identifier):
    return '"' + identifier.replace('"', '""') + '"'


def _report_index_metadata(conn):
    metadata = []
    for index in conn.execute("PRAGMA index_list(reports)"):
        name = index[1]
        columns = [column[2] for column in conn.execute(f"PRAGMA index_info({_quote_sqlite_identifier(name)})")]
        metadata.append({
            "name": name,
            "unique": bool(index[2]),
            "origin": index[3],
            "columns": columns,
        })
    return metadata


def _remove_legacy_report_number_uniqueness(conn):
    legacy_indexes = [
        index for index in _report_index_metadata(conn)
        if index["unique"] and index["columns"] == ["report_number"]
    ]
    if not legacy_indexes:
        return

    if any(index["origin"] == "u" for index in legacy_indexes):
        _rebuild_reports_without_legacy_report_number_uniqueness(conn, legacy_indexes)
        return

    for index in legacy_indexes:
        conn.execute(f"DROP INDEX {_quote_sqlite_identifier(index['name'])}")


def _rebuild_reports_without_legacy_report_number_uniqueness(conn, legacy_indexes):
    _assert_reports_schema_can_be_rebuilt(conn)
    columns = list(conn.execute("PRAGMA table_info(reports)"))
    primary_key_columns = [column for column in columns if column[5]]
    if len(primary_key_columns) > 1:
        raise RuntimeError("cannot safely migrate reports with a composite primary key")

    retained_indexes = [
        index for index in _report_index_metadata(conn)
        if index not in legacy_indexes
    ]
    retained_index_sql = [
        conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?", (index["name"],)
        ).fetchone()[0]
        for index in retained_indexes
        if index["origin"] == "c"
    ]
    retained_trigger_sql = [
        trigger[0] for trigger in conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 'reports'"
        )
    ]

    definitions = []
    for _, name, column_type, not_null, default_value, primary_key in columns:
        definition = f"{_quote_sqlite_identifier(name)} {column_type}"
        if primary_key:
            definition += " PRIMARY KEY"
            if column_type.upper() == "INTEGER":
                definition += " AUTOINCREMENT"
        if not_null:
            definition += " NOT NULL"
        if default_value is not None:
            definition += f" DEFAULT {default_value}"
        definitions.append(definition)

    temporary_table = "reports_composite_migration"
    column_names = ", ".join(_quote_sqlite_identifier(column[1]) for column in columns)
    conn.execute(f"CREATE TABLE {_quote_sqlite_identifier(temporary_table)} ({', '.join(definitions)})")
    conn.execute(
        f"INSERT INTO {_quote_sqlite_identifier(temporary_table)} ({column_names}) "
        f"SELECT {column_names} FROM reports"
    )
    conn.execute("DROP TABLE reports")
    conn.execute(f"ALTER TABLE {_quote_sqlite_identifier(temporary_table)} RENAME TO reports")

    for index in retained_indexes:
        if index["origin"] == "u":
            index_name = f"reports_preserved_unique_{'_'.join(index['columns'])}"
            index_columns = ", ".join(_quote_sqlite_identifier(column) for column in index["columns"])
            conn.execute(
                f"CREATE UNIQUE INDEX {_quote_sqlite_identifier(index_name)} ON reports ({index_columns})"
            )
    for sql in retained_index_sql:
        conn.execute(sql)
    for sql in retained_trigger_sql:
        conn.execute(sql)


def _assert_reports_schema_can_be_rebuilt(conn):
    table_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'reports'"
    ).fetchone()[0]
    ddl_tokens = _sqlite_ddl_tokens(table_sql)
    unsupported_features = {
        "CHECK": "CHECK constraints",
        "REFERENCES": "foreign keys",
        "COLLATE": "collations",
        "GENERATED": "generated columns",
        "STRICT": "STRICT table options",
    }
    for token, description in unsupported_features.items():
        if token in ddl_tokens:
            raise RuntimeError(f"unsupported reports schema cannot be safely rebuilt: {description}")
    token_pairs = set(zip(ddl_tokens, ddl_tokens[1:]))
    unsupported_pairs = {
        ("FOREIGN", "KEY"): "foreign keys",
        ("WITHOUT", "ROWID"): "WITHOUT ROWID table options",
        ("ON", "CONFLICT"): "constraint conflict policies",
    }
    for tokens, description in unsupported_pairs.items():
        if tokens in token_pairs:
            raise RuntimeError(f"unsupported reports schema cannot be safely rebuilt: {description}")
    if any(column[6] for column in conn.execute("PRAGMA table_xinfo(reports)")):
        raise RuntimeError("unsupported reports schema cannot be safely rebuilt: generated columns")


def _sqlite_ddl_tokens(sql):
    """Return unquoted SQLite DDL keywords while skipping literal and comment text."""
    tokens = []
    position = 0
    while position < len(sql):
        character = sql[position]
        if character.isspace():
            position += 1
        elif sql.startswith("--", position):
            newline = sql.find("\n", position + 2)
            position = len(sql) if newline == -1 else newline + 1
        elif sql.startswith("/*", position):
            comment_end = sql.find("*/", position + 2)
            position = len(sql) if comment_end == -1 else comment_end + 2
        elif character in "'\"`":
            quote = character
            position += 1
            while position < len(sql):
                if sql[position] == quote:
                    position += 1
                    if position < len(sql) and sql[position] == quote:
                        position += 1
                        continue
                    break
                position += 1
        elif character == "[":
            identifier_end = sql.find("]", position + 1)
            position = len(sql) if identifier_end == -1 else identifier_end + 1
        elif character.isalpha() or character == "_":
            token_end = position + 1
            while token_end < len(sql) and (sql[token_end].isalnum() or sql[token_end] in "_$"):
                token_end += 1
            tokens.append(sql[position:token_end].upper())
            position = token_end
        else:
            position += 1
    return tokens


def _report_row(username, report_number):
    conn = get_db_connection()
    try:
        return conn.execute(
            "SELECT * FROM reports WHERE username = ? AND report_number = ?",
            (username, report_number),
        ).fetchone()
    finally:
        conn.close()


def _upsert_report_status(username, report_number, status):
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO reports (username, report_number, status, updated_at)
            VALUES (?, ?, ?, datetime('now'))
            ON CONFLICT(username, report_number) DO UPDATE SET
                status = excluded.status,
                updated_at = datetime('now')
            """,
            (username, report_number, status),
        )
        conn.commit()
    finally:
        conn.close()


def _update_report_status_if_exists(username, report_number, status):
    """Atomically update a report status without inserting a missing identity."""
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            """
            UPDATE reports
            SET status = ?, updated_at = datetime('now')
            WHERE username = ? AND report_number = ?
            """,
            (status, username, report_number),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def _user_exists(username):
    conn = get_db_connection()
    try:
        return conn.execute(
            "SELECT 1 FROM users WHERE username = ?",
            (username,),
        ).fetchone() is not None
    finally:
        conn.close()


def _configured_admin_username():
    """Return the enabled environment administrator identity, if configured."""

    if not os.environ.get('ADMIN_PASSWORD'):
        return None
    return os.environ.get('ADMIN_USER', 'admin')


def _database_usernames(conn=None):
    """Load every database authentication principal."""

    owns_connection = conn is None
    if owns_connection:
        conn = get_db_connection()
    try:
        return [row[0] for row in conn.execute("SELECT username FROM users")]
    finally:
        if owns_connection:
            conn.close()


def _authentication_principals(conn=None):
    principals = _database_usernames(conn)
    admin_username = _configured_admin_username()
    if admin_username is not None:
        principals.append(admin_username)
    return principals


def _require_unambiguous_report_owner(owner_username):
    """Fail closed when the selected owner key belongs to multiple principals."""

    if user_storage_key(owner_username) in colliding_user_storage_keys(
        _authentication_principals()
    ):
        raise StorageIdentityCollision("report owner is ambiguous")


def _requested_owner_username(payload=None):
    """Return an explicitly requested report owner from the current request."""
    if payload is not None and hasattr(payload, 'get') and 'owner_username' in payload:
        return payload.get('owner_username')
    if 'owner_username' in request.form:
        return request.form.get('owner_username')
    json_payload = request.get_json(silent=True)
    if json_payload is not None and hasattr(json_payload, 'get') and 'owner_username' in json_payload:
        return json_payload.get('owner_username')
    if 'owner_username' in request.args:
        return request.args.get('owner_username')
    return None


def _effective_report_owner(requested_owner=None):
    """Resolve and authorize the report owner represented by this request."""
    if requested_owner is None:
        requested_owner = _requested_owner_username()
    session_username = session.get('username')
    is_admin = session.get('is_admin') is True
    owner_username = resolve_report_owner(session_username, is_admin, requested_owner)
    _require_unambiguous_report_owner(owner_username)
    explicit_target = isinstance(requested_owner, str) and bool(requested_owner.strip())
    if is_admin and explicit_target and owner_username != session_username and not _user_exists(owner_username):
        raise ReportOwnerNotFound("the requested report owner does not exist")
    return owner_username


def _resolved_energy_report_context(report_number, requested_owner=None, create=False) -> EnergyReportContext:
    """Resolve and validate a request report without requiring it to exist."""
    if requested_owner is None:
        requested_owner = _requested_owner_username()
    owner_username = _effective_report_owner(requested_owner)
    return resolve_energy_report_context(
        app.config['UPLOAD_FOLDER'],
        owner_username,
        False,
        report_number,
        requested_owner=owner_username,
        create=create,
    )


def _energy_report_context(report_number, requested_owner=None, create=False) -> EnergyReportContext:
    """Resolve the current request's authorized, user-scoped report directory."""
    context = _resolved_energy_report_context(report_number, requested_owner, create)
    if not create and not context.report_dir.is_dir():
        raise ReportNotFound("report not found")
    return context


def _report_storage_error_response(exc):
    """Map report-storage failures without exposing filesystem locations."""
    if isinstance(exc, ReportAccessDenied):
        return jsonify({'error': 'Report access denied'}), 403
    if isinstance(exc, InvalidReportPath):
        return jsonify({'error': 'Invalid report path'}), 400
    if isinstance(exc, FileNotFoundError):
        return jsonify({'error': 'Report not found'}), 404
    raise exc


def _internal_report_error_response(log_message, exc):
    """Log report-processing detail while returning a path-safe response."""
    logger.error("%s: %s", log_message, exc, exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500


def init_db():
    conn = None
    try:
        conn = get_db_connection()
        conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        ''')
        conn.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            report_number TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            geometry_used TEXT,
            params TEXT,
            results TEXT
        );
        ''')
        _migrate_reports_schema(conn)
        conn.commit()
    except Exception:
        if conn is not None:
            conn.rollback()
        raise
    finally:
        if conn is not None:
            conn.close()

init_db()

# ==========================================
# 路由 - 身份认证
# ==========================================


# ==========================================
# 路由 - 功能入口
# ==========================================


# ==========================================
# 路由 - 文件上传
# ==========================================


# ==========================================
# 路由 - 模拟计算 (异步化)
# ==========================================

def get_dxf_metrics(dxf_path, wall_layers, window_layers, scale=1.0):
    """从 DXF 中提取面积、墙长和窗长"""
    try:
        if not os.path.exists(dxf_path): return 0, 0, 0
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()

        wall_lines = []
        win_lines = []
        all_pts = []

        for e in msp:
            layer = e.dxf.layer
            pts = []
            if e.dxftype() == 'LINE':
                pts = [(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]
            elif e.dxftype() == 'LWPOLYLINE':
                pts = [(p[0], p[1]) for p in e.get_points(format='xy')]

            if not pts: continue

            # 缩放
            pts = [(p[0]*scale, p[1]*scale) for p in pts]
            all_pts.extend(pts)

            line = LineString(pts)
            if layer in wall_layers:
                wall_lines.append(line)
            if layer in window_layers:
                win_lines.append(line)

        # 计算周长 (去重并合并线段)
        wall_len = MultiLineString(wall_lines).length if wall_lines else 0
        win_len = MultiLineString(win_lines).length if win_lines else 0

        # 计算面积 (使用凸包)
        if all_pts:
            from shapely.geometry import MultiPoint
            poly = MultiPoint(all_pts).convex_hull
            area = poly.area
        else:
            area = 0

        return round(area, 2), round(wall_len, 2), round(win_len, 2)
    except Exception as e:
        print(f"Metrics error: {e}")
        return 0, 0, 0


def _background_energy_report_context(data) -> EnergyReportContext:
    """Resolve a report from identity copied into a background-job payload."""
    owner_username = data.get('owner_username')
    _require_unambiguous_report_owner(owner_username)
    context = resolve_energy_report_context(
        app.config['UPLOAD_FOLDER'],
        owner_username,
        False,
        data.get('report_number', 'default'),
        requested_owner=owner_username,
        create=False,
    )
    if not context.report_dir.is_dir():
        raise ReportNotFound("report not found")
    return context


def simple_simulation_task(job_id, data):
    """简单能效计算模型 (基于度日数法 HDD/CDD + 真实几何)"""
    try:
        u_wall = float(data.get('u_wall', 0.6))
        u_win = float(data.get('u_win', 2.5))
        u_roof = float(data.get('u_roof', 0.4))
        u_floor = float(data.get('u_floor', 0.3))
        height = float(data.get('height', 3.0))
        floors = int(data.get('floors', 1))
        scale = float(data.get('scale', 1.0))
        t_heat = float(data.get('t_heat', 18.0))
        t_cool = float(data.get('t_cool', 26.0))

        wall_layers = data.get('wall_layers', [])
        win_layers = data.get('window_layers', [])

        context = _background_energy_report_context(data)
        dxf_path = resolve_report_child_file(
            context.report_dir, 'building_plan.dxf', required=False,
        )

        # 实时从 DXF 提取几何数据
        if os.path.exists(dxf_path) and (wall_layers or win_layers):
            area, wall_len, win_len = get_dxf_metrics(dxf_path, wall_layers, win_layers, scale)
        else:
            # AUTO PID 或无文件时的兜底值 (500m2, 80m wall, 20m win)
            area, wall_len, win_len = 500*(scale**2), 80*scale, 20*scale

        simulation_jobs.update(job_id, progress=50)


        # 估算总面积与围护结构
        total_floor_area = area * floors

        # 如果 window_layers 为空，则从墙长中分配 15% 给窗户
        if win_len == 0 and wall_len > 0:
            total_len = wall_len
            wall_len = total_len * 0.85
            win_len = total_len * 0.15

        wall_area = wall_len * height * floors
        win_area = win_len * height * floors

        # 度日数模型
        city = data.get('city_id', 'Beijing')
        weather_map = {
            'Beijing': {'hdd': 2800, 'cdd': 500},
            'Shanghai': {'hdd': 1500, 'cdd': 1100},
            'Shenzhen': {'hdd': 500, 'cdd': 1800},
            'NewYork': {'hdd': 2600, 'cdd': 600},
            'London': {'hdd': 3000, 'cdd': 100}
        }
        constants = weather_map.get(city, weather_map['Beijing'])

        # 综合导热系数 (ΣU*A)
        # 包含墙、窗、屋顶、地面
        total_ua = (u_wall * wall_area) + (u_win * win_area) + (u_roof * area) + (u_floor * area)

        # 度日数基准修正 (基于供暖120天，制冷90天估算)
        hdd_corrected = max(0, constants['hdd'] + (t_heat - 18.0) * 120)
        cdd_corrected = max(0, constants['cdd'] + (26.0 - t_cool) * 90)

        # 负荷计算 (Q = UA * DD * 24 / 1000)
        heating = total_ua * hdd_corrected * 24 / 1000
        cooling = total_ua * cdd_corrected * 24 / 1000

        # 内部负荷 (25 kWh/m2.a)
        internal = total_floor_area * 25

        total_cons = heating + cooling + internal
        eui = total_cons / total_floor_area if total_floor_area > 0 else 0

        simulation_jobs.update(job_id, progress=100, status='completed', result={
            'loads': {
                'heating': round(heating, 0),
                'cooling': round(cooling, 0),
                'eui_total': round(eui, 2)
            },
            'geometry': {
                'floor_area': round(total_floor_area, 2),
                'perimeter': round(wall_len + win_len, 2)
            }
        })
        _update_report_status_if_exists(
            data.get('owner_username'), data.get('report_number'), 'calculated',
        )
    except Exception as e:
        logger.error("Simple simulation task error: %s", e, exc_info=True)
        simulation_jobs.update(job_id, status='failed', error='Simulation failed')
        _update_report_status_if_exists(
            data.get('owner_username'), data.get('report_number'), 'failed',
        )


def background_simulation_task(job_id, data):
    """在后台执行复杂的能耗模拟逻辑"""
    try:
        context = _background_energy_report_context(data)
        target_dir = context.report_dir
        dxf_path = resolve_report_child_file(
            target_dir, 'building_plan.dxf', required=False,
        )
        epw_path = resolve_report_child_file(
            target_dir, 'weather_data.epw', required=False,
        )

        params = {
            'scale': float(data.get('scale', 1.0)),
            'height': float(data.get('height', 3.0)),
            'floors': int(data.get('floors', 1)),
            'wall_layers': data.get('wall_layers', []),
            'window_layers': data.get('window_layers', []),
            'u_wall': float(data.get('u_wall', 0.6)),
            'u_win': float(data.get('u_win', 2.5)),
            'u_roof': float(data.get('u_roof', 0.4)),
            'u_floor': float(data.get('u_floor', 0.3)),
            't_heat': float(data.get('t_heat', 18.0)),
            't_cool': float(data.get('t_cool', 26.0)),
        }

        # 1. 执行几何提取 (这部分通常比较快)
        simulation_jobs.update(job_id, progress=20)
        geom_result = extract_geometry(dxf_path, params)

        # 2. 执行 EnergyPlus 模拟 (深度算法)
        simulation_jobs.update(job_id, progress=50)
        runs_root = resolve_report_subdirectory(
            target_dir, 'energyplus_runs', create=True,
        )
        output_dir = create_private_report_subdirectory(runs_root, job_id)
        idf_path = resolve_report_child_file(output_dir, 'run.idf', required=False)

        energyplus_engine.generate_idf(idf_path, geom_result, params)
        csv_path = energyplus_engine.run_eplus(idf_path, epw_path, output_dir)
        eplus_data = energyplus_engine.parse_results(csv_path)

        # 3. 构造最终返回结果
        simulation_jobs.update(job_id, progress=100, status='completed', result={
            'loads': {
                'heating': eplus_data['heating_kwh'],
                'cooling': eplus_data['cooling_kwh'],
                'eui_total': round(eplus_data['total_kwh'] / geom_result['floor_area'], 2) if geom_result['floor_area'] > 0 else 0
            },
            'geometry': {
                'floor_area': round(geom_result['floor_area'], 2),
                'perimeter': round(geom_result['perimeter'], 2),
                'calculated_wwr': 0.35 # 简化展示
            },
            'weather': {
                'location': 'Simulated',
                'heating_dd': 2500,
                'cooling_dd': 1200
            }
        })
        _update_report_status_if_exists(
            data.get('owner_username'), data.get('report_number'), 'calculated',
        )

    except Exception as e:
        logger.error("Simulation task error: %s", e, exc_info=True)
        simulation_jobs.update(job_id, status='failed', error='Simulation failed')
        _update_report_status_if_exists(
            data.get('owner_username'), data.get('report_number'), 'failed',
        )

def extract_geometry(dxf_path, params):
    """通用的 CAD 几何提取模块"""
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    scale = params['scale']
    wall_layers = [l.strip() for l in params['wall_layers']]

    lines = []
    for e in msp:
        layer = e.dxf.layer.strip()
        if layer in wall_layers:
            if e.dxftype() == 'LINE':
                lines.append(LineString([(e.dxf.start.x*scale, e.dxf.start.y*scale), (e.dxf.end.x*scale, e.dxf.end.y*scale)]))
            elif e.dxftype() == 'LWPOLYLINE':
                pts = [(p[0]*scale, p[1]*scale) for p in e.get_points(format='xy')]
                if len(pts) > 1: lines.append(LineString(pts))

    if not lines: raise ValueError("Selected layers contain no valid lines")

    merged = MultiLineString(lines).buffer(0.1)
    if isinstance(merged, MultiPolygon): poly = max(merged.geoms, key=lambda p: p.area)
    else: poly = merged

    return {
        'floor_area': poly.area,
        'perimeter': poly.length,
        'wall_coords': list(poly.exterior.coords)
    }

# ==========================================
# 路由 - IFC/BIM 文件处理
# ==========================================


# ==========================================
# 路由 - 模型精度基准测试
# ==========================================


# ==========================================
# 路由 - 运维监控 API (供 OpenClaw Agent 调用)
# ==========================================


import base64
import cv2
import numpy as np
import time as _time
from energy_pdf_region import (
    crop_page_inputs,
    map_crop_bbox_to_page,
    parse_crop_region_request,
    render_pdf_page_preview,
)

# ==========================================
# 路由 - AI 图纸识别 (语义分割)
# ==========================================

def _recognition_json_path(target_dir):
    return os.path.join(target_dir, 'recognition.json')


def _canonical_report_directory(report_dir):
    return resolve_report_subdirectory(report_dir)


def _validated_report_child_file(report_dir, child_path, *, required=True):
    """Return one regular, non-aliased file directly below *report_dir*."""
    candidate = Path(child_path)
    canonical_report = _canonical_report_directory(report_dir)
    if candidate.parent.absolute() != Path(report_dir).absolute():
        raise InvalidReportPath('report child file is invalid')
    return resolve_report_child_file(canonical_report, candidate.name, required=required)


def _validated_report_child_directory(report_dir, child_path, *, create=False):
    """Return one non-aliased directory directly below *report_dir*."""
    candidate = Path(child_path)
    canonical_report = _canonical_report_directory(report_dir)
    if candidate.parent.absolute() != Path(report_dir).absolute():
        raise InvalidReportPath('report child directory is invalid')
    return resolve_report_subdirectory(canonical_report, candidate.name, create=create)


def _atomic_file_operation(destination, operation):
    """Publish a fixed report child without opening the destination alias."""

    destination = Path(destination)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='wb', dir=destination.parent,
            prefix=f'.{destination.name}.', suffix='.tmp', delete=False,
        ) as handle:
            temporary = Path(handle.name)
        operation(temporary)
        with temporary.open('r+b') as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_save_upload(uploaded, destination):
    _atomic_file_operation(destination, lambda temporary: uploaded.save(temporary))


def _atomic_copy_file(source, destination):
    _atomic_file_operation(destination, lambda temporary: shutil.copyfile(source, temporary))


def _atomic_save_image(image, destination, image_format):
    _atomic_file_operation(
        destination,
        lambda temporary: image.save(temporary, image_format),
    )


def _geometry_summary(geometry):
    geometry = geometry or {}
    return {
        'walls': len(geometry.get('walls', [])),
        'windows': len(geometry.get('windows', [])),
        'doors': len(geometry.get('doors', [])),
    }


def _polyline_total_length(items):
    total = 0.0
    for item in items or []:
        pts = item.get('pts') or []
        for i in range(len(pts) - 1):
            dx = pts[i + 1][0] - pts[i][0]
            dy = pts[i + 1][1] - pts[i][1]
            total += (dx * dx + dy * dy) ** 0.5
    return total


def _build_recognition_payload(
    result,
    preprocessing,
    use_preprocessing,
    raster_path,
    overlay_path,
    mask_path,
    elapsed_sec,
    owner_username,
):
    geometry = result.get('geometry') or {'walls': [], 'windows': [], 'doors': []}
    room_topology = result.get('room_topology') or {
        'status': 'no_closed_rooms',
        'room_count': 0,
        'closure_applied': False,
        'rooms': [],
        'total_area_px2': 0.0,
        'total_area_m2': None,
        'scale_m_per_px': None,
        'load_geometry_ready': False,
    }
    image_size = result.get('image_size')
    if not image_size:
        mask = result.get('mask')
        if mask is not None:
            h, w = mask.shape[:2]
            image_size = [w, h]
    if not image_size:
        image_size = [0, 0]

    model = result.get('model') or {
        'backend': LEGACY_ONNX_BACKEND,
        'name': FLOORPLAN_MODEL_NAME,
        'version': FLOORPLAN_MODEL_VERSION,
        'path': ONNX_MODEL_PATH,
        'mIoU': FLOORPLAN_MODEL_MIOU,
        'classes': ['background', 'wall', 'window', 'door'],
    }
    return {
        'schema_version': RECOGNITION_SCHEMA_VERSION,
        'owner_username': owner_username,
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'model': model,
        'preprocessing': {
            'requested': preprocessing,
            'use_preprocessing': use_preprocessing,
        },
        'source_image': {
            'path': os.path.basename(raster_path),
        },
        'image_size': image_size,
        'stats': result.get('stats') or {},
        'geometry_summary': _geometry_summary(geometry),
        'geometry': geometry,
        'contours': geometry,
        'room_topology': room_topology,
        'scale_calibration': result.get('scale_calibration') or {
            'status': 'manual_required',
            'method': 'raster_manual',
            'scale_m_per_px': None,
            'confidence': 0.0,
            'horizontal': None,
            'vertical': None,
            'axis_difference_percent': None,
            'evidence': [],
        },
        'pdf_page_number': result.get('pdf_page_number'),
        'pdf_page_count': result.get('pdf_page_count'),
        'recognition_mode': result.get('recognition_mode', 'full_page'),
        'crop_bbox_page_px': result.get('crop_bbox_page_px'),
        'crop_bbox_preview_px': result.get('crop_bbox_preview_px'),
        'crop_preview_size': result.get('crop_preview_size'),
        'vector_cleanup': result.get('vector_cleanup') or {
            'enabled': False,
            'has_vector_geometry': False,
            'has_vector_text': False,
        },
        'topology_repair': result.get('topology_repair') or {},
        'exterior_elements': result.get('exterior_elements'),
        'vector_geometry': result.get('vector_geometry'),
        'vector_measurements': result.get('vector_measurements'),
        'vector_inference': result.get('vector_inference'),
        'pixel_lengths': {
            'wall_px': round(_polyline_total_length(geometry.get('walls', [])), 1),
            'window_px': round(_polyline_total_length(geometry.get('windows', [])), 1),
        },
        'mask': {
            'path': os.path.basename(mask_path),
            'shape': image_size,
            'encoding': 'class_color_png',
            'classes': {'background': 0, 'wall': 1, 'window': 2, 'door': 3},
        },
        'artifacts': {
            'original': os.path.basename(raster_path),
            'overlay': os.path.basename(overlay_path),
            'mask': os.path.basename(mask_path),
            'pdf_nonstructural_mask': 'pdf_nonstructural_mask.png'
            if (result.get('vector_cleanup') or {}).get('nonstructural_mask', {}).get('enabled')
            else None,
            'pdf_model_input': 'pdf_model_input.png'
            if (result.get('vector_cleanup') or {}).get('has_vector_geometry')
            else None,
            'pdf_building_roi': 'pdf_building_roi.json'
            if (result.get('vector_cleanup') or {}).get('has_vector_geometry')
            else None,
            'raw_model_mask': 'ai_raw_model_mask.png'
            if result.get('raw_model_mask') is not None
            else None,
        },
    }


def _save_recognition_payload(target_dir, payload):
    path = _validated_report_child_file(
        target_dir,
        _recognition_json_path(target_dir),
        required=False,
    )
    _atomic_write_json(path, payload)
    return path


def _enforce_topology_repair_readiness(topology, topology_repair):
    """A calibrated scale cannot make an unclosed exterior safe for loads."""
    repair = topology_repair or {}
    if (
        repair.get('manual_exterior_wall_required')
        or repair.get('closure_status') in {'partial', 'failed'}
    ):
        topology['load_geometry_ready'] = False
    return topology


def _load_recognition_payload(target_dir):
    if not Path(target_dir).exists():
        return None
    try:
        path = _validated_report_child_file(
            target_dir,
            _recognition_json_path(target_dir),
        )
    except FileNotFoundError:
        return None
    with path.open('r', encoding='utf-8') as f:
        payload = json.load(f)
    if payload.get('schema_version') != RECOGNITION_SCHEMA_VERSION:
        return None
    model = payload.get('model') or {}
    if model.get('backend', LEGACY_ONNX_BACKEND) not in {
        LEGACY_ONNX_BACKEND, VECTOR_PYTORCH_BACKEND, IMAGE_ONNX_BACKEND,
    }:
        return None
    if not isinstance(model.get('version'), str) or not model.get('version'):
        return None
    geometry = payload.get('geometry') or {}
    if not all(key in geometry for key in ('walls', 'windows', 'doors')):
        return None
    return payload


def _public_ai_model_info(model):
    """Return model metadata that is safe to expose through JSON APIs."""
    if not isinstance(model, dict):
        return {}
    return {
        key: model[key]
        for key in ('backend', 'name', 'version', 'mIoU', 'classes')
        if key in model
    }


def _persist_vector_recognition(
    target_dir, raster_path, result, elapsed_sec, owner_username,
):
    """Persist the opt-in vector backend in the existing recognition contract."""
    original_bgr = result['source']
    overlay = result['overlay']
    mask = result['mask']
    h, w = mask.shape
    overlay_path = resolve_report_child_destination(target_dir, 'ai_overlay.jpg')
    mask_path = resolve_report_child_destination(target_dir, 'ai_mask.png')
    _write_image(Path(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
    mask_color = np.zeros((h, w, 3), dtype=np.uint8)
    mask_color[mask == 1] = (60, 76, 231)
    mask_color[mask == 2] = (219, 152, 52)
    mask_color[mask == 3] = (113, 204, 46)
    _write_image(Path(mask_path), mask_color)
    result.update({
        'recognition_mode': result.get('recognition_mode', 'full_page'),
        'vector_cleanup': {'enabled': False, 'has_vector_geometry': False, 'has_vector_text': False},
        'topology_repair': {},
        'pdf_page_number': None,
        'pdf_page_count': None,
        'crop_bbox_page_px': result.get('crop_bbox_page_px'),
        'crop_bbox_preview_px': result.get('crop_bbox_preview_px'),
        'crop_preview_size': result.get('crop_preview_size'),
    })
    recognition_payload = _build_recognition_payload(
        result, 'none', False, raster_path, overlay_path, mask_path, elapsed_sec,
        owner_username,
    )
    recognition_path = _save_recognition_payload(target_dir, recognition_payload)
    _, overlay_buf = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
    _, mask_buf = cv2.imencode('.png', mask_color)
    _, orig_buf = cv2.imencode('.jpg', original_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    geometry = result['geometry']
    return jsonify({
        'success': True,
        'status': 'success',
        'source': 'AI',
        'model': result['model']['name'],
        'model_info': _public_ai_model_info(result['model']),
        'elapsed_sec': elapsed_sec,
        'image_size': [w, h],
        'stats': result['stats'],
        'geometry_summary': {
            'walls': len(geometry['walls']), 'windows': len(geometry['windows']), 'doors': len(geometry['doors']),
        },
        'geometry': geometry,
        'room_topology': result['room_topology'],
        'scale_calibration': result['scale_calibration'],
        'recognition_mode': result['recognition_mode'],
        'vector_geometry': result['vector_geometry'],
        'vector_measurements': result['vector_measurements'],
        'recognition': {
            'schema_version': recognition_payload['schema_version'],
            'path': os.path.basename(recognition_path),
            'model_version': result['model']['version'],
            'preprocessing': recognition_payload['preprocessing'],
        },
        'images': {
            'original': base64.b64encode(orig_buf).decode('utf-8'),
            'overlay': base64.b64encode(overlay_buf).decode('utf-8'),
            'mask': base64.b64encode(mask_buf).decode('utf-8'),
        },
    })


def _pdf_upload_serializer():
    return URLSafeSerializer(app.secret_key, salt='energy-pdf-upload-v1')


def _make_pdf_upload_token(owner_username, report_number, stored_filename, page_count):
    return _pdf_upload_serializer().dumps({
        'owner_username': owner_username,
        'report_number': report_number,
        'stored_filename': stored_filename,
        'page_count': int(page_count),
    })


def _load_pdf_upload_token(token):
    try:
        payload = _pdf_upload_serializer().loads(token)
    except BadSignature as exc:
        raise ValueError('PDF upload token is invalid') from exc
    if not isinstance(payload, dict):
        raise ValueError('PDF upload token is invalid')
    return payload


def _validated_prepared_pdf(owner_username, report_number, pdf_upload_token, pdf_page_number):
    """Return a validated stored PDF path, selected page, and page count."""
    prepared_pdf = _load_pdf_upload_token(pdf_upload_token)
    if prepared_pdf.get('owner_username') != owner_username:
        raise ValueError('PDF upload token does not match this report owner')
    if prepared_pdf.get('report_number') != report_number:
        raise ValueError('PDF upload token does not match this report')

    stored_filename = str(prepared_pdf.get('stored_filename') or '')
    if secure_filename(stored_filename) != stored_filename or not stored_filename.endswith('.pdf'):
        raise ValueError('PDF upload token is invalid')
    try:
        page_count = int(prepared_pdf.get('page_count'))
        page_number = int(pdf_page_number)
    except (TypeError, ValueError) as exc:
        raise ValueError('PDF page number must be an integer') from exc
    if page_number < 1 or page_number > page_count:
        raise ValueError(f'PDF page number must be between 1 and {page_count}')

    context = resolve_energy_report_context(
        app.config['UPLOAD_FOLDER'],
        owner_username,
        False,
        report_number,
        requested_owner=owner_username,
    )
    try:
        pdf_path = _validated_report_child_file(
            context.report_dir,
            context.report_dir / stored_filename,
        )
    except FileNotFoundError:
        raise ValueError('Prepared PDF file no longer exists')
    return pdf_path, page_number, page_count


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_exterior_report_dir(context):
    """Resolve one literal report directory without accepting aliases or links."""
    if not isinstance(context, EnergyReportContext):
        raise ValueError('report context is invalid')
    try:
        resolved_report = _canonical_report_directory(context.report_dir)
    except InvalidReportPath:
        if not context.report_dir.exists():
            raise FileNotFoundError('Report does not exist') from None
        raise
    if resolved_report.parent != context.owner_root.resolve(strict=True):
        raise ValueError('report path is invalid')
    try:
        resolved_artifacts = _validated_report_child_directory(
            resolved_report,
            resolved_report / 'vector_pdf_fusion',
        )
    except FileNotFoundError:
        raise FileNotFoundError('Vector PDF fusion artifacts do not exist')
    return resolved_report, resolved_artifacts


def _read_exterior_artifact(artifact_dir, filename):
    try:
        path = resolve_report_child_file(artifact_dir, filename)
    except FileNotFoundError:
        raise FileNotFoundError(f'{filename} does not exist')
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'{filename} is invalid JSON') from exc
    if not isinstance(payload, dict):
        raise ValueError(f'{filename} must contain an object')
    return raw, payload


def _positive_finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite positive number')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f'{name} must be a finite positive number') from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f'{name} must be a finite positive number')
    return number


def _normalized_crop_bbox(value, name):
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f'{name} must be null or four normalized integers')
    left, top, right, bottom = value
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise ValueError(f'{name} must be null or four normalized integers')
    return list(value)


def _strict_provenance(value, name):
    required_fields = {
        'coordinate_space', 'page_number', 'page_size_pt', 'analysis_size_px',
        'crop_bbox_page_px', 'building_roi_px',
    }
    if not isinstance(value, dict) or set(value) != required_fields:
        raise RuntimeError(f'{name} provenance schema is invalid')
    page_number = value.get('page_number')
    if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
        raise RuntimeError(f'{name} provenance page number is invalid')
    page_size = value.get('page_size_pt')
    if not isinstance(page_size, list) or len(page_size) != 2:
        raise RuntimeError(f'{name} provenance page size is invalid')
    for item in page_size:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise RuntimeError(f'{name} provenance page size is invalid')
        try:
            finite = math.isfinite(float(item))
        except (ValueError, OverflowError):
            finite = False
        if not finite or item <= 0:
            raise RuntimeError(f'{name} provenance page size is invalid')
    image_size = value.get('analysis_size_px')
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in image_size)
    ):
        raise RuntimeError(f'{name} provenance image size is invalid')
    try:
        crop_bbox = _normalized_crop_bbox(
            value.get('crop_bbox_page_px'), f'{name} provenance crop_bbox_page_px',
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    expected_space = 'page-local-px' if crop_bbox is None else 'crop-local-px'
    if value.get('coordinate_space') != expected_space:
        raise RuntimeError(f'{name} provenance coordinate space does not match crop')
    roi = value.get('building_roi_px')
    if (
        not isinstance(roi, list)
        or len(roi) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in roi)
    ):
        raise RuntimeError(f'{name} provenance building ROI is invalid')
    left, top, right, bottom = roi
    if (
        left < 0 or top < 0 or right <= left or bottom <= top
        or right > image_size[0] or bottom > image_size[1]
    ):
        raise RuntimeError(f'{name} provenance building ROI is outside analysis bounds')
    return copy.deepcopy(value)


def _strict_value_equal(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _strict_value_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_value_equal(first, second) for first, second in zip(left, right)
        )
    return left == right


def _artifact_provenance(topology, opening_artifact):
    topology_provenance = _strict_provenance(topology.get('provenance'), 'topology')
    opening_provenance = _strict_provenance(
        opening_artifact.get('provenance'), 'opening artifact',
    )
    if not _strict_value_equal(topology_provenance, opening_provenance):
        raise RuntimeError('Exterior artifact provenance does not match')
    return (
        topology_provenance,
        topology_provenance['page_number'],
        topology_provenance['crop_bbox_page_px'],
        list(topology_provenance['analysis_size_px']),
    )


def _axis_segment(item, name):
    try:
        raw_start = item['start_px']
        raw_end = item['end_px']
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f'{name} endpoints are invalid') from exc
    if (
        not isinstance(raw_start, list) or not isinstance(raw_end, list)
        or len(raw_start) != 2 or len(raw_end) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in raw_start + raw_end
        )
    ):
        raise RuntimeError(f'{name} endpoints are invalid')
    start = [float(value) for value in raw_start]
    end = [float(value) for value in raw_end]
    if len(start) != 2 or len(end) != 2 or not all(math.isfinite(value) for value in start + end):
        raise RuntimeError(f'{name} endpoints are invalid')
    if start[1] == end[1] and start[0] != end[0]:
        orientation = 'horizontal'
        fixed = start[1]
        interval = sorted((start[0], end[0]))
    elif start[0] == end[0] and start[1] != end[1]:
        orientation = 'vertical'
        fixed = start[0]
        interval = sorted((start[1], end[1]))
    else:
        raise RuntimeError(f'{name} must be a nonzero axis-aligned segment')
    return orientation, fixed, interval, start, end


def _strict_string_ids(value, name):
    if not isinstance(value, list):
        raise RuntimeError(f'{name} must be a list')
    if any(not isinstance(item, str) or not item for item in value):
        raise RuntimeError(f'{name} must contain non-empty strings')
    if len(value) != len(set(value)):
        raise RuntimeError(f'{name} must contain unique strings')
    return list(value)


def _metric_matches(value, expected):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number) and math.isclose(
        number, expected, rel_tol=0.0, abs_tol=math.ulp(expected) * 4,
    )


def _validate_opening_bindings(topology, opening_artifact):
    source_ids = _strict_string_ids(topology.get('source_wall_ids'), 'source_wall_ids')
    opening_ids = _strict_string_ids(topology.get('opening_ids'), 'opening_ids')
    selected_bridge_ids = _strict_string_ids(topology.get('bridge_ids'), 'bridge_ids')
    bridges = topology.get('bridges')
    if not isinstance(bridges, list) or any(not isinstance(item, dict) for item in bridges):
        raise RuntimeError('Exterior topology bridges are invalid')
    all_bridge_ids = _strict_string_ids(
        [bridge.get('bridge_id') for bridge in bridges], 'topology bridge IDs',
    )
    if set(selected_bridge_ids) != set(all_bridge_ids):
        raise RuntimeError('Selected bridge IDs must exactly match topology bridges')

    accepted = opening_artifact.get('accepted_openings')
    if not isinstance(accepted, list) or any(not isinstance(item, dict) for item in accepted):
        raise RuntimeError('Accepted openings are invalid')
    accepted_ids = _strict_string_ids(
        [opening.get('opening_id') for opening in accepted], 'accepted opening IDs',
    )
    if set(accepted_ids) != set(opening_ids):
        raise RuntimeError('Accepted openings must exactly match topology opening_ids')

    opening_by_id = dict(zip(accepted_ids, accepted))
    opening_bridges = [
        bridge for bridge in bridges if bridge.get('bridge_type') == 'opening_bridge'
    ]
    bridge_opening_ids = _strict_string_ids(
        [bridge.get('opening_id') for bridge in opening_bridges],
        'opening bridge opening IDs',
    )
    if set(bridge_opening_ids) != set(opening_ids):
        raise RuntimeError('Opening bridges must exactly match accepted openings')

    source_id_set = set(source_ids)
    for bridge in opening_bridges:
        opening_id = bridge['opening_id']
        opening = opening_by_id[opening_id]
        bridge_orientation, bridge_fixed, bridge_interval, _, _ = _axis_segment(
            bridge, f'opening bridge {opening_id}',
        )
        opening_orientation, opening_fixed, opening_interval, _, _ = _axis_segment(
            opening, f'opening {opening_id}',
        )
        if (
            bridge.get('orientation') != bridge_orientation
            or opening.get('orientation') != opening_orientation
            or bridge_orientation != opening_orientation
            or bridge_fixed != opening_fixed
            or bridge_interval != opening_interval
        ):
            raise RuntimeError(f'Opening bridge {opening_id} geometry does not match')
        width = bridge_interval[1] - bridge_interval[0]
        if not _metric_matches(bridge.get('width_px'), width) or not _metric_matches(
            opening.get('width_px'), width,
        ):
            raise RuntimeError(f'Opening bridge {opening_id} width does not match endpoints')
        bridge_hosts = _strict_string_ids(
            bridge.get('host_wall_ids'), f'opening bridge {opening_id} host_wall_ids',
        )
        opening_hosts = _strict_string_ids(
            opening.get('host_wall_ids'), f'opening {opening_id} host_wall_ids',
        )
        if (
            not bridge_hosts or not opening_hosts
            or set(bridge_hosts) != set(opening_hosts)
            or not set(opening_hosts).issubset(source_id_set)
        ):
            raise RuntimeError(f'Opening bridge {opening_id} host walls do not match')
        if opening.get('exterior') is not True or opening.get('kind') not in {'door', 'window'}:
            raise RuntimeError(f'Opening {opening_id} is not a valid exterior door or window')
    return accepted


def _validated_inferred_exterior_edges(topology, polygon_edges, perimeter_px, image_size):
    closure_method = topology.get('closure_method')
    inferred_edges = topology.get('inferred_edges') or []
    if closure_method in (None, ''):
        if inferred_edges:
            raise RuntimeError('Legacy exterior topology must not contain inferred edges')
        return [], 0.0, 0.0
    calculation_only = closure_method == 'calculation_only_endpoint_link'
    if closure_method not in {
        'footprint_guided_inference', 'calculation_only_endpoint_link',
    }:
        raise RuntimeError('Exterior topology closure method is invalid')
    if not isinstance(inferred_edges, list) or not inferred_edges or any(
        not isinstance(edge, dict) for edge in inferred_edges
    ):
        raise RuntimeError('Footprint-guided closure has no inferred edges')

    source_ids = set(_strict_string_ids(
        topology.get('source_wall_ids'), 'source_wall_ids',
    ))
    opening_ids = set(_strict_string_ids(
        topology.get('opening_ids') or [], 'opening_ids',
    ))
    inference_ids = _strict_string_ids(
        [edge.get('inference_id') for edge in inferred_edges], 'inferred edge IDs',
    )
    if opening_ids.intersection(inference_ids):
        raise RuntimeError('Inferred edge IDs must not collide with opening IDs')

    groups = {}
    normalized = []
    for inference_id, edge in zip(inference_ids, inferred_edges):
        edge_index, interval, start, end = _boundary_edge_index(
            edge, f'inferred edge {inference_id}', polygon_edges,
        )
        del edge_index
        length = interval[1] - interval[0]
        if not _metric_matches(edge.get('length_px'), length):
            raise RuntimeError(f'inferred edge {inference_id} length does not match endpoints')
        if (
            not calculation_only
            and length > math.nextafter(min(image_size) * 0.10, math.inf)
        ):
            raise RuntimeError(f'inferred edge {inference_id} exceeds the short-side limit')
        anchors = _strict_string_ids(
            edge.get('anchor_wall_ids'), f'inferred edge {inference_id} anchor_wall_ids',
        )
        if not anchors or not set(anchors).issubset(source_ids):
            raise RuntimeError(f'inferred edge {inference_id} anchors are invalid')
        inference_type = edge.get('inference_type')
        if inference_type not in {'collinear_extension', 'orthogonal_corner'}:
            raise RuntimeError(f'inferred edge {inference_id} type is invalid')
        group_id = edge.get('edge_group_id')
        group_size = edge.get('edge_group_size')
        if (
            not isinstance(group_id, str) or not group_id
            or isinstance(group_size, bool) or not isinstance(group_size, int)
            or group_size not in {1, 2}
        ):
            raise RuntimeError(f'inferred edge {inference_id} group is invalid')
        expected_group_size = (
            group_size if calculation_only
            else (1 if inference_type == 'collinear_extension' else 2)
        )
        if group_size != expected_group_size:
            raise RuntimeError(f'inferred edge {inference_id} group is incomplete')
        if edge.get('decision') != 'accepted_candidate':
            raise RuntimeError(f'inferred edge {inference_id} was not accepted')
        inside_direction = edge.get('inside_direction')
        expected_directions = (
            {'up', 'down'} if edge.get('orientation') == 'horizontal'
            else {'left', 'right'}
        )
        if inside_direction not in expected_directions:
            raise RuntimeError(f'inferred edge {inference_id} inside direction is invalid')
        if not calculation_only:
            try:
                inside_mean = float(edge.get('inside_mean'))
                outside_mean = float(edge.get('outside_mean'))
                boundary_mean = float(edge.get('boundary_mean'))
                difference = float(edge.get('inside_outside_difference'))
                inside_support_fraction = float(edge.get('inside_support_fraction'))
                outside_support_fraction = float(edge.get('outside_support_fraction'))
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(f'inferred edge {inference_id} footprint evidence is invalid') from exc
            if not all(math.isfinite(value) for value in (
                inside_mean, outside_mean, boundary_mean, difference,
                inside_support_fraction, outside_support_fraction,
            )) or not _metric_matches(difference, inside_mean - outside_mean):
                raise RuntimeError(f'inferred edge {inference_id} footprint evidence is invalid')
            if (
                any(value < 0.0 or value > 1.0 for value in (
                    inside_mean, outside_mean, boundary_mean,
                    inside_support_fraction, outside_support_fraction,
                ))
                or difference < -1.0 or difference > 1.0
            ):
                raise RuntimeError(f'inferred edge {inference_id} footprint evidence is invalid')
            if inside_mean < 0.50 or difference < 0.25 or boundary_mean < 0.35:
                raise RuntimeError(f'inferred edge {inference_id} footprint support is insufficient')
            if inside_support_fraction < 0.80:
                raise RuntimeError(f'inferred edge {inference_id} footprint support is discontinuous')
            if outside_support_fraction >= 0.50:
                raise RuntimeError(f'inferred edge {inference_id} crosses building interior')
        item = copy.deepcopy(edge)
        item['start_px'] = start
        item['end_px'] = end
        item['length_px'] = float(length)
        groups.setdefault(group_id, []).append(item)
        normalized.append(item)

    for group_id, members in groups.items():
        inference_type = members[0]['inference_type']
        expected_size = (
            int(members[0]['edge_group_size'])
            if calculation_only
            else (1 if inference_type == 'collinear_extension' else 2)
        )
        if len(members) != expected_size or any(
            member['inference_type'] != inference_type for member in members
        ):
            raise RuntimeError(f'inference group {group_id} is incomplete')
        if expected_size == 2:
            if {member['orientation'] for member in members} != {'horizontal', 'vertical'}:
                raise RuntimeError(f'inference group {group_id} is not an orthogonal corner')
            shared_endpoints = set(map(tuple, (
                members[0]['start_px'], members[0]['end_px'],
            ))).intersection(map(tuple, (
                members[1]['start_px'], members[1]['end_px'],
            )))
            if len(shared_endpoints) != 1:
                raise RuntimeError(f'inference group {group_id} does not meet at one corner')

    total_length = math.fsum(edge['length_px'] for edge in normalized)
    ratio = total_length / perimeter_px
    if not calculation_only and ratio > math.nextafter(0.12, math.inf):
        raise RuntimeError('Inferred exterior length exceeds 12% of perimeter')
    if not _metric_matches(topology.get('inferred_length_px'), total_length) or not _metric_matches(
        topology.get('inferred_perimeter_ratio'), ratio,
    ):
        raise RuntimeError('Inferred exterior metrics do not match selected edges')
    return normalized, total_length, ratio


def _validate_confirmable_topology(topology, opening_artifact, scale, image_size):
    if (
        topology.get('format') != 'pdf-exterior-topology/1'
        or topology.get('status') != 'review_required'
        or topology.get('confirmed') is not False
        or topology.get('load_geometry_ready') is not False
    ):
        raise RuntimeError('Exterior topology is not an unconfirmed closed candidate')
    polygon = topology.get('polygon_px')
    if not isinstance(polygon, list) or len(polygon) < 4:
        raise RuntimeError('Exterior topology is not closed')
    polygon_points = []
    for index, point in enumerate(polygon):
        try:
            values = [float(value) for value in point]
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(f'Exterior polygon point {index} is invalid') from exc
        if len(values) != 2 or not all(math.isfinite(value) for value in values):
            raise RuntimeError(f'Exterior polygon point {index} is invalid')
        polygon_points.append(values)
    if polygon_points[0] == polygon_points[-1]:
        polygon_points.pop()
    if len(polygon_points) < 4 or len({tuple(point) for point in polygon_points}) != len(polygon_points):
        raise RuntimeError('Exterior topology is not a single closed candidate')
    width, height = image_size
    if any(
        point[0] < 0 or point[0] > width or point[1] < 0 or point[1] > height
        for point in polygon_points
    ):
        raise RuntimeError('Exterior polygon is outside the analysis image')
    edges = []
    for index, (start, end) in enumerate(zip(polygon_points, polygon_points[1:] + polygon_points[:1])):
        edges.append(_axis_segment(
            {'start_px': start, 'end_px': end}, f'polygon edge {index}',
        )[:3])
    for first_index, first in enumerate(edges):
        for second_index in range(first_index + 1, len(edges)):
            if second_index == first_index + 1 or (
                first_index == 0 and second_index == len(edges) - 1
            ):
                continue
            second = edges[second_index]
            if first[0] == second[0]:
                intersects = first[1] == second[1] and max(
                    first[2][0], second[2][0],
                ) <= min(first[2][1], second[2][1])
            else:
                horizontal, vertical = (
                    (first, second) if first[0] == 'horizontal' else (second, first)
                )
                intersects = (
                    horizontal[2][0] <= vertical[1] <= horizontal[2][1]
                    and vertical[2][0] <= horizontal[1] <= vertical[2][1]
                )
            if intersects:
                raise RuntimeError('Exterior polygon has non-adjacent edge intersections')
    signed_area = 0.5 * sum(
        start[0] * end[1] - end[0] * start[1]
        for start, end in zip(polygon_points, polygon_points[1:] + polygon_points[:1])
    )
    if not math.isfinite(signed_area) or signed_area == 0:
        raise RuntimeError('Exterior topology is not a single closed candidate')
    area_px2 = abs(signed_area)
    perimeter_px = math.fsum(edge[2][1] - edge[2][0] for edge in edges)
    if not _metric_matches(topology.get('area_px2'), area_px2) or not _metric_matches(
        topology.get('perimeter_px'), perimeter_px,
    ):
        raise RuntimeError('Exterior topology metrics do not match its polygon')
    if topology.get('unresolved_gaps') or topology.get('unresolved'):
        raise RuntimeError('Exterior topology still has unresolved gaps')
    if not isinstance(topology.get('source_wall_ids'), list) or not topology['source_wall_ids']:
        raise RuntimeError('Exterior topology has no real exterior wall evidence')
    inferred_edges, inferred_length_px, inferred_perimeter_ratio = (
        _validated_inferred_exterior_edges(topology, edges, perimeter_px, image_size)
    )
    if opening_artifact.get('format') != 'pdf-opening-candidates/1':
        raise RuntimeError('Opening artifact format is invalid')
    if (
        opening_artifact.get('confirmed') is not False
        or opening_artifact.get('load_geometry_ready') is not False
        or opening_artifact.get('ambiguous_openings')
    ):
        raise RuntimeError('Opening artifact is not confirmable')

    bridges = topology.get('bridges')
    if not isinstance(bridges, list):
        raise RuntimeError('Exterior topology bridges are invalid')
    for index, bridge in enumerate(bridges):
        if not isinstance(bridge, dict):
            raise RuntimeError(f'bridge {index} is invalid')
        if bridge.get('bridge_type') == 'small_gap_repair':
            _, _, interval, _, _ = _axis_segment(bridge, f'small gap repair {index}')
            length_m = (interval[1] - interval[0]) * scale
            if not math.isfinite(length_m) or length_m > math.nextafter(0.6, math.inf):
                raise RuntimeError('Small gap repair exceeds 0.6 m')
    return (
        polygon_points, area_px2, perimeter_px,
        inferred_edges, inferred_length_px, inferred_perimeter_ratio,
    )


def _boundary_edge_index(item, name, edges):
    orientation, fixed, interval, start, end = _axis_segment(item, name)
    if item.get('orientation') != orientation:
        raise RuntimeError(f'{name} orientation does not match endpoints')
    matches = [
        index for index, edge in enumerate(edges)
        if edge[0] == orientation and edge[1] == fixed
        and edge[2][0] <= interval[0] < interval[1] <= edge[2][1]
    ]
    if len(matches) != 1:
        raise RuntimeError(f'{name} must lie completely on exactly one polygon edge')
    return matches[0], interval, start, end


def _validated_real_exterior_wall_geometry(polygon_points, topology, inferred_edges=None):
    edges = [
        _axis_segment({'start_px': start, 'end_px': end}, f'polygon edge {index}')[:3]
        for index, (start, end) in enumerate(
            zip(polygon_points, polygon_points[1:] + polygon_points[:1]), 1,
        )
    ]
    source_ids = set(_strict_string_ids(
        topology.get('source_wall_ids'), 'source_wall_ids',
    ))
    selected_bridge_ids = _strict_string_ids(
        topology.get('bridge_ids'), 'bridge_ids',
    )
    bridges = topology.get('bridges')
    if not isinstance(bridges, list) or any(not isinstance(item, dict) for item in bridges):
        raise RuntimeError('Exterior topology bridges are invalid')
    bridge_ids = _strict_string_ids(
        [bridge.get('bridge_id') for bridge in bridges], 'topology bridge IDs',
    )
    bridge_by_id = dict(zip(bridge_ids, bridges))
    if set(selected_bridge_ids) != set(bridge_by_id):
        raise RuntimeError('Selected bridge IDs must resolve exactly once with no extras')
    if any(
        bridge.get('bridge_type') not in {'opening_bridge', 'small_gap_repair'}
        for bridge in bridges
    ):
        raise RuntimeError('Exterior topology contains an unknown bridge type')

    edge_parts = [[] for _ in edges]
    for bridge_id in selected_bridge_ids:
        bridge = bridge_by_id[bridge_id]
        edge_index, interval, _, _ = _boundary_edge_index(
            bridge, f'bridge {bridge_id}', edges,
        )
        if not _metric_matches(bridge.get('width_px'), interval[1] - interval[0]):
            raise RuntimeError(f'bridge {bridge_id} width does not match endpoints')
        bridge_hosts = _strict_string_ids(
            bridge.get('host_wall_ids'), f'bridge {bridge_id} host_wall_ids',
        )
        if not bridge_hosts or not set(bridge_hosts).issubset(source_ids):
            raise RuntimeError(f'bridge {bridge_id} host walls are invalid')
        if bridge['bridge_type'] == 'small_gap_repair' and bridge.get('opening_id') is not None:
            raise RuntimeError(f'bridge {bridge_id} repair must not reference an opening')
        edge_parts[edge_index].append((interval[0], interval[1], f'bridge {bridge_id}'))

    for inferred in inferred_edges or []:
        inference_id = inferred['inference_id']
        edge_index, interval, _, _ = _boundary_edge_index(
            inferred, f'inferred edge {inference_id}', edges,
        )
        edge_parts[edge_index].append((
            interval[0], interval[1], f'inferred edge {inference_id}',
        ))

    real_segments = topology.get('real_wall_segments')
    if not isinstance(real_segments, list) or not real_segments or any(
        not isinstance(item, dict) for item in real_segments
    ):
        raise RuntimeError('real_wall_segments must be a non-empty list')
    segment_ids = _strict_string_ids(
        [segment.get('segment_id') for segment in real_segments], 'real wall segment IDs',
    )
    walls = []
    for segment_id, segment in zip(segment_ids, real_segments):
        edge_index, interval, start, end = _boundary_edge_index(
            segment, f'real wall segment {segment_id}', edges,
        )
        length = interval[1] - interval[0]
        if not _metric_matches(segment.get('length_px'), length):
            raise RuntimeError(f'real wall segment {segment_id} length does not match endpoints')
        segment_sources = _strict_string_ids(
            segment.get('source_wall_ids'),
            f'real wall segment {segment_id} source_wall_ids',
        )
        if not segment_sources or not set(segment_sources).issubset(source_ids):
            raise RuntimeError(f'real wall segment {segment_id} source walls are invalid')
        edge_parts[edge_index].append((interval[0], interval[1], f'real wall {segment_id}'))
        wall = copy.deepcopy(segment)
        wall.update({
            'id': segment_id,
            'pts': [start, end],
            'exterior': True,
            'real_wall': True,
        })
        walls.append(wall)

    for edge, parts in zip(edges, edge_parts):
        parts.sort(key=lambda item: (item[0], item[1], item[2]))
        cursor = edge[2][0]
        for interval_start, interval_end, label in parts:
            if interval_start != cursor:
                condition = 'overlaps' if interval_start < cursor else 'leaves a gap in'
                raise RuntimeError(f'{label} {condition} the polygon boundary partition')
            cursor = interval_end
        if cursor != edge[2][1]:
            raise RuntimeError(
                'Real walls, selected bridges and inferred edges do not cover the polygon boundary'
            )
    return walls


def _atomic_write_json(path, payload):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=path.parent,
            prefix=f'.{path.name}.', suffix='.tmp', delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


class ExteriorGenerationConflict(RuntimeError):
    """The confirmed exterior no longer belongs to the current fusion generation."""


_exterior_thread_locks_guard = threading.Lock()
_exterior_thread_locks = {}


@contextmanager
def _exterior_report_lock(report_dir):
    """Serialize exterior fusion/confirmation/use for one report across workers."""
    report_dir = _canonical_report_directory(report_dir)
    lock_key = os.fspath(report_dir)
    with _exterior_thread_locks_guard:
        thread_lock = _exterior_thread_locks.setdefault(lock_key, threading.RLock())

    with thread_lock:
        try:
            lock_fd = open_private_report_lock_file(
                report_dir, '.exterior_generation.lock',
            )
        except InvalidReportPath:
            raise
        except OSError as exc:
            raise InvalidReportPath('exterior lock file is invalid') from exc
        with os.fdopen(lock_fd, 'r+b') as lock_file:
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b'0')
                lock_file.flush()
            lock_file.seek(0)
            if os.name == 'nt':
                import msvcrt
                while True:
                    try:
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        time.sleep(0.05)
                try:
                    yield
                finally:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _exterior_generation_path(report_dir):
    return resolve_report_child_destination(report_dir, 'exterior_generation.json')


def _read_exterior_generation(report_dir):
    try:
        path = resolve_report_child_file(report_dir, 'exterior_generation.json')
    except FileNotFoundError:
        return None
    try:
        marker = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return marker if isinstance(marker, dict) else None


def _new_exterior_generation(report_number):
    return {
        'format': 'pdf-exterior-generation/1',
        'report_number': report_number,
        'generation': uuid.uuid4().hex,
        'status': 'running',
        'topology_sha256': None,
        'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }


def _write_exterior_generation(report_dir, marker):
    marker = copy.deepcopy(marker)
    marker['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    _atomic_write_json(_exterior_generation_path(report_dir), marker)
    return marker


def _update_exterior_generation_if_current(report_dir, generation, *, status, topology_sha256=None):
    marker = _read_exterior_generation(report_dir)
    if not marker or marker.get('generation') != generation:
        return False
    marker['status'] = status
    marker['topology_sha256'] = topology_sha256 if status == 'ready' else None
    _write_exterior_generation(report_dir, marker)
    return True


def _require_current_exterior_generation(
    report_dir,
    report_number,
    *,
    recognition=None,
    topology_sha256=None,
    expected_generation=None,
):
    marker = _read_exterior_generation(report_dir)
    if not marker:
        raise ExteriorGenerationConflict('Exterior generation marker is missing; run fusion again')
    marker_generation = marker.get('generation')
    marker_hash = marker.get('topology_sha256')
    if (
        marker.get('format') != 'pdf-exterior-generation/1'
        or marker.get('report_number') != report_number
        or not isinstance(marker_generation, str)
        or not marker_generation
        or marker.get('status') != 'ready'
        or not isinstance(marker_hash, str)
        or len(marker_hash) != 64
        or any(character not in '0123456789abcdef' for character in marker_hash)
    ):
        raise ExteriorGenerationConflict('Exterior generation is not ready for this report')
    if expected_generation is not None and marker_generation != expected_generation:
        raise ExteriorGenerationConflict('Exterior generation changed during confirmation')
    if topology_sha256 is not None and not hmac.compare_digest(marker_hash, topology_sha256):
        raise ExteriorGenerationConflict('Exterior generation topology hash does not match')
    if recognition is not None:
        binding = recognition.get('exterior_generation') or {}
        if (
            recognition.get('report_number') != report_number
            or binding.get('generation') != marker_generation
            or binding.get('topology_sha256') != marker_hash
        ):
            raise ExteriorGenerationConflict('Recognition generation does not match current exterior generation')
    return marker


def _serialized_exterior_report(route):
    """Hold the same report lock for each mutating exterior route."""
    @wraps(route)
    def wrapped(*args, **kwargs):
        if request.is_json:
            supplied = (request.get_json(silent=True) or {}).get('report_number')
        else:
            supplied = request.form.get('report_number', 'default')
        try:
            context = _energy_report_context(
                supplied,
                _requested_owner_username(),
            )
        except _REPORT_STORAGE_EXCEPTIONS as exc:
            return _report_storage_error_response(exc)
        with _exterior_report_lock(context.report_dir):
            return route(*args, **kwargs)
    return wrapped


def _finalize_exterior_confirmation_response(context, response):
    """Persist a confirmation terminal state before its report lock releases."""

    response = app.make_response(response)
    if context is not None:
        status = 'recognized' if 200 <= response.status_code < 300 else 'failed'
        _update_report_status_if_exists(
            context.owner_username, context.report_number, status,
        )
    return response


from routes import register_routes
import sys as _sys
globals().update(register_routes(app, _sys.modules[__name__]))

if __name__ == '__main__':
    _sys.modules.setdefault('web_server_server', _sys.modules[__name__])
    if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)
    app.run(**local_run_options())
