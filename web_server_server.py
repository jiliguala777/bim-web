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
from flask import Flask, request, render_template, jsonify, send_file, send_from_directory, session, redirect, url_for
from itsdangerous import BadSignature, URLSafeSerializer
from werkzeug.utils import secure_filename
from functools import wraps
import logging
import numpy as np
import psutil
import subprocess
from contextlib import contextmanager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
WEATHER_DATA_DIR = os.path.join(BASE_DIR, 'weather_data')
FLOORPLAN_MODEL_NAME = 'M2_UNet_ResNet34_DA'
ONNX_MODEL_PATH = os.environ.get('ONNX_MODEL_PATH', os.path.join(MODELS_DIR, 'M2_pub_plus_user.onnx'))
FLOORPLAN_MODEL_VERSION = os.path.basename(ONNX_MODEL_PATH)
FLOORPLAN_MODEL_MIOU = 0.787
RECOGNITION_SCHEMA_VERSION = 1
LEGACY_ONNX_BACKEND = 'legacy_onnx'
VECTOR_PYTORCH_BACKEND = 'vector_pytorch'

# --- IFC / Benchmark 模块 ---
try:
    import sys
    sys.path.insert(0, os.path.join(BASE_DIR, 'deploy'))
    sys.path.insert(0, BASE_DIR)
    from ifc_parser import IFCParser, HAS_IFC
    from benchmark import run_benchmark, run_simple_model, generate_test_dxf, generate_test_ifc, BESTEST_BENCHMARKS
    HAS_BENCHMARK = True
except ImportError as e:
    HAS_IFC = False
    HAS_BENCHMARK = False
    print(f"IFC/Benchmark import warning: {e}")

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
    import energyplus_engine  # 自定义高级引擎
    HAS_ENERGY_DEPS = True
    
    # 尝试加载高级识别模块
    try:
        import sys
        sys.path.append(BASE_DIR)
        sys.path.append(os.path.join(BASE_DIR, 'deploy'))
        from cad_advanced import AdvancedCADRecognition
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
app.secret_key = os.environ.get('SECRET_KEY') or secrets.token_hex(32)


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
JOBS_DIR = '/home/ubuntu/jobs'
JOBS_DIR = os.environ.get('JOBS_DIR', os.path.join(BASE_DIR, 'jobs'))
os.makedirs(JOBS_DIR, exist_ok=True)

class JobStore:
    def get(self, job_id):
        path = os.path.join(JOBS_DIR, f"{job_id}.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None
        
    def set(self, job_id, data):
        path = os.path.join(JOBS_DIR, f"{job_id}.json")
        with open(path, 'w') as f:
            json.dump(data, f)

    def update(self, job_id, **kwargs):
        job = self.get(job_id)
        if job:
            job.update(kwargs)
            self.set(job_id, job)

simulation_jobs = JobStore()

# --- Authentication Decorator ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
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


def _user_exists(username):
    conn = get_db_connection()
    try:
        return conn.execute(
            "SELECT 1 FROM users WHERE username = ?",
            (username,),
        ).fetchone() is not None
    finally:
        conn.close()


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
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        # 特权管理员 (凭证从环境变量读取；未设置则不启用此后备登录)
        admin_user = os.environ.get('ADMIN_USER', 'admin')
        admin_password = os.environ.get('ADMIN_PASSWORD')
        if admin_password and username == admin_user and password == admin_password:
            session['logged_in'] = True
            session['username'] = username
            session['is_admin'] = True
            session.permanent = True
            return jsonify({'status': 'success'})
            
        # 数据库查询
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password_hash'], password):
            session['logged_in'] = True
            session['username'] = username
            session['is_admin'] = False
            session.permanent = True
            return jsonify({'status': 'success'})
            
        return jsonify({'status': 'fail', 'message': '用户名或密码错误'}), 401
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'status': 'fail', 'message': '请填写完整信息'}), 400
            
        hashed_pw = generate_password_hash(password)
        
        try:
            conn = get_db_connection()
            conn.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, hashed_pw))
            conn.commit()
            conn.close()
            return jsonify({'status': 'success'})
        except sqlite3.IntegrityError:
            return jsonify({'status': 'fail', 'message': '该用户名已被占用'}), 400
        except Exception as e:
            return jsonify({'status': 'fail', 'message': str(e)}), 500
            
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ==========================================
# 路由 - 功能入口
# ==========================================
@app.route('/')
def landing():
    """ 公开着陆页 - 包含功能介绍与登录 """
    return render_template('landing.html', logged_in=('logged_in' in session))

@app.route('/energy')
@login_required
def index():
    return render_template('energy.html', has_deps=HAS_ENERGY_DEPS)

@app.route('/home')
@login_required
def home():
    return render_template('index.html')

@app.route('/floorplan')
@login_required
def floorplan():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': time.time()})

@app.route('/api/v1/simulation_data/<report_number>')
@login_required
def get_simulation_data(report_number):
    """
    Returns extracted geometry and simulation constants for local API bridge.
    Confidential device data is NOT handled here.
    """
    target_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'energy', report_number)
    dxf_path = os.path.join(target_dir, 'building_plan.dxf')
    
    # In a real scenario, we'd store the last simulation results in a DB or JSON file
    # For now, we'll re-extract or return defaults
    return jsonify({
        'report_number': report_number,
        'geometry': {
            'floor_area': 500, # Placeholder or extracted value
            'perimeter': 100
        },
        'env_params': {
            'u_wall': 0.6,
            'u_win': 2.5
        }
    })


@app.route('/energy/library/walls', methods=['GET'])
@login_required
def energy_library_walls():
    if not HAS_BUILDING_LIBRARY:
        return jsonify({'error': 'Building library module is not available'}), 503
    try:
        return jsonify({'items': list_wall_assemblies()})
    except Exception as e:
        logger.error(f"Load wall library error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/energy/library/materials', methods=['GET'])
@login_required
def energy_library_materials():
    if not HAS_BUILDING_LIBRARY:
        return jsonify({'error': 'Building library module is not available'}), 503
    try:
        return jsonify({'items': list_materials()})
    except Exception as e:
        logger.error(f"Load material library error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/energy/library/envelope/<category>', methods=['GET'])
@login_required
def energy_library_envelope(category):
    if not HAS_BUILDING_LIBRARY:
        return jsonify({'error': 'Building library module is not available'}), 503
    allowed_categories = {
        'exterior_wall',
        'window_u',
        'roof_u',
        'floor_u',
        'window_shgc',
        'curtain_shading',
        'door_u',
        'floor_contact_type',
        'window_air_tightness'
    }
    if category not in allowed_categories:
        return jsonify({'error': 'Unsupported envelope parameter category'}), 400
    try:
        return jsonify({'items': list_envelope_parameters(category)})
    except Exception as e:
        logger.error(f"Load envelope library error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# ==========================================
# 路由 - 文件上传
# ==========================================
@app.route('/energy/upload', methods=['POST'])
@login_required
def energy_upload():
    try:
        report_number = request.form.get('report_number', 'default')
        report_number = secure_filename(report_number) or 'default'
        target_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'energy', report_number)
        os.makedirs(target_dir, exist_ok=True)
        
        result = {'status': 'success', 'files': {}}
        dxf_uploaded = False
        dxf_path = os.path.join(target_dir, 'building_plan.dxf') # Define dxf_path early
        
        if 'dxf_file' in request.files:
            f = request.files['dxf_file']
            if f and allowed_file(f.filename, ALLOWED_DXF):
                f.save(dxf_path)
                result['files']['dxf'] = 'building_plan.dxf'
                dxf_uploaded = True
        
        # 处理光栅图/PDF上传 (Learning-based 接口)
        raster_uploaded = False
        if 'raster_file' in request.files:
            f = request.files['raster_file']
            if f and allowed_file(f.filename, ALLOWED_RASTER):
                ext = f.filename.rsplit('.', 1)[1].lower()
                raster_path = os.path.join(target_dir, f'building_plan.{ext}')
                f.save(raster_path)
                result['files']['raster'] = f'building_plan.{ext}'
                raster_uploaded = True
                
                # 如果是 PDF，尝试转换为 PNG 进行后续解析
                if ext == 'pdf':
                    try:
                        from pdf2image import convert_from_path
                        images = convert_from_path(
                            raster_path,
                            poppler_path=_resolve_poppler_path(),
                        )
                        if images:
                            images[0].save(os.path.join(target_dir, 'building_plan.png'), 'PNG')
                            result['files']['raster_converted'] = 'building_plan.png'
                    except Exception as e:
                        logger.warning(f"PDF to Image conversion failed: {e}")
        
        # 仅在本次请求确实上传了新文件时才解析层级信息
        if dxf_uploaded and os.path.exists(dxf_path):
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
            layers = {}
            all_geoms = {} # layer_name -> list of pts
            
            for e in msp:
                layer = e.dxf.layer
                layers[layer] = layers.get(layer, 0) + 1
                if layer not in all_geoms: all_geoms[layer] = []
                
                if e.dxftype() == 'LINE':
                    all_geoms[layer].append({
                        'type': 'line',
                        'pts': [(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]
                    })
                elif e.dxftype() == 'LWPOLYLINE':
                    all_geoms[layer].append({
                        'type': 'polyline',
                        'pts': [(p[0], p[1]) for p in e.get_points(format='xy')]
                    })
            
            result['layers'] = [{'name': k, 'count': v} for k,v in layers.items()]
            result['all_geoms'] = all_geoms
            result['status'] = 'success'
        elif not dxf_uploaded and not raster_uploaded and not 'epw_file' in request.files:
             # 如果啥也没传，明确告知
             result['message'] = 'No new files uploaded in this request.'

        if 'epw_file' in request.files:
            f = request.files['epw_file']
            if f and allowed_file(f.filename, ALLOWED_EPW):
                p = os.path.join(target_dir, 'weather_data.epw')
                f.save(p)
                result['files']['epw'] = 'weather_data.epw'
        elif request.form.get('city_id'):
            city_id = request.form['city_id']
            src = os.path.join(WEATHER_DATA_DIR, f'{city_id}.epw')
            if os.path.exists(src):
                import shutil
                shutil.copy(src, os.path.join(target_dir, 'weather_data.epw'))
                result['files']['epw'] = 'weather_data.epw'

        return jsonify(result)
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/energy/layers/geometry', methods=['GET'])
@login_required
def get_all_layers_geometry():
    """获取所有图层的几何数据及边框"""
    try:
        report_number = request.args.get('project', 'default')
        target_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'energy', report_number)
        dxf_path = os.path.join(target_dir, 'building_plan.dxf')
        
        if not os.path.exists(dxf_path):
            return jsonify({'error': 'DXF file not found'}), 404
            
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        
        layers_data = {}
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        for e in msp:
            layer = e.dxf.layer
            if layer not in layers_data: layers_data[layer] = []
            
            pts = []
            if e.dxftype() == 'LINE':
                pts = [(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]
            elif e.dxftype() == 'LWPOLYLINE':
                pts = [(p[0], p[1]) for p in e.get_points(format='xy')]
            
            if pts:
                layers_data[layer].append({'points': pts, 'closed': getattr(e, 'is_closed', False)})
                for px, py in pts:
                    min_x, min_y = min(min_x, px), min(min_y, py)
                    max_x, max_y = max(max_x, px), max(max_y, py)
        
        bounds = {
            'minX': min_x, 'minY': min_y, 
            'width': max_x - min_x if max_x != float('-inf') else 1,
            'height': max_y - min_y if max_y != float('-inf') else 1
        }
        return jsonify({'layers': layers_data, 'bounds': bounds})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/energy/geometry', methods=['POST'])
@login_required
def energy_geometry():
    """获取指定图层的几何数据用于预览"""
    try:
        data = request.get_json()
        report_number = secure_filename(data.get('report_number', 'default')) or 'default'
        layers = data.get('layers', [])
        scale = float(data.get('scale', 1.0))
        
        target_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'energy', report_number)
        dxf_path = os.path.join(target_dir, 'building_plan.dxf')
        
        if not os.path.exists(dxf_path):
            return jsonify({'error': 'DXF file not found'}), 404
            
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        
        geoms = []
        for e in msp:
            if e.dxf.layer in layers:
                if e.dxftype() == 'LINE':
                    geoms.append({
                        'type': 'line',
                        'pts': [(e.dxf.start.x * scale, e.dxf.start.y * scale), 
                                (e.dxf.end.x * scale, e.dxf.end.y * scale)]
                    })
                elif e.dxftype() == 'LWPOLYLINE':
                    geoms.append({
                        'type': 'polyline',
                        'pts': [(p[0] * scale, p[1] * scale) for p in e.get_points(format='xy')]
                    })
        return jsonify({'geoms': geoms})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/energy/geometry_advanced', methods=['POST'])
@login_required
def energy_geometry_advanced():
    """
    使用高级识别算法 (Alpha Shape + Dilate-Subtract) 获取几何数据
    """
    try:
        data = request.get_json()
        report_number = data.get('report_number', 'default')
        scale = float(data.get('scale', 1.0))
        
        target_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'energy', report_number)
        dxf_path = os.path.join(target_dir, 'building_plan.dxf')
        
        if not os.path.exists(dxf_path):
            return jsonify({'error': 'DXF file not found'}), 404
            
        if AdvancedCADRecognition is None:
            return jsonify({'error': 'Advanced CAD Recognition module or dependencies (OpenCV) not found'}), 512

        # 初始化高级引擎
        engine = AdvancedCADRecognition(dxf_path)
        
        # 如果有位图则使用位图路由，否则使用矢量启发式路由 (混合模式)
        png_path = os.path.join(target_dir, 'building_plan.png')
        jpg_path = os.path.join(target_dir, 'building_plan.jpg')
        
        raster_path = png_path if os.path.exists(png_path) else (jpg_path if os.path.exists(jpg_path) else None)
        
        if raster_path:
            # 方案 A: 基于图像的语义识别 (Learning-based)
            # 这里调用报告中提到的技术路线
            try:
                result = engine.run_advanced_pipeline(raster_path)
                walls = [{"type": "polyline", "pts": [[p[0]*scale, p[1]*scale] for p in result['walls']]}]
            except Exception as e:
                logger.warning(f"Raster recognition failed, falling back to vector: {e}")
                raster_path = None # 触发 Fallback
        
        if not raster_path:
            # 方案 B: 基于矢量的启发式识别 (Vector-based)
            # 处理传统的 CAD 矢量提取
            doc = ezdxf.readfile(dxf_path)
            msp = doc.modelspace()
            lines = []
            for e in msp:
                if e.dxftype() in ['LINE', 'LWPOLYLINE']:
                    # 使用报告中的 '墙体最小面积' 逻辑的矢量变体
                    # 简化：获取所有线条并执行缓冲区合并
                    if e.dxftype() == 'LINE':
                        lines.append(LineString([(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]))
                    else:
                        pts = [(p[0], p[1]) for p in e.get_points(format='xy')]
                        if len(pts) > 1: lines.append(LineString(pts))
            
            # 使用 Alpha Shape 逻辑 (Shapely buffer 0.1 + union)
            merged = unary_union([l.buffer(2.0) for l in lines]) # 2.0 像素宽度兼容
            if isinstance(merged, MultiPolygon):
                poly = max(merged.geoms, key=lambda p: p.area)
            else:
                poly = merged
            
            # 正则化
            coords = list(poly.exterior.coords)
            walls = [{"type": "polyline", "pts": [[p[0]*scale, p[1]*scale] for p in coords]}]

        return jsonify({'geoms': walls, 'mode': 'advanced'})
    except Exception as e:
        logger.error(f"Advanced geometry error: {e}")
        return jsonify({'error': str(e)}), 500


# ==========================================
# 路由 - 模拟计算 (异步化)
# ==========================================
@app.route('/energy/calculate', methods=['POST'])
@login_required
def energy_calculate():
    """启动计算任务 (支持 Simple 和 EnergyPlus)"""
    data = request.get_json()
    job_id = str(uuid.uuid4())
    mode = data.get('mode', 'simple') # 默认简单模式
    
    # 初始化状态
    simulation_jobs.set(job_id, {
        'status': 'processing',
        'progress': 0,
        'result': None,
        'error': None
    })

    
    if mode == 'simple':
        # 简单模式直接在主线程或轻量级线程处理计算 (此处为了统一逻辑仍用线程)
        thread = threading.Thread(target=simple_simulation_task, args=(job_id, data))
    else:
        thread = threading.Thread(target=background_simulation_task, args=(job_id, data))
        
    thread.start()
    return jsonify({'job_id': job_id})

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
        
        report_number = data.get('report_number', 'default')
        target_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'energy', report_number)
        dxf_path = os.path.join(target_dir, 'building_plan.dxf')
        
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
    except Exception as e:
        simulation_jobs.update(job_id, status='failed', error=str(e))


@app.route('/energy/status/<job_id>')
@login_required
def energy_status(job_id):
    job = simulation_jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(job)

def background_simulation_task(job_id, data):
    """在后台执行复杂的能耗模拟逻辑"""
    try:
        report_number = data.get('report_number', 'default')
        target_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'energy', report_number)
        dxf_path = os.path.join(target_dir, 'building_plan.dxf')
        epw_path = os.path.join(target_dir, 'weather_data.epw')
        
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
        output_dir = os.path.join(target_dir, 'energyplus_runs', job_id)

        idf_path = os.path.join(output_dir, 'run.idf')
        os.makedirs(output_dir, exist_ok=True)
        
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

    except Exception as e:
        logger.error(f"Simulation task error: {e}", exc_info=True)
        simulation_jobs.update(job_id, status='failed', error=str(e))

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
@app.route('/energy/upload_ifc', methods=['POST'])
@login_required
def upload_ifc():
    """上传 IFC 文件并解析建筑信息"""
    if not HAS_IFC:
        return jsonify({'error': 'IFC support not available (ifcopenshell not installed)'}), 501
    
    try:
        report_number = request.form.get('report_number', 'default')
        report_number = secure_filename(report_number) or 'default'
        target_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'energy', report_number)
        os.makedirs(target_dir, exist_ok=True)
        
        if 'ifc_file' not in request.files:
            return jsonify({'error': 'No IFC file provided'}), 400
        
        f = request.files['ifc_file']
        if not f or not allowed_file(f.filename, ALLOWED_IFC):
            return jsonify({'error': 'Invalid file type. Only .ifc files accepted'}), 400
        
        ifc_path = os.path.join(target_dir, 'building_model.ifc')
        f.save(ifc_path)
        
        # 解析 IFC
        parser = IFCParser(ifc_path)
        
        result = {
            'status': 'success',
            'source': 'IFC',
            'project_info': parser.get_project_info(),
            'element_summary': parser.get_element_summary(),
            'storeys': parser.get_storeys(),
            'simulation_data': parser.extract_for_simulation(),
        }
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"IFC upload error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/energy/ifc_properties', methods=['GET'])
@login_required
def ifc_properties():
    """获取已上传 IFC 模型的属性树"""
    if not HAS_IFC:
        return jsonify({'error': 'IFC support not available'}), 501
    
    try:
        report_number = request.args.get('project', 'default')
        target_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'energy', report_number)
        ifc_path = os.path.join(target_dir, 'building_model.ifc')
        
        if not os.path.exists(ifc_path):
            return jsonify({'error': 'IFC file not found. Upload one first.'}), 404
        
        parser = IFCParser(ifc_path)
        tree = parser.get_property_tree()
        
        return jsonify(tree)
    except Exception as e:
        logger.error(f"IFC property tree error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/energy/ifc_walls', methods=['GET'])
@login_required
def ifc_walls():
    """获取 IFC 模型中的墙体列表及热工参数"""
    if not HAS_IFC:
        return jsonify({'error': 'IFC support not available'}), 501
    
    try:
        report_number = request.args.get('project', 'default')
        target_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'energy', report_number)
        ifc_path = os.path.join(target_dir, 'building_model.ifc')
        
        if not os.path.exists(ifc_path):
            return jsonify({'error': 'IFC file not found'}), 404
        
        parser = IFCParser(ifc_path)
        walls = parser.get_walls()
        windows = parser.get_windows()
        spaces = parser.get_spaces()
        
        return jsonify({
            'walls': walls,
            'windows': windows,
            'spaces': spaces,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/energy/ifc_simulate', methods=['POST'])
@login_required
def ifc_simulate():
    """使用 IFC 提取的数据直接运行能耗模拟"""
    if not HAS_IFC:
        return jsonify({'error': 'IFC support not available'}), 501
    
    try:
        data = request.get_json() or {}
        report_number = data.get('report_number', 'default')
        target_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'energy', report_number)
        ifc_path = os.path.join(target_dir, 'building_model.ifc')
        
        if not os.path.exists(ifc_path):
            return jsonify({'error': 'IFC file not found'}), 404
        
        parser = IFCParser(ifc_path)
        sim_data = parser.extract_for_simulation()
        
        # 构造简单模型输入
        case_data = {
            'geometry': {
                'floor_area': sim_data['geometry']['floor_area'] / max(sim_data['building'].get('floors', 1), 1),
                'height': 3.0,
                'floors': sim_data['building'].get('floors', 1),
                'wall_area': sim_data['geometry']['wall_area'],
                'window_area': sim_data['geometry']['window_area'],
                'roof_area': sim_data['geometry']['floor_area'] / max(sim_data['building'].get('floors', 1), 1),
            },
            'thermal': {
                'u_wall': data.get('u_wall', sim_data['thermal']['u_wall']),
                'u_window': data.get('u_window', sim_data['thermal']['u_window']),
                'u_roof': data.get('u_roof', 0.4),
                'u_floor': data.get('u_floor', 0.3),
                'infiltration_ach': data.get('infiltration_ach', 0.5),
            },
            'climate': {
                'city': data.get('city', 'Beijing'),
                'hdd_18': 2800,
                'cdd_26': 500,
            },
        }
        
        # 根据城市查真实度日数
        weather_map = {
            'Beijing': {'hdd_18': 2800, 'cdd_26': 500},
            'Shanghai': {'hdd_18': 1500, 'cdd_26': 1100},
            'Shenzhen': {'hdd_18': 500, 'cdd_26': 1800},
        }
        if data.get('city') in weather_map:
            case_data['climate'].update(weather_map[data['city']])
        
        result = run_simple_model(case_data)
        result['ifc_data'] = sim_data
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"IFC simulation error: {e}")
        return jsonify({'error': str(e)}), 500


# ==========================================
# 路由 - 模型精度基准测试
# ==========================================
@app.route('/ops/benchmark', methods=['GET', 'POST'])
def ops_benchmark():
    """运行模型精度基准测试"""
    if not HAS_BENCHMARK:
        return jsonify({'error': 'Benchmark module not available'}), 501
    
    try:
        case_name = request.args.get('case')
        if not case_name and request.method == 'POST':
            case_name = (request.get_json(silent=True) or {}).get('case')
        report = run_benchmark(case_name)
        return jsonify(report)
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/ops/benchmark/generate_test_files', methods=['POST'])
@login_required
def generate_test_files():
    """生成 BESTEST 标准测试文件 (DXF + IFC)"""
    try:
        test_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'energy', 'bestest')
        os.makedirs(test_dir, exist_ok=True)
        
        files = {}
        for case_name in BESTEST_BENCHMARKS:
            dxf_path = os.path.join(test_dir, f'{case_name}.dxf')
            ifc_path = os.path.join(test_dir, f'{case_name}.ifc')
            
            if generate_test_dxf(dxf_path, case_name):
                files[f'{case_name}.dxf'] = 'generated'
            
            if HAS_IFC and generate_test_ifc(ifc_path, case_name):
                files[f'{case_name}.ifc'] = 'generated'
        
        return jsonify({'status': 'success', 'files': files, 'directory': test_dir})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==========================================
# 路由 - 运维监控 API (供 OpenClaw Agent 调用)
# ==========================================
@app.route('/ops/health', methods=['GET'])
def ops_health():
    """综合健康检查"""
    health = {
        'timestamp': time.time(),
        'services': {},
        'system': {},
    }
    
    # Helper: scan processes with psutil
    def _count_procs(name_pattern):
        count = 0
        for proc in psutil.process_iter(['name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info.get('cmdline') or [])
                if name_pattern in cmdline or name_pattern in (proc.info.get('name') or ''):
                    count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return count
    
    # 1. Gunicorn
    gunicorn_count = _count_procs('gunicorn')
    health['services']['gunicorn'] = {
        'status': 'running' if gunicorn_count > 0 else 'stopped',
        'workers': gunicorn_count,
    }
    
    # 2. Nginx
    nginx_count = _count_procs('nginx')
    health['services']['nginx'] = {
        'status': 'running' if nginx_count > 0 else 'stopped',
        'workers': nginx_count,
    }
    
    # 3. Docker / OpenClaw
    docker_count = _count_procs('openclaw')
    health['services']['openclaw_gateway'] = {
        'status': 'running' if docker_count > 0 else 'stopped',
    }
    
    # 4. System resources
    health['system'] = {
        'cpu_percent': psutil.cpu_percent(interval=0.5),
        'memory': {
            'total_gb': round(psutil.virtual_memory().total / 1024**3, 1),
            'used_gb': round(psutil.virtual_memory().used / 1024**3, 1),
            'percent': psutil.virtual_memory().percent,
        },
        'disk': {
            'total_gb': round(psutil.disk_usage('/').total / 1024**3, 1),
            'used_gb': round(psutil.disk_usage('/').used / 1024**3, 1),
            'percent': psutil.disk_usage('/').percent,
        },
    }
    
    # 5. Overall status
    all_running = all(
        s.get('status') == 'running' 
        for s in health['services'].values()
    )
    health['overall'] = 'healthy' if all_running else 'degraded'
    
    return jsonify(health)


@app.route('/ops/logs', methods=['GET'])
def ops_logs():
    """获取最近的错误日志"""
    lines = int(request.args.get('lines', 50))
    source = request.args.get('source', 'all')  # all, gunicorn, nginx
    
    logs = {}
    
    if source in ('all', 'gunicorn'):
        try:
            log_path = os.environ.get('GUNICORN_LOG', os.path.join(BASE_DIR, 'gunicorn.log'))
            with open(log_path, 'r') as f:
                content = f.readlines()
                logs['gunicorn'] = content[-lines:]
        except:
            logs['gunicorn'] = ['File not accessible']
    
    if source in ('all', 'nginx'):
        try:
            result = subprocess.run(
                ['tail', f'-{lines}', '/var/log/nginx/error.log'],
                capture_output=True, text=True, timeout=5
            )
            logs['nginx_error'] = result.stdout.strip().split('\n') if result.stdout else []
        except:
            logs['nginx_error'] = ['File not accessible']
    
    return jsonify(logs)


@app.route('/ops/metrics', methods=['GET'])
def ops_metrics():
    """系统详细指标"""
    return jsonify({
        'cpu': {
            'count': psutil.cpu_count(),
            'percent_per_cpu': psutil.cpu_percent(interval=1, percpu=True),
            'load_avg': list(os.getloadavg()),
        },
        'memory': {
            'total_mb': round(psutil.virtual_memory().total / 1024**2),
            'available_mb': round(psutil.virtual_memory().available / 1024**2),
            'percent': psutil.virtual_memory().percent,
            'swap_percent': psutil.swap_memory().percent,
        },
        'disk': {
            'total_gb': round(psutil.disk_usage('/').total / 1024**3, 1),
            'free_gb': round(psutil.disk_usage('/').free / 1024**3, 1),
            'percent': psutil.disk_usage('/').percent,
        },
        'uptime_hours': round((time.time() - psutil.boot_time()) / 3600, 1),
        'connections': len(psutil.net_connections()),
    })

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


def _build_recognition_payload(result, preprocessing, use_preprocessing, raster_path, overlay_path, mask_path, elapsed_sec):
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
    path = _recognition_json_path(target_dir)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
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
    path = _recognition_json_path(target_dir)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    if payload.get('schema_version') != RECOGNITION_SCHEMA_VERSION:
        return None
    model = payload.get('model') or {}
    if model.get('backend', LEGACY_ONNX_BACKEND) not in {LEGACY_ONNX_BACKEND, VECTOR_PYTORCH_BACKEND}:
        return None
    if not isinstance(model.get('version'), str) or not model.get('version'):
        return None
    geometry = payload.get('geometry') or {}
    if not all(key in geometry for key in ('walls', 'windows', 'doors')):
        return None
    return payload


def _persist_vector_recognition(target_dir, raster_path, result, elapsed_sec):
    """Persist the opt-in vector backend in the existing recognition contract."""
    original_bgr = result['source']
    overlay = result['overlay']
    mask = result['mask']
    h, w = mask.shape
    overlay_path = os.path.join(target_dir, 'ai_overlay.jpg')
    mask_path = os.path.join(target_dir, 'ai_mask.png')
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
        'model_info': result['model'],
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


def _make_pdf_upload_token(report_number, stored_filename, page_count):
    return _pdf_upload_serializer().dumps({
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


def _validated_prepared_pdf(report_number, pdf_upload_token, pdf_page_number):
    """Return a validated stored PDF path, selected page, and page count."""
    prepared_pdf = _load_pdf_upload_token(pdf_upload_token)
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

    pdf_path = Path(app.config['UPLOAD_FOLDER']) / 'energy' / report_number / stored_filename
    if not pdf_path.is_file():
        raise ValueError('Prepared PDF file no longer exists')
    return pdf_path, page_number, page_count


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_exterior_report_dir(report_number):
    """Resolve one literal report directory without accepting aliases or links."""
    if not isinstance(report_number, str) or not report_number:
        raise ValueError('report_number is required')
    if secure_filename(report_number) != report_number:
        raise ValueError('report_number is invalid')
    energy_root = (Path(app.config['UPLOAD_FOLDER']) / 'energy').resolve()
    report_path = energy_root / report_number
    if not report_path.is_dir():
        raise FileNotFoundError('Report does not exist')
    resolved_report = report_path.resolve(strict=True)
    if report_path.is_symlink() or resolved_report.parent != energy_root:
        raise ValueError('report_number is invalid')
    artifact_path = resolved_report / 'vector_pdf_fusion'
    if not artifact_path.is_dir():
        raise FileNotFoundError('Vector PDF fusion artifacts do not exist')
    resolved_artifacts = artifact_path.resolve(strict=True)
    if artifact_path.is_symlink() or resolved_artifacts.parent != resolved_report:
        raise ValueError('Vector PDF fusion artifact path is invalid')
    return resolved_report, resolved_artifacts


def _read_exterior_artifact(artifact_dir, filename):
    path = artifact_dir / filename
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f'{filename} does not exist')
    resolved = path.resolve(strict=True)
    if resolved.parent != artifact_dir:
        raise ValueError(f'{filename} path is invalid')
    raw = resolved.read_bytes()
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
    report_dir = Path(report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    lock_key = os.fspath(report_dir)
    with _exterior_thread_locks_guard:
        thread_lock = _exterior_thread_locks.setdefault(lock_key, threading.RLock())

    with thread_lock:
        lock_path = report_dir / '.exterior_generation.lock'
        with open(lock_path, 'a+b') as lock_file:
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
    return Path(report_dir) / 'exterior_generation.json'


def _read_exterior_generation(report_dir):
    path = _exterior_generation_path(report_dir)
    if not path.is_file() or path.is_symlink():
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
        report_number = secure_filename(supplied or '') if isinstance(supplied, str) else ''
        if not report_number or report_number != supplied:
            return jsonify({'error': 'report_number is invalid'}), 400
        report_dir = Path(app.config['UPLOAD_FOLDER']) / 'energy' / report_number
        with _exterior_report_lock(report_dir):
            return route(*args, **kwargs)
    return wrapped


@app.route('/energy/pdf_prepare', methods=['POST'])
@login_required
def prepare_energy_pdf():
    """Persist one PDF upload and return its page count for page selection."""
    report_number = secure_filename(request.form.get('report_number', 'default')) or 'default'
    uploaded = request.files.get('raster_file')
    if not uploaded or not uploaded.filename:
        return jsonify({'error': 'No PDF file provided'}), 400

    original_filename = secure_filename(uploaded.filename)
    extension = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''
    if extension != 'pdf':
        return jsonify({'error': 'Only PDF files can be prepared'}), 400

    target_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'energy', report_number)
    os.makedirs(target_dir, exist_ok=True)
    stored_filename = f'building_plan_prepared_{uuid.uuid4().hex}.pdf'
    stored_path = os.path.join(target_dir, stored_filename)
    uploaded.save(stored_path)

    try:
        import pdfplumber

        with pdfplumber.open(stored_path) as pdf:
            page_count = len(pdf.pages)
        if page_count < 1:
            raise ValueError('PDF contains no pages')
    except Exception as exc:
        if os.path.isfile(stored_path):
            os.remove(stored_path)
        return jsonify({'error': f'Cannot read PDF: {exc}'}), 400

    return jsonify({
        'success': True,
        'upload_token': _make_pdf_upload_token(report_number, stored_filename, page_count),
        'page_count': page_count,
        'filename': original_filename,
    })


@app.route('/energy/pdf_page_preview', methods=['POST'])
@login_required
def preview_energy_pdf_page():
    """Render a selected prepared-PDF page for client-side region selection."""
    report_number = secure_filename(request.form.get('report_number', 'default')) or 'default'
    try:
        pdf_path, page_number, page_count = _validated_prepared_pdf(
            report_number,
            request.form.get('pdf_upload_token', '').strip(),
            request.form.get('pdf_page_number', '1'),
        )
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    try:
        preview_bgr = render_pdf_page_preview(
            pdf_path,
            page_number,
            page_count,
            _resolve_poppler_path(),
        )
        encoded, jpeg = cv2.imencode(
            '.jpg',
            preview_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
        if not encoded:
            raise ValueError('Cannot encode PDF preview')
    except Exception as exc:
        return jsonify({'error': f'PDF preview failed: {exc}'}), 500

    height, width = preview_bgr.shape[:2]
    return jsonify({
        'success': True,
        'pdf_page_number': page_number,
        'pdf_page_count': page_count,
        'image_size': [width, height],
        'image': base64.b64encode(jpeg).decode('utf-8'),
    })


@app.route('/energy/crop_region.js', methods=['GET'])
@login_required
def energy_crop_region_helper():
    """Serve the energy crop-selection helper."""
    return send_from_directory(
        os.path.join(BASE_DIR, 'static', 'energy'),
        'crop_region.js',
        mimetype='application/javascript',
    )


@app.route('/energy/pdf_recognition_state.js', methods=['GET'])
@login_required
def energy_pdf_recognition_state_helper():
    """Serve the PDF recognition request-state helper."""
    return send_from_directory(
        os.path.join(BASE_DIR, 'static', 'energy'),
        'pdf_recognition_state.js',
        mimetype='application/javascript',
    )


@app.route('/energy/vector_pdf_fusion', methods=['POST'])
@login_required
@_serialized_exterior_report
def vector_pdf_fusion():
    """Publish isolated native-PDF/model diagnostics without energy geometry."""
    if not HAS_VECTOR_PDF_FUSION or _vector_pdf_fusion_config is None:
        return jsonify({'error': 'Vector PDF fusion module is not configured'}), 501
    report_number = secure_filename(request.form.get('report_number', 'default')) or 'default'
    try:
        pdf_path, page_number, page_count = _validated_prepared_pdf(
            report_number,
            request.form.get('pdf_upload_token', '').strip(),
            request.form.get('pdf_page_number', '1'),
        )
        region_request = parse_crop_region_request(
            request.form.get('recognition_mode', 'full_page'),
            request.form.get('crop_bbox_px'),
            request.form.get('crop_preview_size'),
        )
        crop_bbox_page_px = None
        if region_request['mode'] == 'crop_region':
            import pdfplumber

            with pdfplumber.open(str(pdf_path)) as document:
                page = document.pages[page_number - 1]
                render_size = [
                    round(float(page.width) * 100 / 72),
                    round(float(page.height) * 100 / 72),
                ]
            crop_bbox_page_px = map_crop_bbox_to_page(
                region_request['crop_bbox_px'],
                region_request['preview_size'],
                render_size,
            )
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    report_dir = Path(app.config['UPLOAD_FOLDER']) / 'energy' / report_number
    target_dir = report_dir / 'vector_pdf_fusion'
    generation_marker = _new_exterior_generation(report_number)
    _write_exterior_generation(report_dir, generation_marker)
    try:
        result = analyze_vector_pdf_page(
            pdf_path,
            page_number,
            target_dir,
            _vector_pdf_fusion_config,
            crop_bbox_page_px=crop_bbox_page_px,
        )
        original = cv2.imread(str(target_dir / 'pdf_vector_model_input.png'), cv2.IMREAD_COLOR)
        overlay = cv2.imread(str(target_dir / 'pdf_vector_fusion_overlay.png'), cv2.IMREAD_COLOR)
        exterior_overlay = cv2.imread(str(target_dir / 'pdf_exterior_overlay.png'), cv2.IMREAD_COLOR)
        component_overlay = cv2.imread(str(target_dir / 'pdf_component_overlay.png'), cv2.IMREAD_COLOR)
        images = {}
        if original is not None:
            encoded, buffer = cv2.imencode('.png', original)
            if encoded:
                images['original'] = base64.b64encode(buffer).decode('utf-8')
        if overlay is not None:
            encoded, buffer = cv2.imencode('.png', overlay)
            if encoded:
                images['overlay'] = base64.b64encode(buffer).decode('utf-8')
        if exterior_overlay is not None:
            encoded, buffer = cv2.imencode('.png', exterior_overlay)
            if encoded:
                images['exterior_overlay'] = base64.b64encode(buffer).decode('utf-8')
        if component_overlay is not None:
            encoded, buffer = cv2.imencode('.png', component_overlay)
            if encoded:
                images['component_overlay'] = base64.b64encode(buffer).decode('utf-8')
        topology_sha256 = _sha256_file(target_dir / 'pdf_exterior_topology.json')
    except ValueError as exc:
        _update_exterior_generation_if_current(
            report_dir, generation_marker['generation'], status='failed',
        )
        return jsonify({'error': str(exc)}), 400
    except Exception as exc:
        _update_exterior_generation_if_current(
            report_dir, generation_marker['generation'], status='failed',
        )
        return jsonify({'error': f'Vector PDF fusion failed: {exc}'}), 500

    exterior_topology = result.get('exterior_topology') or {}
    opening_candidates = result.get('opening_candidates') or {}
    accepted_openings = opening_candidates.get('accepted_openings') or []
    pending_openings = opening_candidates.get('pending_openings') or []
    exterior_summary = dict(result.get('exterior_summary') or {})
    exterior_summary.update({
        'area_px2': exterior_topology.get('area_px2'),
        'perimeter_px': exterior_topology.get('perimeter_px'),
        'door_count': sum(
            opening.get('kind') == 'door' for opening in accepted_openings
        ),
        'door_total_width_px': math.fsum(
            float(opening.get('width_px') or 0.0)
            for opening in accepted_openings
            if opening.get('kind') == 'door'
        ),
        'window_count': sum(
            opening.get('kind') == 'window' for opening in accepted_openings
        ),
        'window_total_width_px': math.fsum(
            float(opening.get('width_px') or 0.0)
            for opening in accepted_openings
            if opening.get('kind') == 'window'
        ),
        'pending_opening_count': len(pending_openings),
        'small_repair_count': sum(
            bridge.get('bridge_type') == 'small_gap_repair'
            for bridge in (exterior_topology.get('bridges') or [])
        ),
        'unresolved_gap_count': len(exterior_topology.get('unresolved') or []),
    })
    exterior_summary.setdefault('recovered_wall_count', 0)
    exterior_summary.setdefault('confirmed_door_arc_count', 0)
    exterior_summary.setdefault('pending_door_arc_count', 0)
    exterior_summary.setdefault('closure_method', exterior_topology.get('closure_method'))
    exterior_summary.setdefault(
        'inferred_edge_count', len(exterior_topology.get('inferred_edges') or []),
    )
    exterior_summary.setdefault(
        'inferred_length_px', float(exterior_topology.get('inferred_length_px') or 0.0),
    )
    exterior_summary.setdefault(
        'inferred_perimeter_ratio',
        float(exterior_topology.get('inferred_perimeter_ratio') or 0.0),
    )

    _update_exterior_generation_if_current(
        report_dir,
        generation_marker['generation'],
        status='ready',
        topology_sha256=topology_sha256,
    )
    return jsonify({
        'success': True,
        'report_number': report_number,
        'fusion_debug': True,
        'status': result['status'],
        'exterior_status': exterior_topology.get('status'),
        'load_geometry_ready': False,
        'pdf_page_number': page_number,
        'pdf_page_count': page_count,
        'recognition_mode': region_request['mode'],
        'crop_bbox_page_px': crop_bbox_page_px,
        'image_size': (result.get('page') or {}).get('analysis_size_px'),
        'summary': result.get('summary') or {},
        'exterior_summary': exterior_summary,
        'topology_sha256': topology_sha256,
        'reason_codes': result.get('reason_codes') or [],
        'artifacts': {
            'native_candidates': 'vector_pdf_fusion/pdf_native_candidates.json',
            'fusion': 'vector_pdf_fusion/pdf_vector_fusion.json',
            'overlay': 'vector_pdf_fusion/pdf_vector_fusion_overlay.png',
            'opening_candidates': 'vector_pdf_fusion/pdf_opening_candidates.json',
            'exterior_topology': 'vector_pdf_fusion/pdf_exterior_topology.json',
            'exterior_overlay': 'vector_pdf_fusion/pdf_exterior_overlay.png',
            'component_overlay': 'vector_pdf_fusion/pdf_component_overlay.png',
        },
        'images': images,
    })


@app.route('/energy/vector_pdf_exterior_confirm', methods=['POST'])
@login_required
@_serialized_exterior_report
def vector_pdf_exterior_confirm():
    """Confirm only the current, hashed server-side exterior artifacts."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({'error': 'JSON request body is required'}), 400
    if payload.get('confirmed') is not True:
        return jsonify({'error': 'confirmed must be boolean true'}), 400
    try:
        calibration_method = payload.get(
            'calibration_method', 'manual_exterior_confirmation',
        )
        if calibration_method not in {'manual_two_point', 'manual_exterior_confirmation'}:
            raise ValueError('calibration_method is invalid')
        scale = _positive_finite_number(payload.get('scale_m_per_px'), 'scale_m_per_px')
        request_page = payload.get('page_number')
        if isinstance(request_page, bool) or not isinstance(request_page, int) or request_page < 1:
            raise ValueError('page_number must be a positive integer')
        if 'crop_bbox_page_px' not in payload:
            raise ValueError('crop_bbox_page_px is required (null for a full page)')
        request_crop = _normalized_crop_bbox(
            payload.get('crop_bbox_page_px'), 'crop_bbox_page_px',
        )
        supplied_hash = payload.get('topology_sha256')
        if not isinstance(supplied_hash, str) or len(supplied_hash) != 64:
            raise ValueError('topology_sha256 must be a SHA-256 hex digest')
        try:
            int(supplied_hash, 16)
        except ValueError:
            raise ValueError('topology_sha256 must be a SHA-256 hex digest') from None
        report_number = payload.get('report_number')
        report_dir, artifact_dir = _require_exterior_report_dir(report_number)
        topology_raw, topology = _read_exterior_artifact(
            artifact_dir, 'pdf_exterior_topology.json',
        )
        opening_raw, opening_artifact = _read_exterior_artifact(
            artifact_dir, 'pdf_opening_candidates.json',
        )
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 404
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    current_hash = hashlib.sha256(topology_raw).hexdigest()
    if not hmac.compare_digest(current_hash, supplied_hash.lower()):
        return jsonify({'error': 'Exterior topology has changed; review it again'}), 409

    try:
        generation_marker = _require_current_exterior_generation(
            report_dir, report_number, topology_sha256=current_hash,
        )
    except ExteriorGenerationConflict as exc:
        return jsonify({'error': str(exc)}), 409

    opening_hash = topology.get('opening_artifact_sha256')
    if (
        not isinstance(opening_hash, str)
        or len(opening_hash) != 64
        or any(character not in '0123456789abcdef' for character in opening_hash)
        or not hmac.compare_digest(hashlib.sha256(opening_raw).hexdigest(), opening_hash)
    ):
        return jsonify({'error': 'Opening artifact has changed; review it again'}), 409

    try:
        provenance, artifact_page, artifact_crop, image_size = _artifact_provenance(
            topology, opening_artifact,
        )
        if request_page != artifact_page or request_crop != artifact_crop:
            raise RuntimeError('Confirmed page or crop does not match current artifacts')
        (
            polygon_points, area_px2, perimeter_px,
            inferred_edges, inferred_length_px, inferred_perimeter_ratio,
        ) = _validate_confirmable_topology(topology, opening_artifact, scale, image_size)
        openings = _validate_opening_bindings(topology, opening_artifact)

        from vector_pdf_energy_geometry import apply_scale_to_exterior
        normalized_topology = copy.deepcopy(topology)
        normalized_topology['area_px2'] = area_px2
        normalized_topology['perimeter_px'] = perimeter_px
        normalized_topology['inferred_edges'] = inferred_edges
        normalized_topology['inferred_length_px'] = inferred_length_px
        normalized_topology['inferred_perimeter_ratio'] = inferred_perimeter_ratio
        scaled_topology, scaled_openings = apply_scale_to_exterior(
            normalized_topology, openings, scale,
        )
        scaled_topology['confirmed'] = True
        scaled_topology['load_geometry_ready'] = True
        scaled_topology['status'] = 'confirmed'

        walls = _validated_real_exterior_wall_geometry(
            polygon_points, topology, inferred_edges,
        )
        doors = []
        windows = []
        for opening in scaled_openings:
            _, _, _, start, end = _axis_segment(opening, 'opening')
            geometry_opening = copy.deepcopy(opening)
            geometry_opening.update({
                'id': str(opening['opening_id']),
                'pts': [start, end],
                'length_px': float(opening['width_px']),
            })
            if opening.get('kind') == 'door':
                doors.append(geometry_opening)
            elif opening.get('kind') == 'window':
                windows.append(geometry_opening)
            else:
                raise RuntimeError('Opening kind must be door or window')
        door_width = math.fsum(float(opening['width_m']) for opening in doors)
        window_width = math.fsum(float(opening['width_m']) for opening in windows)
        if not math.isfinite(door_width) or not math.isfinite(window_width):
            raise RuntimeError('Opening width totals must be finite')
        geometry = {'walls': walls, 'windows': windows, 'doors': doors}
        recognition = {
            'schema_version': RECOGNITION_SCHEMA_VERSION,
            'report_number': report_number,
            'exterior_generation': {
                'generation': generation_marker['generation'],
                'topology_sha256': current_hash,
            },
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'model': {
                'backend': VECTOR_PYTORCH_BACKEND,
                'name': 'vector_pdf_fusion',
                'version': current_hash,
            },
            'preprocessing': {'requested': 'vector_pdf_fusion', 'use_preprocessing': False},
            'source_image': {'path': 'vector_pdf_fusion/pdf_vector_model_input.png'},
            'source_page': copy.deepcopy(provenance),
            'image_size': image_size,
            'stats': {},
            'geometry_summary': _geometry_summary(geometry),
            'geometry': geometry,
            'contours': copy.deepcopy(geometry),
            'room_topology': {
                'status': 'exterior_only',
                'room_count': 0,
                'closure_applied': False,
                'rooms': [],
                'total_area_px2': 0.0,
                'total_area_m2': None,
                'scale_m_per_px': None,
                'load_geometry_ready': False,
            },
            'exterior_topology': scaled_topology,
            'openings': scaled_openings,
            'opening_widths': {
                'door_total_width_m': door_width,
                'window_total_width_m': window_width,
            },
            'scale_calibration': {
                'status': 'confirmed',
                'method': calibration_method,
                'scale_m_per_px': scale,
                'confidence': 1.0,
                'evidence': [{'topology_sha256': current_hash}],
            },
            'pdf_page_number': artifact_page,
            'pdf_page_count': None,
            'recognition_mode': 'crop_region' if artifact_crop is not None else 'full_page',
            'crop_bbox_page_px': artifact_crop,
            'crop_bbox_preview_px': None,
            'crop_preview_size': None,
            'pixel_lengths': {
                'wall_px': _polyline_total_length(walls),
                'window_px': _polyline_total_length(windows),
            },
            'artifacts': {
                'original': 'vector_pdf_fusion/pdf_vector_model_input.png',
                'overlay': 'vector_pdf_fusion/pdf_exterior_overlay.png',
                'exterior_topology': 'vector_pdf_fusion/pdf_exterior_topology.json',
                'opening_candidates': 'vector_pdf_fusion/pdf_opening_candidates.json',
                'topology_sha256': current_hash,
            },
        }
        _require_current_exterior_generation(
            report_dir,
            report_number,
            topology_sha256=current_hash,
            expected_generation=generation_marker['generation'],
        )
        _atomic_write_json(report_dir / 'recognition.json', recognition)
        _require_current_exterior_generation(
            report_dir,
            report_number,
            recognition=recognition,
            topology_sha256=current_hash,
            expected_generation=generation_marker['generation'],
        )
    except ExteriorGenerationConflict as exc:
        return jsonify({'error': str(exc)}), 409
    except (RuntimeError, ValueError) as exc:
        return jsonify({'error': str(exc)}), 409
    except Exception as exc:
        return jsonify({'error': f'Cannot persist exterior recognition: {exc}'}), 500

    return jsonify({
        'success': True,
        'report_number': report_number,
        'status': 'confirmed',
        'load_geometry_ready': True,
        'topology_sha256': current_hash,
        'exterior_topology': scaled_topology,
        'openings': scaled_openings,
        'opening_widths': recognition['opening_widths'],
        'geometry_summary': recognition['geometry_summary'],
        'recognition': recognition,
    })


@app.route('/energy/ai_recognize', methods=['POST'])
@login_required
def ai_recognize():
    """
    AI 识别建筑平面图中的墙体、窗户、门
    支持 raster_file (PNG/JPG) 上传
    与 DXF 上传同步：用户可选择 DXF 矢量 或 图片AI识别
    """
    try:
        t0 = _time.time()
        model_backend = request.form.get('model_backend', LEGACY_ONNX_BACKEND)
        if model_backend not in {LEGACY_ONNX_BACKEND, VECTOR_PYTORCH_BACKEND}:
            return jsonify({'error': 'Unsupported model backend'}), 400
        if model_backend == LEGACY_ONNX_BACKEND and not HAS_FLOORPLAN_AI:
            return jsonify({'error': 'Legacy ONNX recognition module is not available'}), 501
        if model_backend == VECTOR_PYTORCH_BACKEND and not HAS_VECTOR_FLOORPLAN_AI:
            return jsonify({'error': 'Vector recognition module is not configured'}), 501
        report_number = request.form.get('report_number', 'default')
        report_number = secure_filename(report_number) or 'default'
        target_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'energy', report_number)
        os.makedirs(target_dir, exist_ok=True)

        preprocessing = request.form.get('preprocessing', 'auto')
        use_preprocessing = preprocessing != 'none'

        try:
            region_request = parse_crop_region_request(
                request.form.get('recognition_mode', 'full_page'),
                request.form.get('crop_bbox_px'),
                request.form.get('crop_preview_size'),
            )
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400

        pdf_upload_token = request.form.get('pdf_upload_token', '').strip()
        if (
            region_request['mode'] == 'crop_region'
            and not pdf_upload_token
            and model_backend != VECTOR_PYTORCH_BACKEND
        ):
            return jsonify({
                'error': 'crop_region requires a prepared PDF upload or the vector model',
            }), 400
        pdf_page_number = None
        pdf_page_count = None

        if pdf_upload_token:
            try:
                prepared_pdf_path, pdf_page_number, pdf_page_count = _validated_prepared_pdf(
                    report_number,
                    pdf_upload_token,
                    request.form.get('pdf_page_number', '1'),
                )
            except ValueError as exc:
                return jsonify({'error': str(exc)}), 400
            raster_path = os.fspath(prepared_pdf_path)
            ext = 'pdf'
        else:
            if 'raster_file' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400

            f = request.files['raster_file']
            if not f or not f.filename:
                return jsonify({'error': 'Empty file'}), 400

            ext = f.filename.rsplit('.', 1)[1].lower() if '.' in f.filename else 'png'
            raster_path = os.path.join(target_dir, f'building_plan_ai.{ext}')
            f.save(raster_path)
            if ext == 'pdf':
                pdf_page_number = 1

        if model_backend == VECTOR_PYTORCH_BACKEND:
            if ext == 'pdf':
                return jsonify({'error': 'Vector recognition currently accepts PNG or JPG; render the target PDF page first.'}), 400
            crop_bbox_image_px = None
            if region_request['mode'] == 'crop_region':
                source_bgr = cv2.imread(raster_path)
                if source_bgr is None:
                    return jsonify({'error': 'Cannot decode image'}), 400
                image_height, image_width = source_bgr.shape[:2]
                crop_bbox_image_px = map_crop_bbox_to_page(
                    region_request['crop_bbox_px'],
                    region_request['preview_size'],
                    [image_width, image_height],
                )
                left, top, right, bottom = crop_bbox_image_px
                raster_path = os.path.join(target_dir, 'building_plan_ai_crop.png')
                _write_image(Path(raster_path), source_bgr[top:bottom, left:right].copy())
            result = _vector_platform_adapter.predict(Path(raster_path), Path(target_dir))
            result.update({
                'recognition_mode': region_request['mode'],
                'crop_bbox_page_px': crop_bbox_image_px,
                'crop_bbox_preview_px': region_request['crop_bbox_px'],
                'crop_preview_size': region_request['preview_size'],
            })
            return _persist_vector_recognition(
                target_dir, raster_path, result, round(_time.time() - t0, 3),
            )

        scale_calibration = {
            'status': 'manual_required',
            'method': 'raster_manual',
            'scale_m_per_px': None,
            'confidence': 0.0,
            'horizontal': None,
            'vertical': None,
            'axis_difference_percent': None,
            'evidence': [],
        }
        dimension_page_data = None
        dimension_candidates = []
        dimension_cleanup_candidates = []
        vector_cleanup = {
            'enabled': False,
            'has_vector_geometry': False,
            'has_vector_text': False,
            'ocr': {'status': 'not_needed', 'candidate_count': 0, 'accepted_count': 0},
            'nonstructural_mask': {'enabled': False},
            'structural_mask': {'enabled': False},
            'building_roi': {'enabled': False, 'bbox_px': None},
        }

        # PDF → PNG 转换，并优先从矢量文字和线段中自动标定比例尺。
        prepared_page = None
        if ext == 'pdf':
            try:
                prepared_page = prepare_pdf_page(
                    raster_path,
                    page_number=pdf_page_number,
                    poppler_path=_resolve_poppler_path(),
                    segmenter=_floorplan_segmenter,
                    output_dir=target_dir,
                )
                pdf_page_count = prepared_page.page_count
                scale_calibration = prepared_page.scale_calibration
                vector_cleanup = dict(prepared_page.vector_cleanup)
                png_path = os.path.join(target_dir, 'building_plan_ai.png')
                raster_path = png_path
            except Exception as e:
                return jsonify({'error': f'PDF conversion failed: {e}'}), 500

        # AI 推理
        img_bgr = (
            prepared_page.cleaned_bgr.copy()
            if prepared_page is not None
            else cv2.imread(raster_path)
        )
        if img_bgr is None:
            return jsonify({'error': 'Cannot decode image'}), 400

        original_bgr = (
            prepared_page.render_bgr.copy()
            if prepared_page is not None
            else img_bgr.copy()
        )
        combined_cleanup_mask = (
            prepared_page.cleanup_mask.copy()
            if prepared_page is not None
            else np.zeros(img_bgr.shape[:2], dtype=np.uint8)
        )
        structural_support_mask = (
            prepared_page.structural_support_mask
            if prepared_page is not None
            else None
        )
        inference_roi = (
            prepared_page.inference_roi
            if prepared_page is not None
            else None
        )
        crop_bbox_page_px = None
        if prepared_page is not None and region_request['mode'] == 'crop_region':
            page_height, page_width = prepared_page.render_bgr.shape[:2]
            crop_bbox_page_px = map_crop_bbox_to_page(
                region_request['crop_bbox_px'],
                region_request['preview_size'],
                [page_width, page_height],
            )
            cropped_inputs = crop_page_inputs(
                prepared_page.render_bgr,
                prepared_page.cleaned_bgr,
                prepared_page.cleanup_mask,
                prepared_page.structural_support_mask,
                prepared_page.inference_roi,
                crop_bbox_page_px,
            )
            original_bgr = cropped_inputs['render_bgr']
            img_bgr = cropped_inputs['cleaned_bgr']
            combined_cleanup_mask = cropped_inputs['cleanup_mask']
            structural_support_mask = cropped_inputs['structural_support_mask']
            inference_roi = cropped_inputs['inference_roi']
            building_roi = dict(vector_cleanup.get('building_roi') or {})
            building_roi.update({
                'enabled': True,
                'bbox_px': inference_roi,
            })
            vector_cleanup['building_roi'] = building_roi

        if prepared_page is not None:
            _write_image(Path(raster_path), original_bgr)

        has_vector_geometry = bool(
            prepared_page is not None
            and prepared_page.vector_cleanup.get('has_vector_geometry')
        )

        if (
            prepared_page is not None
            and (
                np.any(combined_cleanup_mask)
                or vector_cleanup.get('nonstructural_mask', {}).get('enabled')
            )
        ):
            _write_image(
                Path(target_dir) / 'pdf_nonstructural_mask.png',
                combined_cleanup_mask,
            )

        if prepared_page is not None and has_vector_geometry:
            if (
                structural_support_mask is not None
                and vector_cleanup.get('structural_mask', {}).get('enabled')
            ):
                _write_image(
                    Path(target_dir) / 'pdf_structural_mask.png',
                    structural_support_mask,
                )
            building_roi = vector_cleanup.get('building_roi') or {
                'enabled': False,
                'bbox_px': None,
            }
            roi_path = os.path.join(target_dir, 'pdf_building_roi.json')
            with open(roi_path, 'w', encoding='utf-8') as roi_file:
                json.dump(building_roi, roi_file, ensure_ascii=False, indent=2)

        if prepared_page is not None and has_vector_geometry:
            _write_image(Path(target_dir) / 'pdf_model_input.png', img_bgr)

        predict_kwargs = {}
        if has_vector_geometry:
            topology_min_room_area_px = min(
                5000.0,
                max(500.0, img_bgr.shape[0] * img_bgr.shape[1] * 0.0002),
            )
            topology_repair_context = None
            if (
                inference_roi is not None
                and structural_support_mask is not None
                and vector_cleanup.get('structural_mask', {}).get('enabled')
            ):
                topology_repair_context = {
                    'building_roi': inference_roi,
                    'structural_support_mask': structural_support_mask,
                    'max_exterior_gap_px': max(
                        12,
                        min(64, round(min(img_bgr.shape[:2]) * 0.015)),
                    ),
                    'max_internal_component_area_px': min(
                        5000,
                        max(200, round(img_bgr.shape[0] * img_bgr.shape[1] * 0.00015)),
                    ),
                }
            predict_kwargs = {
                'inference_roi': inference_roi,
                'preserve_full_context': True,
                'topology_repair_context': topology_repair_context,
                'topology_max_gap_px': 12,
                'topology_min_room_area_px': topology_min_room_area_px,
            }
        result = _floorplan_segmenter.predict(
            img_bgr,
            use_preprocessing=use_preprocessing,
            **predict_kwargs,
        )
        result['scale_calibration'] = scale_calibration
        result['pdf_page_number'] = pdf_page_number
        result['pdf_page_count'] = pdf_page_count
        result['recognition_mode'] = region_request['mode']
        result['crop_bbox_page_px'] = crop_bbox_page_px
        result['crop_bbox_preview_px'] = region_request['crop_bbox_px']
        result['crop_preview_size'] = region_request['preview_size']
        result['vector_cleanup'] = vector_cleanup

        mask = result['mask']
        overlay = result['overlay']
        stats = result['stats']
        geometry = result['geometry']
        room_topology = result.get('room_topology') or {
            'status': 'no_closed_rooms',
            'room_count': 0,
            'rooms': [],
            'total_area_px2': 0.0,
            'total_area_m2': None,
            'load_geometry_ready': False,
        }
        if scale_calibration.get('status') == 'confirmed':
            room_topology = apply_scale_to_room_topology(
                room_topology,
                scale_calibration.get('scale_m_per_px'),
            )
        room_topology = _enforce_topology_repair_readiness(
            room_topology,
            result.get('topology_repair'),
        )
        result['room_topology'] = room_topology
        h, w = mask.shape

        # 保存结果
        overlay_path = os.path.join(target_dir, 'ai_overlay.jpg')
        mask_path = os.path.join(target_dir, 'ai_mask.png')
        _write_image(
            Path(overlay_path),
            overlay,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )

        raw_model_mask = result.get('raw_model_mask')
        if raw_model_mask is not None:
            raw_mask_color = np.zeros((h, w, 3), dtype=np.uint8)
            raw_mask_color[raw_model_mask == 1] = (60, 76, 231)
            raw_mask_color[raw_model_mask == 2] = (219, 152, 52)
            raw_mask_color[raw_model_mask == 3] = (113, 204, 46)
            _write_image(Path(target_dir) / 'ai_raw_model_mask.png', raw_mask_color)

        # mask 转彩色保存
        mask_color = np.zeros((h, w, 3), dtype=np.uint8)
        mask_color[mask == 1] = (60, 76, 231)   # wall - red
        mask_color[mask == 2] = (219, 152, 52)   # window - blue
        mask_color[mask == 3] = (113, 204, 46)   # door - green
        _write_image(Path(mask_path), mask_color)

        # 编码为base64用于前端展示
        _, overlay_buf = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        overlay_b64 = base64.b64encode(overlay_buf).decode('utf-8')

        _, mask_buf = cv2.imencode('.png', mask_color)
        mask_b64 = base64.b64encode(mask_buf).decode('utf-8')

        _, orig_buf = cv2.imencode('.jpg', original_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        orig_b64 = base64.b64encode(orig_buf).decode('utf-8')

        elapsed = round(_time.time() - t0, 3)

        # 构建几何数据，可直接用于能耗模拟
        geo_summary = {
            'walls': len(geometry['walls']),
            'windows': len(geometry['windows']),
            'doors': len(geometry['doors']),
        }

        # 计算墙体/窗户轮廓的总长度
        wall_total_length = 0
        for wall in geometry['walls']:
            pts = wall['pts']
            for i in range(len(pts) - 1):
                dx = pts[i+1][0] - pts[i][0]
                dy = pts[i+1][1] - pts[i][1]
                wall_total_length += (dx*dx + dy*dy) ** 0.5

        win_total_length = 0
        for win in geometry['windows']:
            pts = win['pts']
            for i in range(len(pts) - 1):
                dx = pts[i+1][0] - pts[i][0]
                dy = pts[i+1][1] - pts[i][1]
                win_total_length += (dx*dx + dy*dy) ** 0.5

        recognition_payload = _build_recognition_payload(
            result,
            preprocessing,
            use_preprocessing,
            raster_path,
            overlay_path,
            mask_path,
            elapsed,
        )
        recognition_path = _save_recognition_payload(target_dir, recognition_payload)

        return jsonify({
            'success': True,
            'status': 'success',
            'source': 'AI',
            'model': FLOORPLAN_MODEL_NAME,
            'model_info': {
                'name': FLOORPLAN_MODEL_NAME,
                'version': FLOORPLAN_MODEL_VERSION,
                'mIoU': FLOORPLAN_MODEL_MIOU,
            },
            'mIoU': FLOORPLAN_MODEL_MIOU,
            'elapsed_sec': elapsed,
            'image_size': [w, h],
            'stats': stats,
            'geometry_summary': geo_summary,
            'geometry': geometry,
            'room_topology': room_topology,
            'scale_calibration': scale_calibration,
            'pdf_page_number': pdf_page_number,
            'pdf_page_count': pdf_page_count,
            'recognition_mode': result['recognition_mode'],
            'crop_bbox_page_px': result['crop_bbox_page_px'],
            'crop_bbox_preview_px': result['crop_bbox_preview_px'],
            'crop_preview_size': result['crop_preview_size'],
            'vector_cleanup': vector_cleanup,
            'topology_repair': result.get('topology_repair') or {},
            'pixel_lengths': {
                'wall_px': round(wall_total_length, 1),
                'window_px': round(win_total_length, 1),
            },
            'recognition': {
                'schema_version': recognition_payload['schema_version'],
                'path': os.path.basename(recognition_path),
                'model_version': FLOORPLAN_MODEL_VERSION,
                'preprocessing': recognition_payload['preprocessing'],
            },
            'images': {
                'original': orig_b64,
                'overlay': overlay_b64,
                'mask': mask_b64,
            },
        })

    except Exception as e:
        logger.error(f"AI recognize error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/energy/ai_status')
def ai_status():
    """检查 AI 识别模块状态"""
    return jsonify({
        'available': HAS_FLOORPLAN_AI or HAS_VECTOR_FLOORPLAN_AI,
        'model': (
            FLOORPLAN_MODEL_NAME if HAS_FLOORPLAN_AI
            else ('vector-resnet34-unet' if HAS_VECTOR_FLOORPLAN_AI else None)
        ),
        'model_version': (
            FLOORPLAN_MODEL_VERSION if HAS_FLOORPLAN_AI
            else (_vector_platform_config.checkpoint_path.name if HAS_VECTOR_FLOORPLAN_AI else None)
        ),
        'mIoU': FLOORPLAN_MODEL_MIOU if HAS_FLOORPLAN_AI else None,
        'classes': ['background', 'wall', 'window', 'door'] if HAS_FLOORPLAN_AI else [],
        'backends': {
            LEGACY_ONNX_BACKEND: {
                'available': HAS_FLOORPLAN_AI,
                'model': FLOORPLAN_MODEL_NAME if HAS_FLOORPLAN_AI else None,
                'model_version': FLOORPLAN_MODEL_VERSION if HAS_FLOORPLAN_AI else None,
            },
            VECTOR_PYTORCH_BACKEND: {
                'available': HAS_VECTOR_FLOORPLAN_AI,
                'model': 'vector-resnet34-unet' if HAS_VECTOR_FLOORPLAN_AI else None,
                'model_version': (
                    _vector_platform_config.checkpoint_path.name
                    if HAS_VECTOR_FLOORPLAN_AI else None
                ),
            },
        },
    })


@app.route('/energy/scale_calibration', methods=['POST'])
@login_required
def save_scale_calibration():
    """Persist a manual two-point scale when automatic PDF evidence is unavailable."""
    try:
        data = request.get_json() or {}
        report_number = secure_filename(str(data.get('report_number') or ''))
        if not report_number:
            raise ValueError('report_number is required')
        target_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'energy', report_number)
        recognition = _load_recognition_payload(target_dir)
        if recognition is None:
            return jsonify({'error': 'No current recognition.json found. Run AI recognition first.'}), 404

        image_size = recognition.get('image_size') or [0, 0]
        width, height = float(image_size[0] or 0), float(image_size[1] or 0)

        def parse_point(key):
            point = data.get(key)
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError(f'{key} must contain x and y')
            x, y = float(point[0]), float(point[1])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f'{key} must contain finite coordinates')
            if x < 0 or y < 0 or x >= width or y >= height:
                raise ValueError(f'{key} is outside the recognized image')
            return [x, y]

        point_a = parse_point('point_a')
        point_b = parse_point('point_b')
        pixel_distance = math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])
        if pixel_distance <= 0:
            raise ValueError('calibration points must be different')

        actual_length = float(data.get('actual_length'))
        if not math.isfinite(actual_length) or actual_length <= 0:
            raise ValueError('actual_length must be positive and finite')
        unit = str(data.get('unit') or '').lower()
        if unit not in ('m', 'mm'):
            raise ValueError('unit must be m or mm')
        actual_length_m = actual_length / 1000.0 if unit == 'mm' else actual_length
        scale_m_per_px = actual_length_m / pixel_distance

        calibration = {
            'status': 'confirmed',
            'method': 'manual_two_point',
            'scale_m_per_px': scale_m_per_px,
            'confidence': 1.0,
            'horizontal': None,
            'vertical': None,
            'axis_difference_percent': None,
            'evidence': [{
                'point_a': point_a,
                'point_b': point_b,
                'pixel_distance': pixel_distance,
                'actual_length': actual_length,
                'unit': unit,
                'actual_length_m': actual_length_m,
            }],
        }
        recognition['scale_calibration'] = calibration
        recognition['room_topology'] = apply_scale_to_room_topology(
            recognition.get('room_topology') or {},
            scale_m_per_px,
        )
        recognition['room_topology'] = _enforce_topology_repair_readiness(
            recognition['room_topology'],
            recognition.get('topology_repair'),
        )
        _save_recognition_payload(target_dir, recognition)
        return jsonify({
            'success': True,
            'scale_calibration': calibration,
            'room_topology': recognition['room_topology'],
        })
    except (TypeError, ValueError) as exc:
        return jsonify({'error': str(exc)}), 400


@app.route('/energy/ai_simulate', methods=['POST'])
@login_required
def ai_simulate():
    """
    使用 AI 识别结果结合详细参数进行能耗计算
    """
    if not (HAS_FLOORPLAN_AI or HAS_VECTOR_FLOORPLAN_AI):
        return jsonify({'error': 'AI module not available'}), 501

    exterior_lock_context = None
    exterior_lock_acquired = False
    try:
        data = request.get_json() or {}
        report_number = data.get('report_number', 'default')
        scale = float(data.get('scale', 0.01))  # 默认 1px = 1cm
        height = float(data.get('height', 3.0))
        floors = int(data.get('floors', 1))

        def parse_boolean(key, default):
            if key not in data:
                return default
            value = data[key]
            if type(value) is not bool:
                raise ValueError(f"{key} 必须是布尔值")
            return value

        calculate_heating = parse_boolean("calculate_heating", True)
        calculate_cooling = parse_boolean("calculate_cooling", True)
        if not calculate_heating and not calculate_cooling:
            raise ValueError("至少选择供暖或制冷中的一种计算")

        heating_system = data.get("heating_system", "heat_pump_air")
        cooling_system = data.get("cooling_system", "vrv")
        heating_efficiency_defaults = {
            "heat_pump_air": 1.90,
            "heat_pump_geo": 1.90,
            "electric": 1.0,
            "gas_boiler": 0.89,
            "coal_boiler": 0.75,
            "district": 0.80,
            "none": 1.0,
        }
        cooling_efficiency_defaults = {
            "central_chiller": 5.0,
            "vrv": 3.8,
            "split_ac": 3.2,
            "evaporative": 8.0,
            "none": 1.0,
        }
        if heating_system not in heating_efficiency_defaults:
            raise ValueError("不支持的供暖系统类型")
        if cooling_system not in cooling_efficiency_defaults:
            raise ValueError("不支持的制冷系统类型")

        def parse_finite_number(key, default, minimum=None, maximum=None, integer=False):
            try:
                value = float(data.get(key, default))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} 必须是有效数字") from exc
            if not math.isfinite(value):
                raise ValueError(f"{key} 必须是有限数字")
            if minimum is not None and value < minimum:
                raise ValueError(f"{key} 不得小于 {minimum}")
            if maximum is not None and value > maximum:
                raise ValueError(f"{key} 不得大于 {maximum}")
            if integer:
                if not value.is_integer():
                    raise ValueError(f"{key} 必须是整数")
                return int(value)
            return value

        area_per_person = parse_finite_number("area_per_person_m2", 10.0, minimum=0.000001)
        fresh_air_per_person = parse_finite_number("fresh_air_m3h_per_person", 30.0, minimum=0.0)
        daily_operation_hours = parse_finite_number("daily_operation_hours", 12.0, minimum=0.0, maximum=24.0)
        annual_operation_days = parse_finite_number(
            "annual_operation_days", 250, minimum=0, maximum=365, integer=True
        )
        recovery_key = (
            "heat_recovery_efficiency"
            if "heat_recovery_efficiency" in data
            else "heat_recovery_efficiency_percent"
        )
        heat_recovery_percent = parse_finite_number(recovery_key, 0.0, minimum=0.0, maximum=100.0)
        infiltration_ach = parse_finite_number("infiltration_ach", 0.0, minimum=0.0)
        fan_power = parse_finite_number("fan_power_w_per_m3h", 0.5, minimum=0.0)
        heating_seasonal_efficiency = parse_finite_number(
            "heating_seasonal_efficiency",
            heating_efficiency_defaults[heating_system],
            minimum=0.000001,
        )
        cooling_seasonal_efficiency = parse_finite_number(
            "cooling_seasonal_efficiency",
            cooling_efficiency_defaults[cooling_system],
            minimum=0.000001,
        )
        winter_design_temperature = parse_finite_number("winter_design_temperature_c", -5.0)
        summer_design_temperature = parse_finite_number("summer_design_temperature_c", 34.9)
        peak_solar_irradiance = parse_finite_number(
            "peak_solar_irradiance_w_m2", 500.0, minimum=0.0
        )
        summer_outdoor_rh = parse_finite_number(
            "summer_outdoor_relative_humidity_percent", 60.0, minimum=0.0, maximum=100.0
        )
        summer_indoor_rh = parse_finite_number(
            "summer_indoor_relative_humidity_percent", 50.0, minimum=0.0, maximum=100.0
        )
        people_sensible = parse_finite_number("sensible_heat_w_per_person", 75.0, minimum=0.0)
        people_latent = parse_finite_number("latent_heat_w_per_person", 55.0, minimum=0.0)
        stable_internal_gain_fraction = parse_finite_number(
            "stable_internal_gain_fraction", 0.0, minimum=0.0, maximum=1.0
        )
        door_invasion_heat = parse_finite_number("door_invasion_heat_w", 0.0, minimum=0.0)
        heating_addition_factor = parse_finite_number("heating_addition_factor", 1.0, minimum=0.0)
        cooling_load_factor = parse_finite_number("cooling_load_factor", 1.0, minimum=0.0)
        lighting_cooling_load_factor = parse_finite_number("lighting_cooling_load_factor", 1.0, minimum=0.0)
        equipment_cooling_load_factor = parse_finite_number("equipment_cooling_load_factor", 1.0, minimum=0.0)
        air_density = parse_finite_number("air_density_kg_m3", 1.13, minimum=0.000001)

        target_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'energy', report_number)

        # 查找已有的 AI 识别结果图
        recognition = _load_recognition_payload(target_dir)
        if recognition is None:
            return jsonify({
                'error': 'No current recognition.json found. Run AI recognition first.'
            }), 404

        initially_confirmed_exterior = recognition.get('exterior_topology') or {}
        if (
            initially_confirmed_exterior.get('confirmed') is True
            and initially_confirmed_exterior.get('load_geometry_ready') is True
        ):
            exterior_lock_context = _exterior_report_lock(Path(target_dir))
            exterior_lock_context.__enter__()
            exterior_lock_acquired = True
            recognition = _load_recognition_payload(target_dir)
            if recognition is None:
                raise ExteriorGenerationConflict(
                    'Recognition generation disappeared while acquiring the exterior lock'
                )

        scale_source = 'request'
        scale_calibration = recognition.get('scale_calibration') or {}
        if scale_calibration.get('status') == 'confirmed':
            calibrated_scale = float(scale_calibration.get('scale_m_per_px') or 0.0)
            if math.isfinite(calibrated_scale) and calibrated_scale > 0:
                scale = calibrated_scale
                scale_source = scale_calibration.get('method') or 'confirmed_calibration'

        # 获取平面图的尺寸和AI分割的几何数据
        geometry = recognition['geometry']
        image_size = recognition.get('image_size') or [0, 0]
        image_width = float(image_size[0] or 0)
        image_height = float(image_size[1] or 0)

        # 优先使用已确认的外轮廓；旧识别结果继续走原有房间/手工回退。
        exterior = recognition.get('exterior_topology') or {}
        room_topology = recognition.get('room_topology') or {}
        room_area_px2 = float(room_topology.get('total_area_px2') or 0.0)
        room_count = int(room_topology.get('room_count') or 0)
        exterior_repair_required = bool(
            (recognition.get('topology_repair') or {}).get('manual_exterior_wall_required')
        )
        exterior_geometry = None
        if exterior.get('confirmed') is True and exterior.get('load_geometry_ready') is True:
            try:
                _require_current_exterior_generation(
                    Path(target_dir), report_number, recognition=recognition,
                )
            except ExteriorGenerationConflict:
                if (
                    not exterior_repair_required
                    and room_topology.get('load_geometry_ready') is True
                    and room_count > 0
                    and math.isfinite(room_area_px2)
                    and room_area_px2 > 0
                ):
                    exterior = {}
                else:
                    raise

        if exterior.get('confirmed') is True and exterior.get('load_geometry_ready') is True:
            from vector_pdf_energy_geometry import build_exterior_energy_geometry

            exterior_geometry = build_exterior_energy_geometry(
                exterior,
                recognition.get('openings') or [],
                storey_height_m=height,
                floors=floors,
                door_height_m=parse_finite_number(
                    'door_height_m', 2.1, minimum=0.1,
                ),
                window_height_m=parse_finite_number(
                    'window_height_m', 1.5, minimum=0.1,
                ),
                door_repeat_count=parse_finite_number(
                    'door_repeat_count', 1, minimum=1, maximum=floors, integer=True,
                ),
                window_repeat_count=parse_finite_number(
                    'window_repeat_count', floors, minimum=1, maximum=floors, integer=True,
                ),
            )
            floor_area_m2 = exterior_geometry['per_floor_footprint_area_m2']
            floor_area_source = 'confirmed_exterior_footprint'
        elif (
            not exterior_repair_required
            and room_count > 0
            and math.isfinite(room_area_px2)
            and room_area_px2 > 0
        ):
            floor_area_m2 = room_area_px2 * (scale ** 2)
            floor_area_source = 'room_polygons'
        elif data.get('floor_area_m2') not in (None, ''):
            floor_area_m2 = parse_finite_number('floor_area_m2', 0.0, minimum=0.000001)
            floor_area_source = 'manual'
        elif 'room_topology' not in recognition:
            # Backward compatibility for recognition.json files created before room polygons existed.
            floor_area_m2 = image_width * image_height * (scale ** 2)
            floor_area_source = 'legacy_image_bounds'
        else:
            raise ValueError('No closed room polygon was recognized; provide a corrected room boundary or manual floor area.')

        if exterior_geometry is not None:
            wall_area_m2 = exterior_geometry['wall_area_m2']
            window_area_m2 = exterior_geometry['window_area_m2']
            door_area_m2 = exterior_geometry['door_area_m2']
            roof_area_m2 = exterior_geometry['per_floor_footprint_area_m2']
        else:
            # Preserve the existing room/manual calculations exactly.
            door_pixels = sum(d.get('area', 0) for d in geometry['doors'])
            wall_total_length = 0
            for wall in geometry['walls']:
                pts = wall['pts']
                for i in range(len(pts) - 1):
                    dx = pts[i+1][0] - pts[i][0]
                    dy = pts[i+1][1] - pts[i][1]
                    wall_total_length += (dx*dx + dy*dy) ** 0.5

            win_total_length = 0
            for win in geometry['windows']:
                pts = win['pts']
                for i in range(len(pts) - 1):
                    dx = pts[i+1][0] - pts[i][0]
                    dy = pts[i+1][1] - pts[i][1]
                    win_total_length += (dx*dx + dy*dy) ** 0.5

            wall_area_m2 = (wall_total_length * scale) * height
            window_area_m2 = (win_total_length * scale) * height
            door_area_m2 = door_pixels * (scale ** 2)
            roof_area_m2 = floor_area_m2

        # 3. 构造传递给 energy_calc.py 的输入参数
        calc_params = {
            "calculation_mode": data.get("calculation_mode", "simple"),
            "geometry": {
                "floor_area_m2": floor_area_m2,
                "wall_area_m2": wall_area_m2,
                "window_area_m2": window_area_m2,
                "roof_area_m2": roof_area_m2,
                "door_area_m2": door_area_m2,
                "include_roof": data.get("include_roof") is True,
                "include_floor": data.get("include_floor") is True
            },
            "building": {
                "height": height,
                "floors": floors,
                "building_type": data.get("building_type", "office"),
                "orientation": data.get("orientation", "south")
            },
            "envelope": {
                "u_wall": float(data.get("u_wall", 0.6)),
                "u_window": float(data.get("u_window") or data.get("u_win") or 2.5),
                "u_roof": float(data.get("u_roof", 0.4)),
                "u_floor": float(data.get("u_floor", 0.3)),
                "u_door": float(data.get("u_door", 3.0)),
                "shgc": float(data.get("shgc", 0.4)),
                "window_shgc": float(data.get("window_shgc", data.get("shgc", 0.4))),
                "curtain_shading": float(data.get("curtain_shading", 1.0)),
                "peak_solar_irradiance_w_m2": peak_solar_irradiance,
                "solar_orientation_factor": float(data.get("solar_orientation_factor", 1.0)),
                "floor_contact_type": data.get("floor_contact_type", "ground"),
                "floor_contact_factor": float(data.get("floor_contact_factor", 1.0)),
                "annual_solar_irradiation_kwh_m2a": max(
                    0.0,
                    float(data.get("annual_solar_irradiation_kwh_m2a", 150.0)),
                ),
                "window_air_tightness": data.get("window_air_tightness", "level_6"),
                "window_air_tightness_value": float(data.get("window_air_tightness_value", 6.0))
            },
            "climate": {
                "city_id": data.get("city_id", "beijing")
            },
            "heating": {
                "enabled": calculate_heating,
                "system_type": heating_system,
                "t_set": float(data.get("t_heat", 18.0)),
                "seasonal_efficiency": heating_seasonal_efficiency,
                "outdoor_design_temperature_c": winter_design_temperature,
                "stable_internal_gain_fraction": stable_internal_gain_fraction,
                "door_invasion_heat_w": door_invasion_heat,
                "heating_addition_factor": heating_addition_factor
            },
            "cooling": {
                "enabled": calculate_cooling,
                "system_type": cooling_system,
                "t_set": float(data.get("t_cool", 26.0)),
                "seasonal_efficiency": cooling_seasonal_efficiency,
                "outdoor_design_temperature_c": summer_design_temperature
            },
            "lighting": {
                "lpd": float(data.get("lpd", 8.0)),
                "control_factor": float(data.get("lighting_control", 1.0)),
                "cooling_load_factor": lighting_cooling_load_factor
            },
            "occupancy": {
                "area_per_person_m2": area_per_person,
                "sensible_heat_w_per_person": people_sensible,
                "latent_heat_w_per_person": people_latent,
                "cooling_load_factor": cooling_load_factor
            },
            "ventilation": {
                "fresh_air_m3h_per_person": fresh_air_per_person,
                "heat_recovery_efficiency": heat_recovery_percent / 100.0,
                "infiltration_ach": infiltration_ach,
                "fan_power_w_per_m3h": fan_power,
                "summer_outdoor_relative_humidity_percent": summer_outdoor_rh,
                "summer_indoor_relative_humidity_percent": summer_indoor_rh,
                "air_density_kg_m3": air_density
            },
            "dhw": {
                "occupants": int(data.get("dhw_occupants", max(1, int(floor_area_m2 * floors * 0.1)))),
                "daily_liter_pp": float(data.get("dhw_liter_pp", 5.0)),
                "efficiency": float(data.get("dhw_efficiency", 0.85))
            },
            "equipment": {
                "epd": float(data.get("epd", 15.0)),
                "cooling_load_factor": equipment_cooling_load_factor
            },
            "schedule": {
                "daily_operation_hours": daily_operation_hours,
                "annual_operation_days": annual_operation_days
            },
            "detailed_envelope": data.get("detailed_envelope", {})
        }

        # 4. 调用能耗计算引擎
        if HAS_ENERGY_CALC:
            res = energy_calc.calculate_energy(calc_params)
            if HAS_DESIGN_LOAD_CALC:
                res["design_loads"] = design_load_calc.calculate_design_loads(calc_params)
            res["recognition_source"] = "recognition_json"
            res["recognition_model_version"] = recognition.get("model", {}).get("version")
            res["recognition_preprocessing"] = recognition.get("preprocessing", {})
            res["floor_area_source"] = floor_area_source
            res["scale_source"] = scale_source

            # 将计算结果、输入参数、几何信息持久化到 SQLite 数据库中
            try:
                username = session.get('username', 'guest')
                conn = get_db_connection()
                existing = conn.execute('SELECT id FROM reports WHERE report_number = ?', (report_number,)).fetchone()
                if existing:
                    conn.execute('''
                        UPDATE reports 
                        SET username = ?, geometry_used = ?, params = ?, results = ?, created_at = datetime('now')
                        WHERE report_number = ?
                    ''', (username, json.dumps(calc_params.get('geometry', {})), json.dumps(calc_params), json.dumps(res), report_number))
                else:
                    conn.execute('''
                        INSERT INTO reports (username, report_number, geometry_used, params, results)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (username, report_number, json.dumps(calc_params.get('geometry', {})), json.dumps(calc_params), json.dumps(res)))
                conn.commit()
                conn.close()
            except Exception as db_err:
                logger.error(f"Database save report error: {db_err}", exc_info=True)

            return jsonify(res)
        else:
            return jsonify({'error': 'Energy calculation engine missing on server'}), 500

    except ExteriorGenerationConflict as e:
        return jsonify({'error': str(e)}), 409
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"AI simulate error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    finally:
        if exterior_lock_acquired:
            exterior_lock_context.__exit__(None, None, None)


@app.route('/energy/report/<report_number>', methods=['GET'])
@login_required
def get_energy_report(report_number):
    """
    根据 report_number 获取历史能效报告
    """
    try:
        conn = get_db_connection()
        row = conn.execute('SELECT * FROM reports WHERE report_number = ?', (report_number,)).fetchone()
        conn.close()
        if not row:
            return jsonify({'error': 'Report not found'}), 404

        return jsonify({
            'report_number': row['report_number'],
            'username': row['username'],
            'created_at': row['created_at'],
            'geometry_used': json.loads(row['geometry_used'] or '{}'),
            'params': json.loads(row['params'] or '{}'),
            'results': json.loads(row['results'] or '{}')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/energy/reports', methods=['GET'])
@login_required
def get_user_reports():
    """
    获取当前用户所有的历史能效报告列表
    """
    try:
        username = session.get('username', 'guest')
        conn = get_db_connection()
        rows = conn.execute('SELECT report_number, created_at, geometry_used, results FROM reports WHERE username = ? ORDER BY created_at DESC', (username,)).fetchall()
        conn.close()

        results = []
        for r in rows:
            res_data = json.loads(r['results'] or '{}')
            summary = res_data.get('summary', {})
            results.append({
                'report_number': r['report_number'],
                'created_at': r['created_at'],
                'floor_area': json.loads(r['geometry_used'] or '{}').get('floor_area_m2', 0),
                'total_energy': summary.get('total_energy_kwh', 0),
                'eui': summary.get('eui', 0),
                'rating': summary.get('rating', '-')
            })
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
