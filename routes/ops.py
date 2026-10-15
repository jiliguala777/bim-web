"""Ops HTTP routes with explicit runtime dependencies."""
from flask import Blueprint


def register(app, runtime):
    blueprint = Blueprint("ops", __name__)

    @blueprint.route('/ops/benchmark', methods=['GET', 'POST'])
    @runtime.admin_required
    def ops_benchmark():
        """运行模型精度基准测试"""
        if not runtime.HAS_BENCHMARK:
            return runtime.jsonify({'error': 'Benchmark module not available'}), 501

        try:
            case_name = runtime.request.args.get('case')
            if not case_name and runtime.request.method == 'POST':
                case_name = (runtime.request.get_json(silent=True) or {}).get('case')
            report = runtime.run_benchmark(case_name)
            return runtime.jsonify(report)
        except Exception as e:
            runtime.logger.error(f"Benchmark error: {e}")
            return runtime.jsonify({'error': str(e)}), 500

    @blueprint.route('/ops/benchmark/generate_test_files', methods=['POST'])
    @runtime.admin_required
    def generate_test_files():
        """生成 BESTEST 标准测试文件 (DXF + IFC)"""
        if not runtime.HAS_BENCHMARK:
            return runtime.jsonify({'error': 'Benchmark module not available'}), 501
        try:
            test_dir = runtime.os.path.join(runtime.app.config['UPLOAD_FOLDER'], 'ops', 'bestest')
            runtime.os.makedirs(test_dir, exist_ok=True)

            files = {}
            for case_name in runtime.BESTEST_BENCHMARKS:
                dxf_path = runtime.os.path.join(test_dir, f'{case_name}.dxf')
                ifc_path = runtime.os.path.join(test_dir, f'{case_name}.ifc')

                if runtime.generate_test_dxf(dxf_path, case_name):
                    files[f'{case_name}.dxf'] = 'generated'

                if runtime.HAS_IFC and runtime.generate_test_ifc(ifc_path, case_name):
                    files[f'{case_name}.ifc'] = 'generated'

            return runtime.jsonify({'status': 'success', 'files': files, 'directory': test_dir})
        except Exception as e:
            return runtime._internal_report_error_response("Benchmark fixture generation error", e)

    @blueprint.route('/ops/health', methods=['GET'])
    @runtime.admin_required
    def ops_health():
        """综合健康检查"""
        health = {
            'timestamp': runtime.time.time(),
            'services': {},
            'system': {},
        }

        # Helper: scan processes with psutil
        def _count_procs(name_pattern):
            count = 0
            for proc in runtime.psutil.process_iter(['name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info.get('cmdline') or [])
                    if name_pattern in cmdline or name_pattern in (proc.info.get('name') or ''):
                        count += 1
                except (runtime.psutil.NoSuchProcess, runtime.psutil.AccessDenied):
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
                'cpu_percent': runtime.psutil.cpu_percent(interval=None),
            'memory': {
                'total_gb': round(runtime.psutil.virtual_memory().total / 1024**3, 1),
                'used_gb': round(runtime.psutil.virtual_memory().used / 1024**3, 1),
                'percent': runtime.psutil.virtual_memory().percent,
            },
            'disk': {
                'total_gb': round(runtime.psutil.disk_usage('/').total / 1024**3, 1),
                'used_gb': round(runtime.psutil.disk_usage('/').used / 1024**3, 1),
                'percent': runtime.psutil.disk_usage('/').percent,
            },
        }

        # 5. Overall status
        all_running = all(
            s.get('status') == 'running'
            for s in health['services'].values()
        )
        health['overall'] = 'healthy' if all_running else 'degraded'

        return runtime.jsonify(health)

    @blueprint.route('/ops/logs', methods=['GET'])
    @runtime.admin_required
    def ops_logs():
        """获取最近的错误日志"""
        try:
            lines = int(runtime.request.args.get('lines', 50))
        except (ValueError, TypeError):
            return runtime.jsonify({'error': 'lines must be an integer between 1 and 500'}), 400
        source = runtime.request.args.get('source', 'all')  # all, gunicorn, nginx
        if not 1 <= lines <= 500 or source not in {'all', 'gunicorn', 'nginx'}:
            return runtime.jsonify({'error': 'Invalid log query'}), 400

        logs = {}

        if source in ('all', 'gunicorn'):
            try:
                log_path = runtime.os.environ.get('GUNICORN_LOG', runtime.os.path.join(runtime.BASE_DIR, 'gunicorn.log'))
                with open(log_path, 'r') as f:
                    content = f.readlines()
                    logs['gunicorn'] = content[-lines:]
            except OSError:
                logs['gunicorn'] = ['File not accessible']

        if source in ('all', 'nginx'):
            try:
                result = runtime.subprocess.run(
                    ['tail', f'-{lines}', '/var/log/nginx/error.log'],
                    capture_output=True, text=True, timeout=5
                )
                logs['nginx_error'] = result.stdout.strip().split('\n') if result.stdout else []
            except (OSError, runtime.subprocess.SubprocessError):
                logs['nginx_error'] = ['File not accessible']

        return runtime.jsonify(logs)

    @blueprint.route('/ops/metrics', methods=['GET'])
    @runtime.admin_required
    def ops_metrics():
        """系统详细指标"""
        return runtime.jsonify({
            'cpu': {
                'count': runtime.psutil.cpu_count(),
                'percent_per_cpu': runtime.psutil.cpu_percent(interval=None, percpu=True),
                'load_avg': list(runtime.os.getloadavg()) if hasattr(runtime.os, 'getloadavg') else None,
            },
            'memory': {
                'total_mb': round(runtime.psutil.virtual_memory().total / 1024**2),
                'available_mb': round(runtime.psutil.virtual_memory().available / 1024**2),
                'percent': runtime.psutil.virtual_memory().percent,
                'swap_percent': runtime.psutil.swap_memory().percent,
            },
            'disk': {
                'total_gb': round(runtime.psutil.disk_usage('/').total / 1024**3, 1),
                'free_gb': round(runtime.psutil.disk_usage('/').free / 1024**3, 1),
                'percent': runtime.psutil.disk_usage('/').percent,
            },
            'uptime_hours': round((runtime.time.time() - runtime.psutil.boot_time()) / 3600, 1),
            'connections': len(runtime.psutil.net_connections()),
        })

    app.register_blueprint(blueprint)
    return {'ops_benchmark': ops_benchmark, 'generate_test_files': generate_test_files, 'ops_health': ops_health, 'ops_logs': ops_logs, 'ops_metrics': ops_metrics}
