"""Reports HTTP routes with explicit runtime dependencies."""
from flask import Blueprint


def register(app, runtime):
    blueprint = Blueprint("reports", __name__)

    @blueprint.route('/api/v1/simulation_data/<report_number>')
    @runtime.login_required
    def get_simulation_data(report_number):
        """
        Returns extracted geometry and simulation constants for local API bridge.
        Confidential device data is NOT handled here.
        """
        try:
            context = runtime._energy_report_context(report_number)
            dxf_path = runtime.resolve_report_child_file(
                context.report_dir, 'building_plan.dxf', required=False,
            )

            # In a real scenario, we'd store the last simulation results in a DB or JSON file
            # For now, we'll re-extract or return defaults
            return runtime.jsonify({
                'report_number': context.report_number,
                'geometry': {
                    'floor_area': 500, # Placeholder or extracted value
                    'perimeter': 100
                },
                'env_params': {
                    'u_wall': 0.6,
                    'u_win': 2.5
                }
            })
        except runtime._REPORT_STORAGE_EXCEPTIONS as e:
            return runtime._report_storage_error_response(e)

    @blueprint.route('/energy/report/<report_number>', methods=['GET'])
    @runtime.login_required
    def get_energy_report(report_number):
        """
        根据 report_number 获取历史能效报告
        """
        conn = None
        try:
            context = runtime._resolved_energy_report_context(
                report_number,
                runtime._requested_owner_username(),
            )
            conn = runtime.get_db_connection()
            row = conn.execute('''
                SELECT username, report_number, status, created_at, updated_at,
                       geometry_used, params, results
                FROM reports
                WHERE username = ? AND report_number = ?
            ''', (context.owner_username, context.report_number)).fetchone()
            if not row:
                return runtime.jsonify({'error': 'Report not found'}), 404

            return runtime.jsonify({
                'report_number': row['report_number'],
                'username': row['username'],
                'status': row['status'],
                'created_at': row['created_at'],
                'updated_at': row['updated_at'],
                'geometry_used': runtime.json.loads(row['geometry_used'] or '{}'),
                'params': runtime.json.loads(row['params'] or '{}'),
                'results': runtime.json.loads(row['results'] or '{}')
            })
        except runtime._REPORT_STORAGE_EXCEPTIONS as exc:
            return runtime._report_storage_error_response(exc)
        except Exception as exc:
            return runtime._internal_report_error_response('Report detail error', exc)
        finally:
            if conn is not None:
                conn.close()

    @blueprint.route('/energy/reports', methods=['GET'])
    @runtime.login_required
    def get_user_reports():
        """
        获取当前用户所有的历史能效报告列表
        """
        conn = None
        try:
            conn = runtime.get_db_connection()
            if runtime.session.get('is_admin') is True:
                rows = conn.execute('''
                    SELECT username, report_number, status, created_at, updated_at,
                           geometry_used, results
                    FROM reports
                    ORDER BY updated_at DESC, id DESC
                ''').fetchall()
            else:
                username = runtime._effective_report_owner()
                rows = conn.execute('''
                    SELECT username, report_number, status, created_at, updated_at,
                           geometry_used, results
                    FROM reports
                    WHERE username = ?
                    ORDER BY updated_at DESC, id DESC
                ''', (username,)).fetchall()

            results = []
            for r in rows:
                res_data = runtime.json.loads(r['results'] or '{}')
                if not isinstance(res_data, dict):
                    res_data = {}
                summary = res_data.get('summary', {})
                if not isinstance(summary, dict):
                    summary = {}
                geometry_data = runtime.json.loads(r['geometry_used'] or '{}')
                if not isinstance(geometry_data, dict):
                    geometry_data = {}
                results.append({
                    'username': r['username'],
                    'report_number': r['report_number'],
                    'status': r['status'],
                    'created_at': r['created_at'],
                    'updated_at': r['updated_at'],
                    'floor_area': geometry_data.get('floor_area_m2', 0),
                    'total_energy': summary.get('total_energy_kwh', 0),
                    'eui': summary.get('eui', 0),
                    'rating': summary.get('rating', '-')
                })
            return runtime.jsonify(results)
        except runtime._REPORT_STORAGE_EXCEPTIONS as exc:
            return runtime._report_storage_error_response(exc)
        except Exception as exc:
            return runtime._internal_report_error_response('Report list error', exc)
        finally:
            if conn is not None:
                conn.close()

    app.register_blueprint(blueprint)
    return {'get_simulation_data': get_simulation_data, 'get_energy_report': get_energy_report, 'get_user_reports': get_user_reports}
