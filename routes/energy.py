"""Energy HTTP routes with explicit runtime dependencies."""
from flask import Blueprint


def register(app, runtime):
    blueprint = Blueprint("energy", __name__)

    @blueprint.route('/')
    def landing():
        """ 公开着陆页 - 包含功能介绍与登录 """
        return runtime.render_template(
            'landing.html',
            logged_in=runtime.session.get('logged_in') is True and bool(runtime.session.get('username')),
        )

    @blueprint.route('/energy')
    @runtime.login_required
    def index():
        return runtime.render_template('energy.html', has_deps=runtime.HAS_ENERGY_DEPS)

    @blueprint.route('/home')
    @runtime.login_required
    def home():
        return runtime.render_template('index.html')

    @blueprint.route('/floorplan')
    @runtime.login_required
    def floorplan():
        return runtime.render_template('index.html')

    @blueprint.route('/health')
    def health():
        return runtime.jsonify({'status': 'healthy', 'timestamp': runtime.time.time()})

    @blueprint.route('/energy/library/walls', methods=['GET'])
    @runtime.login_required
    def energy_library_walls():
        if not runtime.HAS_BUILDING_LIBRARY:
            return runtime.jsonify({'error': 'Building library module is not available'}), 503
        try:
            return runtime.jsonify({'items': runtime.list_wall_assemblies()})
        except Exception as e:
            runtime.logger.error(f"Load wall library error: {e}", exc_info=True)
            return runtime.jsonify({'error': str(e)}), 500

    @blueprint.route('/energy/library/materials', methods=['GET'])
    @runtime.login_required
    def energy_library_materials():
        if not runtime.HAS_BUILDING_LIBRARY:
            return runtime.jsonify({'error': 'Building library module is not available'}), 503
        try:
            return runtime.jsonify({'items': runtime.list_materials()})
        except Exception as e:
            runtime.logger.error(f"Load material library error: {e}", exc_info=True)
            return runtime.jsonify({'error': str(e)}), 500

    @blueprint.route('/energy/library/envelope/<category>', methods=['GET'])
    @runtime.login_required
    def energy_library_envelope(category):
        if not runtime.HAS_BUILDING_LIBRARY:
            return runtime.jsonify({'error': 'Building library module is not available'}), 503
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
            return runtime.jsonify({'error': 'Unsupported envelope parameter category'}), 400
        try:
            return runtime.jsonify({'items': runtime.list_envelope_parameters(category)})
        except Exception as e:
            runtime.logger.error(f"Load envelope library error: {e}", exc_info=True)
            return runtime.jsonify({'error': str(e)}), 500

    @blueprint.route('/energy/upload', methods=['POST'])
    @runtime.login_required
    def energy_upload():
        context = None
        try:
            report_number = runtime.request.form.get('report_number', 'default')
            context = runtime._energy_report_context(report_number, create=True)
            target_dir = context.report_dir
            runtime._upsert_report_status(context.owner_username, context.report_number, 'created')

            result = {'status': 'success', 'files': {}}
            files_uploaded = False
            dxf_uploaded = False
            dxf_path = runtime.resolve_report_child_destination(target_dir, 'building_plan.dxf')

            if 'dxf_file' in runtime.request.files:
                f = runtime.request.files['dxf_file']
                if f and runtime.allowed_file(f.filename, runtime.ALLOWED_DXF):
                    runtime._atomic_save_upload(f, dxf_path)
                    result['files']['dxf'] = 'building_plan.dxf'
                    dxf_uploaded = True
                    files_uploaded = True

            # 处理光栅图/PDF上传 (Learning-based 接口)
            raster_uploaded = False
            if 'raster_file' in runtime.request.files:
                f = runtime.request.files['raster_file']
                if f and runtime.allowed_file(f.filename, runtime.ALLOWED_RASTER):
                    ext = f.filename.rsplit('.', 1)[1].lower()
                    raster_path = runtime.resolve_report_child_destination(
                        target_dir, f'building_plan.{ext}',
                    )
                    runtime._atomic_save_upload(f, raster_path)
                    raster_path = runtime.resolve_report_child_file(
                        target_dir, f'building_plan.{ext}',
                    )
                    result['files']['raster'] = f'building_plan.{ext}'
                    raster_uploaded = True
                    files_uploaded = True

                    # 如果是 PDF，尝试转换为 PNG 进行后续解析
                    if ext == 'pdf':
                        try:
                            from pdf2image import convert_from_path
                            images = convert_from_path(
                                raster_path,
                                poppler_path=runtime._resolve_poppler_path(),
                            )
                            if images:
                                converted_path = runtime.resolve_report_child_destination(
                                    target_dir, 'building_plan.png',
                                )
                                runtime._atomic_save_image(images[0], converted_path, 'PNG')
                                result['files']['raster_converted'] = 'building_plan.png'
                        except Exception as e:
                            runtime.logger.warning(f"PDF to Image conversion failed: {e}")

            # 仅在本次请求确实上传了新文件时才解析层级信息
            if dxf_uploaded and dxf_path.is_file():
                dxf_path = runtime.resolve_report_child_file(target_dir, 'building_plan.dxf')
                doc = runtime.ezdxf.readfile(dxf_path)
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
            elif not dxf_uploaded and not raster_uploaded and not 'epw_file' in runtime.request.files:
                 # 如果啥也没传，明确告知
                 result['message'] = 'No new files uploaded in this request.'

            if 'epw_file' in runtime.request.files:
                f = runtime.request.files['epw_file']
                if f and runtime.allowed_file(f.filename, runtime.ALLOWED_EPW):
                    p = runtime.resolve_report_child_destination(target_dir, 'weather_data.epw')
                    runtime._atomic_save_upload(f, p)
                    result['files']['epw'] = 'weather_data.epw'
                    files_uploaded = True
            elif runtime.request.form.get('city_id'):
                city_id = runtime.request.form['city_id']
                src = runtime.os.path.join(runtime.WEATHER_DATA_DIR, f'{city_id}.epw')
                if runtime.os.path.exists(src):
                    import shutil
                    destination = runtime.resolve_report_child_destination(
                        target_dir, 'weather_data.epw',
                    )
                    runtime._atomic_copy_file(src, destination)
                    result['files']['epw'] = 'weather_data.epw'
                    files_uploaded = True

            if files_uploaded:
                runtime._upsert_report_status(context.owner_username, context.report_number, 'uploaded')

            return runtime.jsonify(result)
        except (runtime.ReportAccessDenied, runtime.InvalidReportPath, runtime.ReportOwnerNotFound) as e:
            return runtime._report_storage_error_response(e)
        except Exception as e:
            if context is not None:
                try:
                    runtime._update_report_status_if_exists(context.owner_username, context.report_number, 'failed')
                except Exception:
                    runtime.logger.exception("Could not update failed upload status")
            return runtime._internal_report_error_response("Upload error", e)

    @blueprint.route('/energy/layers/geometry', methods=['GET'])
    @runtime.login_required
    def get_all_layers_geometry():
        """获取所有图层的几何数据及边框"""
        try:
            report_number = runtime.request.args.get('project', 'default')
            context = runtime._energy_report_context(report_number)
            dxf_path = runtime.resolve_report_child_file(
                context.report_dir, 'building_plan.dxf', required=False,
            )

            if not runtime.os.path.exists(dxf_path):
                return runtime.jsonify({'error': 'DXF file not found'}), 404

            doc = runtime.ezdxf.readfile(dxf_path)
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
            return runtime.jsonify({'layers': layers_data, 'bounds': bounds})
        except Exception as e:
            if isinstance(e, runtime._REPORT_STORAGE_EXCEPTIONS):
                return runtime._report_storage_error_response(e)
            return runtime._internal_report_error_response("Layer geometry error", e)

    @blueprint.route('/energy/geometry', methods=['POST'])
    @runtime.login_required
    def energy_geometry():
        """获取指定图层的几何数据用于预览"""
        try:
            data = runtime.request.get_json()
            report_number = data.get('report_number', 'default')
            layers = data.get('layers', [])
            scale = float(data.get('scale', 1.0))

            context = runtime._energy_report_context(report_number, requested_owner=runtime._requested_owner_username(data))
            dxf_path = runtime.resolve_report_child_file(
                context.report_dir, 'building_plan.dxf', required=False,
            )

            if not runtime.os.path.exists(dxf_path):
                return runtime.jsonify({'error': 'DXF file not found'}), 404

            doc = runtime.ezdxf.readfile(dxf_path)
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
            return runtime.jsonify({'geoms': geoms})
        except Exception as e:
            if isinstance(e, runtime._REPORT_STORAGE_EXCEPTIONS):
                return runtime._report_storage_error_response(e)
            return runtime._internal_report_error_response("Energy geometry error", e)

    @blueprint.route('/energy/geometry_advanced', methods=['POST'])
    @runtime.login_required
    def energy_geometry_advanced():
        """
        使用高级识别算法 (Alpha Shape + Dilate-Subtract) 获取几何数据
        """
        try:
            data = runtime.request.get_json()
            report_number = data.get('report_number', 'default')
            scale = float(data.get('scale', 1.0))

            context = runtime._energy_report_context(report_number, requested_owner=runtime._requested_owner_username(data))
            target_dir = context.report_dir
            dxf_path = runtime.resolve_report_child_file(
                target_dir, 'building_plan.dxf', required=False,
            )

            if not runtime.os.path.exists(dxf_path):
                return runtime.jsonify({'error': 'DXF file not found'}), 404

            if runtime.AdvancedCADRecognition is None:
                return runtime.jsonify({'error': 'Advanced CAD Recognition module or dependencies (OpenCV) not found'}), 512

            # 初始化高级引擎
            engine = runtime.AdvancedCADRecognition(dxf_path)

            # 如果有位图则使用位图路由，否则使用矢量启发式路由 (混合模式)
            png_path = runtime.resolve_report_child_file(
                target_dir, 'building_plan.png', required=False,
            )
            jpg_path = runtime.resolve_report_child_file(
                target_dir, 'building_plan.jpg', required=False,
            )

            raster_path = png_path if runtime.os.path.exists(png_path) else (jpg_path if runtime.os.path.exists(jpg_path) else None)

            if raster_path:
                # 方案 A: 基于图像的语义识别 (Learning-based)
                # 这里调用报告中提到的技术路线
                try:
                    result = engine.run_advanced_pipeline(raster_path)
                    walls = [{"type": "polyline", "pts": [[p[0]*scale, p[1]*scale] for p in result['walls']]}]
                except Exception as e:
                    runtime.logger.warning(f"Raster recognition failed, falling back to vector: {e}")
                    raster_path = None # 触发 Fallback

            if not raster_path:
                # 方案 B: 基于矢量的启发式识别 (Vector-based)
                # 处理传统的 CAD 矢量提取
                doc = runtime.ezdxf.readfile(dxf_path)
                msp = doc.modelspace()
                lines = []
                for e in msp:
                    if e.dxftype() in ['LINE', 'LWPOLYLINE']:
                        # 使用报告中的 '墙体最小面积' 逻辑的矢量变体
                        # 简化：获取所有线条并执行缓冲区合并
                        if e.dxftype() == 'LINE':
                            lines.append(runtime.LineString([(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]))
                        else:
                            pts = [(p[0], p[1]) for p in e.get_points(format='xy')]
                            if len(pts) > 1: lines.append(runtime.LineString(pts))

                # 使用 Alpha Shape 逻辑 (Shapely buffer 0.1 + union)
                merged = runtime.unary_union([l.buffer(2.0) for l in lines]) # 2.0 像素宽度兼容
                if isinstance(merged, runtime.MultiPolygon):
                    poly = max(merged.geoms, key=lambda p: p.area)
                else:
                    poly = merged

                # 正则化
                coords = list(poly.exterior.coords)
                walls = [{"type": "polyline", "pts": [[p[0]*scale, p[1]*scale] for p in coords]}]

            return runtime.jsonify({'geoms': walls, 'mode': 'advanced'})
        except Exception as e:
            if isinstance(e, runtime._REPORT_STORAGE_EXCEPTIONS):
                return runtime._report_storage_error_response(e)
            return runtime._internal_report_error_response("Advanced geometry error", e)

    @blueprint.route('/energy/calculate', methods=['POST'])
    @runtime.login_required
    def energy_calculate():
        """启动计算任务 (支持 Simple 和 EnergyPlus)"""
        data = runtime.request.get_json() or {}
        try:
            context = runtime._energy_report_context(
                data.get('report_number', 'default'),
                requested_owner=runtime._requested_owner_username(data),
            )
        except runtime._REPORT_STORAGE_EXCEPTIONS as e:
            return runtime._report_storage_error_response(e)
        data = dict(data)
        data['report_number'] = context.report_number
        data['owner_username'] = context.owner_username
        job_id = str(runtime.uuid.uuid4())
        mode = data.get('mode', 'simple') # 默认简单模式
        if mode not in {'simple', 'energyplus'}:
            return runtime.jsonify({'error': 'Unsupported simulation mode'}), 400
        if mode == 'energyplus' and runtime.energyplus_engine is None:
            return runtime.jsonify({'error': 'Legacy EnergyPlus engine is not configured'}), 501

        # 初始化状态
        runtime.simulation_jobs.set(job_id, {
            'owner_username': context.owner_username,
            'report_number': context.report_number,
            'status': 'processing',
            'progress': 0,
            'result': None,
            'error': None
        })
        runtime._upsert_report_status(
            context.owner_username, context.report_number, 'calculating',
        )

        try:
            runtime.task_dispatcher.submit(mode, job_id, data)
        except (runtime.QueueFull, runtime.QueueUnavailable):
            runtime._update_report_status_if_exists(context.owner_username, context.report_number, 'failed')
            runtime.simulation_jobs.update(job_id, status='failed', error='Simulation queue unavailable')
            return runtime.jsonify({'error': 'Simulation queue unavailable; retry later'}), 503, {'Retry-After': '10'}
        except Exception as exc:
            runtime._update_report_status_if_exists(
                context.owner_username, context.report_number, 'failed',
            )
            runtime.simulation_jobs.update(job_id, status='failed', error='Simulation failed')
            return runtime._internal_report_error_response('Cannot start simulation task', exc)
        return runtime.jsonify({'job_id': job_id})

    @blueprint.route('/energy/status/<job_id>')
    @runtime.login_required
    def energy_status(job_id):
        try:
            job = runtime.simulation_jobs.get(job_id)
        except ValueError:
            return runtime.jsonify({'error': 'Job not found'}), 404
        if not isinstance(job, dict):
            return runtime.jsonify({'error': 'Job not found'}), 404
        job_owner = job.get('owner_username')
        job_report_number = job.get('report_number')
        if not isinstance(job_owner, str) or not isinstance(job_report_number, str):
            return runtime.jsonify({'error': 'Job not found'}), 404
        try:
            context = runtime._resolved_energy_report_context(
                job_report_number,
                runtime._requested_owner_username(),
            )
            if context.owner_username != job_owner:
                raise runtime.ReportAccessDenied('job belongs to another report owner')
        except runtime._REPORT_STORAGE_EXCEPTIONS:
            return runtime.jsonify({'error': 'Job not found'}), 404
        return runtime.jsonify(job)

    @blueprint.route('/energy/upload_ifc', methods=['POST'])
    @runtime.login_required
    def upload_ifc():
        """上传 IFC 文件并解析建筑信息"""
        report_number = runtime.request.form.get('report_number', 'default')
        try:
            requested_owner = runtime._requested_owner_username()
            runtime._resolved_energy_report_context(report_number, requested_owner)
        except runtime._REPORT_STORAGE_EXCEPTIONS as e:
            return runtime._report_storage_error_response(e)
        if not runtime.HAS_IFC:
            return runtime.jsonify({'error': 'IFC support not available (ifcopenshell not installed)'}), 501

        context = None
        try:
            context = runtime._energy_report_context(report_number, requested_owner=requested_owner, create=True)
            target_dir = context.report_dir
            runtime._upsert_report_status(context.owner_username, context.report_number, 'created')

            if 'ifc_file' not in runtime.request.files:
                return runtime.jsonify({'error': 'No IFC file provided'}), 400

            f = runtime.request.files['ifc_file']
            if not f or not runtime.allowed_file(f.filename, runtime.ALLOWED_IFC):
                return runtime.jsonify({'error': 'Invalid file type. Only .ifc files accepted'}), 400

            ifc_path = runtime.resolve_report_child_destination(target_dir, 'building_model.ifc')
            runtime._atomic_save_upload(f, ifc_path)
            ifc_path = runtime.resolve_report_child_file(target_dir, 'building_model.ifc')
            runtime._upsert_report_status(context.owner_username, context.report_number, 'uploaded')

            # 解析 IFC
            parser = runtime.IFCParser(ifc_path)

            result = {
                'status': 'success',
                'source': 'IFC',
                'project_info': parser.get_project_info(),
                'element_summary': parser.get_element_summary(),
                'storeys': parser.get_storeys(),
                'simulation_data': parser.extract_for_simulation(),
            }

            return runtime.jsonify(result)
        except runtime._REPORT_STORAGE_EXCEPTIONS as e:
            return runtime._report_storage_error_response(e)
        except Exception as e:
            if context is not None:
                try:
                    runtime._update_report_status_if_exists(context.owner_username, context.report_number, 'failed')
                except Exception:
                    runtime.logger.exception("Could not update failed IFC upload status")
            return runtime._internal_report_error_response("IFC upload error", e)

    @blueprint.route('/energy/ifc_properties', methods=['GET'])
    @runtime.login_required
    def ifc_properties():
        """获取已上传 IFC 模型的属性树"""
        report_number = runtime.request.args.get('project', 'default')
        try:
            requested_owner = runtime._requested_owner_username()
            runtime._resolved_energy_report_context(report_number, requested_owner)
        except runtime._REPORT_STORAGE_EXCEPTIONS as e:
            return runtime._report_storage_error_response(e)
        if not runtime.HAS_IFC:
            return runtime.jsonify({'error': 'IFC support not available'}), 501

        try:
            context = runtime._energy_report_context(report_number, requested_owner=requested_owner)
            ifc_path = runtime.resolve_report_child_file(
                context.report_dir, 'building_model.ifc', required=False,
            )

            if not runtime.os.path.exists(ifc_path):
                return runtime.jsonify({'error': 'IFC file not found. Upload one first.'}), 404

            parser = runtime.IFCParser(ifc_path)
            tree = parser.get_property_tree()

            return runtime.jsonify(tree)
        except runtime._REPORT_STORAGE_EXCEPTIONS as e:
            return runtime._report_storage_error_response(e)
        except Exception as e:
            return runtime._internal_report_error_response("IFC property tree error", e)

    @blueprint.route('/energy/ifc_walls', methods=['GET'])
    @runtime.login_required
    def ifc_walls():
        """获取 IFC 模型中的墙体列表及热工参数"""
        report_number = runtime.request.args.get('project', 'default')
        try:
            requested_owner = runtime._requested_owner_username()
            runtime._resolved_energy_report_context(report_number, requested_owner)
        except runtime._REPORT_STORAGE_EXCEPTIONS as e:
            return runtime._report_storage_error_response(e)
        if not runtime.HAS_IFC:
            return runtime.jsonify({'error': 'IFC support not available'}), 501

        try:
            context = runtime._energy_report_context(report_number, requested_owner=requested_owner)
            ifc_path = runtime.resolve_report_child_file(
                context.report_dir, 'building_model.ifc', required=False,
            )

            if not runtime.os.path.exists(ifc_path):
                return runtime.jsonify({'error': 'IFC file not found'}), 404

            parser = runtime.IFCParser(ifc_path)
            walls = parser.get_walls()
            windows = parser.get_windows()
            spaces = parser.get_spaces()

            return runtime.jsonify({
                'walls': walls,
                'windows': windows,
                'spaces': spaces,
            })
        except runtime._REPORT_STORAGE_EXCEPTIONS as e:
            return runtime._report_storage_error_response(e)
        except Exception as e:
            return runtime._internal_report_error_response("IFC walls error", e)

    @blueprint.route('/energy/ifc_simulate', methods=['POST'])
    @runtime.login_required
    def ifc_simulate():
        """使用 IFC 提取的数据直接运行能耗模拟"""
        data = runtime.request.get_json(silent=True) or {}
        report_number = data.get('report_number', 'default')
        try:
            requested_owner = runtime._requested_owner_username(data)
            runtime._resolved_energy_report_context(report_number, requested_owner)
        except runtime._REPORT_STORAGE_EXCEPTIONS as e:
            return runtime._report_storage_error_response(e)
        if not runtime.HAS_IFC or not callable(runtime.run_simple_model):
            return runtime.jsonify({'error': 'IFC support not available'}), 501

        try:
            context = runtime._energy_report_context(report_number, requested_owner=requested_owner)
            ifc_path = runtime.resolve_report_child_file(
                context.report_dir, 'building_model.ifc', required=False,
            )

            if not runtime.os.path.exists(ifc_path):
                return runtime.jsonify({'error': 'IFC file not found'}), 404

            parser = runtime.IFCParser(ifc_path)
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

            result = runtime.run_simple_model(case_data)
            result['ifc_data'] = sim_data

            return runtime.jsonify(result)
        except runtime._REPORT_STORAGE_EXCEPTIONS as e:
            return runtime._report_storage_error_response(e)
        except Exception as e:
            return runtime._internal_report_error_response("IFC simulation error", e)

    app.register_blueprint(blueprint)
    return {'landing': landing, 'index': index, 'home': home, 'floorplan': floorplan, 'health': health, 'energy_library_walls': energy_library_walls, 'energy_library_materials': energy_library_materials, 'energy_library_envelope': energy_library_envelope, 'energy_upload': energy_upload, 'get_all_layers_geometry': get_all_layers_geometry, 'energy_geometry': energy_geometry, 'energy_geometry_advanced': energy_geometry_advanced, 'energy_calculate': energy_calculate, 'energy_status': energy_status, 'upload_ifc': upload_ifc, 'ifc_properties': ifc_properties, 'ifc_walls': ifc_walls, 'ifc_simulate': ifc_simulate}
