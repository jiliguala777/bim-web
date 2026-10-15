"""Recognition HTTP routes with explicit runtime dependencies."""
from flask import Blueprint


def register(app, runtime):
    blueprint = Blueprint("recognition", __name__)

    @blueprint.route('/energy/ai_recognize', methods=['POST'])
    @runtime.login_required
    def ai_recognize():
        """
        AI 识别建筑平面图中的墙体、窗户、门
        支持 raster_file (PNG/JPG) 上传
        与 DXF 上传同步：用户可选择 DXF 矢量 或 图片AI识别
        """
        model_backend = runtime.request.form.get('model_backend', runtime.LEGACY_ONNX_BACKEND)
        availability = {
            runtime.LEGACY_ONNX_BACKEND: runtime.HAS_FLOORPLAN_AI,
            runtime.VECTOR_PYTORCH_BACKEND: runtime.HAS_VECTOR_FLOORPLAN_AI,
            runtime.IMAGE_ONNX_BACKEND: runtime.HAS_IMAGE_FLOORPLAN_AI,
        }
        if model_backend not in availability:
            return runtime.jsonify({'error': 'Unsupported model backend'}), 400
        if not availability[model_backend]:
            return runtime.jsonify({'error': 'Recognition model is not available; see docs/models.md'}), 501
        try:
            context = runtime._resolved_energy_report_context(
                runtime.request.form.get('report_number', 'default'),
                runtime._requested_owner_username(runtime.request.form),
                create=True,
            )
        except runtime._REPORT_STORAGE_EXCEPTIONS as exc:
            return runtime._report_storage_error_response(exc)

        runtime._upsert_report_status(
            context.owner_username, context.report_number, 'recognizing',
        )

        @runtime.after_this_request
        def update_recognition_status(response):
            status = 'recognized' if 200 <= response.status_code < 300 else 'failed'
            runtime._update_report_status_if_exists(
                context.owner_username, context.report_number, status,
            )
            return response

        try:
            t0 = runtime._time.time()
            report_number = context.report_number
            target_dir = context.report_dir

            preprocessing = runtime.request.form.get('preprocessing', 'auto')
            use_preprocessing = preprocessing != 'none'

            try:
                region_request = runtime.parse_crop_region_request(
                    runtime.request.form.get('recognition_mode', 'full_page'),
                    runtime.request.form.get('crop_bbox_px'),
                    runtime.request.form.get('crop_preview_size'),
                )
            except ValueError as exc:
                return runtime.jsonify({'error': str(exc)}), 400

            pdf_upload_token = runtime.request.form.get('pdf_upload_token', '').strip()
            if (
                region_request['mode'] == 'crop_region'
                and not pdf_upload_token
                and model_backend not in {runtime.VECTOR_PYTORCH_BACKEND, runtime.IMAGE_ONNX_BACKEND}
            ):
                return runtime.jsonify({
                    'error': 'crop_region requires a prepared PDF upload or the vector model',
                }), 400
            pdf_page_number = None
            pdf_page_count = None

            if pdf_upload_token:
                try:
                    prepared_pdf_path, pdf_page_number, pdf_page_count = runtime._validated_prepared_pdf(
                        context.owner_username,
                        report_number,
                        pdf_upload_token,
                        runtime.request.form.get('pdf_page_number', '1'),
                    )
                except runtime._REPORT_STORAGE_EXCEPTIONS as exc:
                    return runtime._report_storage_error_response(exc)
                except ValueError as exc:
                    return runtime.jsonify({'error': str(exc)}), 400
                raster_path = runtime.os.fspath(prepared_pdf_path)
                ext = 'pdf'
            else:
                if 'raster_file' not in runtime.request.files:
                    return runtime.jsonify({'error': 'No image file provided'}), 400

                f = runtime.request.files['raster_file']
                if not f or not f.filename:
                    return runtime.jsonify({'error': 'Empty file'}), 400

                ext = f.filename.rsplit('.', 1)[1].lower() if '.' in f.filename else 'png'
                raster_path = runtime.resolve_report_child_destination(
                    target_dir, f'building_plan_ai.{ext}',
                )
                runtime._atomic_save_upload(f, raster_path)
                raster_path = runtime.resolve_report_child_file(
                    target_dir, f'building_plan_ai.{ext}',
                )
                if ext == 'pdf':
                    pdf_page_number = 1

            if model_backend == runtime.VECTOR_PYTORCH_BACKEND:
                if ext == 'pdf':
                    return runtime.jsonify({'error': 'Vector recognition currently accepts PNG or JPG; render the target PDF page first.'}), 400
                crop_bbox_image_px = None
                if region_request['mode'] == 'crop_region':
                    source_bgr = runtime.cv2.imread(str(raster_path))
                    if source_bgr is None:
                        return runtime.jsonify({'error': 'Cannot decode image'}), 400
                    image_height, image_width = source_bgr.shape[:2]
                    crop_bbox_image_px = runtime.map_crop_bbox_to_page(
                        region_request['crop_bbox_px'],
                        region_request['preview_size'],
                        [image_width, image_height],
                    )
                    left, top, right, bottom = crop_bbox_image_px
                    raster_path = runtime.resolve_report_child_destination(
                        target_dir, 'building_plan_ai_crop.png',
                    )
                    runtime._write_image(runtime.Path(raster_path), source_bgr[top:bottom, left:right].copy())
                result = runtime._vector_platform_adapter.predict(runtime.Path(raster_path), runtime.Path(target_dir))
                result.update({
                    'recognition_mode': region_request['mode'],
                    'crop_bbox_page_px': crop_bbox_image_px,
                    'crop_bbox_preview_px': region_request['crop_bbox_px'],
                    'crop_preview_size': region_request['preview_size'],
                })
                return runtime._persist_vector_recognition(
                    target_dir, raster_path, result, round(runtime._time.time() - t0, 3),
                    context.owner_username,
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
                    for artifact_name in (
                        'page_render.png',
                        'cleaned_page.png',
                        'model_view.png',
                        'model_input_512.png',
                        'preprocessing.json',
                    ):
                        runtime.resolve_report_child_file(
                            target_dir, artifact_name, required=False,
                        )
                    prepared_page = runtime.prepare_pdf_page(
                        raster_path,
                        page_number=pdf_page_number,
                        poppler_path=runtime._resolve_poppler_path(),
                        segmenter=(
                            runtime._floorplan_image_segmenter
                            if model_backend == runtime.IMAGE_ONNX_BACKEND else runtime._floorplan_segmenter
                        ),
                        output_dir=target_dir,
                    )
                    pdf_page_count = prepared_page.page_count
                    scale_calibration = prepared_page.scale_calibration
                    vector_cleanup = dict(prepared_page.vector_cleanup)
                    png_path = runtime.resolve_report_child_destination(
                        target_dir, 'building_plan_ai.png',
                    )
                    raster_path = png_path
                except runtime._REPORT_STORAGE_EXCEPTIONS as exc:
                    return runtime._report_storage_error_response(exc)
                except Exception as exc:
                    return runtime._internal_report_error_response('PDF conversion failed', exc)

            # AI 推理
            img_bgr = (
                prepared_page.cleaned_bgr.copy()
                if prepared_page is not None
                    else runtime.cv2.imread(str(raster_path))
            )
            if img_bgr is None:
                return runtime.jsonify({'error': 'Cannot decode image'}), 400

            original_bgr = (
                prepared_page.render_bgr.copy()
                if prepared_page is not None
                else img_bgr.copy()
            )
            combined_cleanup_mask = (
                prepared_page.cleanup_mask.copy()
                if prepared_page is not None
                else runtime.np.zeros(img_bgr.shape[:2], dtype=runtime.np.uint8)
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
                crop_bbox_page_px = runtime.map_crop_bbox_to_page(
                    region_request['crop_bbox_px'],
                    region_request['preview_size'],
                    [page_width, page_height],
                )
                cropped_inputs = runtime.crop_page_inputs(
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
                runtime._write_image(runtime.Path(raster_path), original_bgr)

            has_vector_geometry = bool(
                prepared_page is not None
                and prepared_page.vector_cleanup.get('has_vector_geometry')
            )

            if (
                prepared_page is not None
                and (
                    runtime.np.any(combined_cleanup_mask)
                    or vector_cleanup.get('nonstructural_mask', {}).get('enabled')
                )
            ):
                runtime._write_image(
                    runtime.resolve_report_child_destination(target_dir, 'pdf_nonstructural_mask.png'),
                    combined_cleanup_mask,
                )

            if prepared_page is not None and has_vector_geometry:
                if (
                    structural_support_mask is not None
                    and vector_cleanup.get('structural_mask', {}).get('enabled')
                ):
                    runtime._write_image(
                        runtime.resolve_report_child_destination(target_dir, 'pdf_structural_mask.png'),
                        structural_support_mask,
                    )
                building_roi = vector_cleanup.get('building_roi') or {
                    'enabled': False,
                    'bbox_px': None,
                }
                roi_path = runtime.resolve_report_child_destination(
                    target_dir, 'pdf_building_roi.json',
                )
                runtime._atomic_write_json(roi_path, building_roi)

            if prepared_page is not None and has_vector_geometry:
                runtime._write_image(
                    runtime.resolve_report_child_destination(target_dir, 'pdf_model_input.png'),
                    img_bgr,
                )

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
            segmenter = (
                runtime._floorplan_image_segmenter
                if model_backend == runtime.IMAGE_ONNX_BACKEND else runtime._floorplan_segmenter
            )
            result = segmenter.predict(
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
            if model_backend == runtime.IMAGE_ONNX_BACKEND:
                result['model'] = {
                    'backend': runtime.IMAGE_ONNX_BACKEND,
                    'name': runtime.FLOORPLAN_IMAGE_MODEL_NAME,
                    'version': runtime.FLOORPLAN_IMAGE_MODEL_VERSION,
                    'path': runtime.IMAGE_ONNX_MODEL_PATH,
                    'classes': ['background', 'wall', 'window', 'door'],
                }

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
                room_topology = runtime.apply_scale_to_room_topology(
                    room_topology,
                    scale_calibration.get('scale_m_per_px'),
                )
            room_topology = runtime._enforce_topology_repair_readiness(
                room_topology,
                result.get('topology_repair'),
            )
            result['room_topology'] = room_topology
            h, w = mask.shape

            # 保存结果
            overlay_path = runtime.resolve_report_child_destination(target_dir, 'ai_overlay.jpg')
            mask_path = runtime.resolve_report_child_destination(target_dir, 'ai_mask.png')
            footprint_path = runtime.resolve_report_child_destination(target_dir, 'ai_footprint_mask.png')
            runtime._write_image(
                runtime.Path(overlay_path),
                overlay,
                [runtime.cv2.IMWRITE_JPEG_QUALITY, 90],
            )

            raw_model_mask = result.get('raw_model_mask')
            if raw_model_mask is not None:
                raw_mask_color = runtime.np.zeros((h, w, 3), dtype=runtime.np.uint8)
                raw_mask_color[raw_model_mask == 1] = (60, 76, 231)
                raw_mask_color[raw_model_mask == 2] = (219, 152, 52)
                raw_mask_color[raw_model_mask == 3] = (113, 204, 46)
                runtime._write_image(
                    runtime.resolve_report_child_destination(target_dir, 'ai_raw_model_mask.png'),
                    raw_mask_color,
                )

            # mask 转彩色保存
            mask_color = runtime.np.zeros((h, w, 3), dtype=runtime.np.uint8)
            mask_color[mask == 1] = (60, 76, 231)   # wall - red
            mask_color[mask == 2] = (219, 152, 52)   # window - blue
            mask_color[mask == 3] = (113, 204, 46)   # door - green
            runtime._write_image(runtime.Path(mask_path), mask_color)
            if result.get('footprint_mask') is not None:
                runtime._write_image(runtime.Path(footprint_path), result['footprint_mask'] * 255)

            # 编码为base64用于前端展示
            _, overlay_buf = runtime.cv2.imencode('.jpg', overlay, [runtime.cv2.IMWRITE_JPEG_QUALITY, 85])
            overlay_b64 = runtime.base64.b64encode(overlay_buf).decode('utf-8')

            _, mask_buf = runtime.cv2.imencode('.png', mask_color)
            mask_b64 = runtime.base64.b64encode(mask_buf).decode('utf-8')

            _, orig_buf = runtime.cv2.imencode('.jpg', original_bgr, [runtime.cv2.IMWRITE_JPEG_QUALITY, 85])
            orig_b64 = runtime.base64.b64encode(orig_buf).decode('utf-8')

            elapsed = round(runtime._time.time() - t0, 3)

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

            recognition_payload = runtime._build_recognition_payload(
                result,
                preprocessing,
                use_preprocessing,
                raster_path,
                overlay_path,
                mask_path,
                elapsed,
                context.owner_username,
            )
            if result.get('footprint_mask') is not None:
                recognition_payload['footprint'] = {
                    'path': runtime.os.path.basename(footprint_path),
                    'pixels': int(result['footprint_mask'].sum()),
                    'color': '#b450b4',
                }
            recognition_path = runtime._save_recognition_payload(target_dir, recognition_payload)

            return runtime.jsonify({
                'success': True,
                'status': 'success',
                'source': 'AI',
                'model': result.get('model', {}).get('name', runtime.FLOORPLAN_MODEL_NAME),
                'model_info': {
                    'backend': result.get('model', {}).get('backend', runtime.LEGACY_ONNX_BACKEND),
                    'name': result.get('model', {}).get('name', runtime.FLOORPLAN_MODEL_NAME),
                    'version': result.get('model', {}).get('version', runtime.FLOORPLAN_MODEL_VERSION),
                    'mIoU': result.get('model', {}).get('mIoU'),
                },
                'mIoU': result.get('model', {}).get('mIoU'),
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
                'footprint': recognition_payload.get('footprint'),
                'exterior_elements': recognition_payload.get('exterior_elements'),
                'pixel_lengths': {
                    'wall_px': round(wall_total_length, 1),
                    'window_px': round(win_total_length, 1),
                },
                'recognition': {
                    'schema_version': recognition_payload['schema_version'],
                    'path': runtime.os.path.basename(recognition_path),
                    'model_version': runtime.FLOORPLAN_MODEL_VERSION,
                    'preprocessing': recognition_payload['preprocessing'],
                },
                'images': {
                    'original': orig_b64,
                    'overlay': overlay_b64,
                    'mask': mask_b64,
                },
            })

        except runtime._REPORT_STORAGE_EXCEPTIONS as exc:
            return runtime._report_storage_error_response(exc)
        except Exception as exc:
            return runtime._internal_report_error_response('AI recognize error', exc)

    @blueprint.route('/energy/ai_status')
    def ai_status():
        """检查 AI 识别模块状态"""
        legacy_model = runtime._public_ai_model_info({
            'backend': runtime.LEGACY_ONNX_BACKEND,
            'name': runtime.FLOORPLAN_MODEL_NAME,
            'version': runtime.FLOORPLAN_MODEL_VERSION,
            'mIoU': runtime.FLOORPLAN_MODEL_MIOU,
            'classes': ['background', 'wall', 'window', 'door'],
        })
        vector_model = runtime._public_ai_model_info({
            'backend': runtime.VECTOR_PYTORCH_BACKEND,
            'name': 'vector-resnet34-unet',
            'version': (
                runtime._vector_platform_config.checkpoint_path.name
                if runtime.HAS_VECTOR_FLOORPLAN_AI else None
            ),
        })
        image_model = runtime._public_ai_model_info({
            'backend': runtime.IMAGE_ONNX_BACKEND,
            'name': runtime.FLOORPLAN_IMAGE_MODEL_NAME,
            'version': runtime.FLOORPLAN_IMAGE_MODEL_VERSION if runtime.HAS_IMAGE_FLOORPLAN_AI else None,
            'classes': ['background', 'wall', 'window', 'door'],
        })
        backends = {
            runtime.LEGACY_ONNX_BACKEND: {
                'available': runtime.HAS_FLOORPLAN_AI,
                'model': legacy_model.get('name') if runtime.HAS_FLOORPLAN_AI else None,
                'model_version': legacy_model.get('version') if runtime.HAS_FLOORPLAN_AI else None,
            },
            runtime.VECTOR_PYTORCH_BACKEND: {
                'available': runtime.HAS_VECTOR_FLOORPLAN_AI,
                'model': vector_model.get('name') if runtime.HAS_VECTOR_FLOORPLAN_AI else None,
                'model_version': vector_model.get('version') if runtime.HAS_VECTOR_FLOORPLAN_AI else None,
            },
            runtime.IMAGE_ONNX_BACKEND: {
                'available': runtime.HAS_IMAGE_FLOORPLAN_AI,
                'model': image_model.get('name') if runtime.HAS_IMAGE_FLOORPLAN_AI else None,
                'model_version': image_model.get('version') if runtime.HAS_IMAGE_FLOORPLAN_AI else None,
            },
        }
        active_model = (
            legacy_model if runtime.HAS_FLOORPLAN_AI
            else (vector_model if runtime.HAS_VECTOR_FLOORPLAN_AI else (image_model if runtime.HAS_IMAGE_FLOORPLAN_AI else {}))
        )
        return runtime.jsonify({
            'available': runtime.HAS_FLOORPLAN_AI or runtime.HAS_IMAGE_FLOORPLAN_AI or runtime.HAS_VECTOR_FLOORPLAN_AI,
            'model': active_model.get('name') if active_model else None,
            'model_version': active_model.get('version') if active_model else None,
            'mIoU': runtime.FLOORPLAN_MODEL_MIOU if runtime.HAS_FLOORPLAN_AI else None,
            'classes': ['background', 'wall', 'window', 'door'] if runtime.HAS_FLOORPLAN_AI else [],
            'backends': backends,
        })

    @blueprint.route('/energy/scale_calibration', methods=['POST'])
    @runtime.login_required
    def save_scale_calibration():
        """Persist a manual two-point scale when automatic PDF evidence is unavailable."""
        try:
            data = runtime.request.get_json() or {}
            context = runtime._resolved_energy_report_context(
                data.get('report_number'),
                runtime._requested_owner_username(data),
            )
            target_dir = context.report_dir
            recognition = runtime._load_recognition_payload(target_dir)
            if recognition is None:
                return runtime.jsonify({'error': 'No current recognition.json found. Run AI recognition first.'}), 404

            image_size = recognition.get('image_size') or [0, 0]
            width, height = float(image_size[0] or 0), float(image_size[1] or 0)

            def parse_point(key):
                point = data.get(key)
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError(f'{key} must contain x and y')
                x, y = float(point[0]), float(point[1])
                if not runtime.math.isfinite(x) or not runtime.math.isfinite(y):
                    raise ValueError(f'{key} must contain finite coordinates')
                if x < 0 or y < 0 or x >= width or y >= height:
                    raise ValueError(f'{key} is outside the recognized image')
                return [x, y]

            point_a = parse_point('point_a')
            point_b = parse_point('point_b')
            pixel_distance = runtime.math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])
            if pixel_distance <= 0:
                raise ValueError('calibration points must be different')

            actual_length = float(data.get('actual_length'))
            if not runtime.math.isfinite(actual_length) or actual_length <= 0:
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
            recognition['room_topology'] = runtime.apply_scale_to_room_topology(
                recognition.get('room_topology') or {},
                scale_m_per_px,
            )
            recognition['room_topology'] = runtime._enforce_topology_repair_readiness(
                recognition['room_topology'],
                recognition.get('topology_repair'),
            )
            runtime._save_recognition_payload(target_dir, recognition)
            return runtime.jsonify({
                'success': True,
                'scale_calibration': calibration,
                'room_topology': recognition['room_topology'],
            })
        except runtime._REPORT_STORAGE_EXCEPTIONS as exc:
            return runtime._report_storage_error_response(exc)
        except (TypeError, ValueError) as exc:
            return runtime.jsonify({'error': str(exc)}), 400
        except Exception as exc:
            return runtime._internal_report_error_response('Scale calibration error', exc)

    @blueprint.route('/energy/ai_simulate', methods=['POST'])
    @runtime.login_required
    def ai_simulate():
        """
        使用 AI 识别结果结合详细参数进行能耗计算
        """
        exterior_lock_context = None
        exterior_lock_acquired = False
        try:
            data = runtime.request.get_json() or {}
            context = runtime._resolved_energy_report_context(
                data.get('report_number', 'default'),
                runtime._requested_owner_username(data),
            )
            report_number = context.report_number
            target_dir = context.report_dir
            if not (runtime.HAS_FLOORPLAN_AI or runtime.HAS_IMAGE_FLOORPLAN_AI or runtime.HAS_VECTOR_FLOORPLAN_AI):
                return runtime.jsonify({'error': 'AI module not available'}), 501
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
                if not runtime.math.isfinite(value):
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

            # 查找已有的 AI 识别结果图
            recognition = runtime._load_recognition_payload(target_dir)
            if recognition is None:
                return runtime.jsonify({
                    'error': 'No current recognition.json found. Run AI recognition first.'
                }), 404

            initially_confirmed_exterior = recognition.get('exterior_topology') or {}
            if (
                initially_confirmed_exterior.get('confirmed') is True
                and initially_confirmed_exterior.get('load_geometry_ready') is True
            ):
                exterior_lock_context = runtime._exterior_report_lock(runtime.Path(target_dir))
                exterior_lock_context.__enter__()
                exterior_lock_acquired = True
                recognition = runtime._load_recognition_payload(target_dir)
                if recognition is None:
                    raise runtime.ExteriorGenerationConflict(
                        'Recognition generation disappeared while acquiring the exterior lock'
                    )

            scale_source = 'request'
            scale_calibration = recognition.get('scale_calibration') or {}
            if scale_calibration.get('status') == 'confirmed':
                calibrated_scale = float(scale_calibration.get('scale_m_per_px') or 0.0)
                if runtime.math.isfinite(calibrated_scale) and calibrated_scale > 0:
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
            footprint = recognition.get('footprint') or {}
            footprint_pixels = float(footprint.get('pixels') or 0.0)
            if footprint_pixels > 0:
                floor_area_m2 = footprint_pixels * (scale ** 2)
                floor_area_source = 'image_footprint'
            elif exterior.get('confirmed') is True and exterior.get('load_geometry_ready') is True:
                try:
                    runtime._require_current_exterior_generation(
                        runtime.Path(target_dir), report_number, recognition=recognition,
                    )
                except runtime.ExteriorGenerationConflict:
                    if (
                        not exterior_repair_required
                        and room_topology.get('load_geometry_ready') is True
                        and room_count > 0
                        and runtime.math.isfinite(room_area_px2)
                        and room_area_px2 > 0
                    ):
                        exterior = {}
                    else:
                        raise

            if footprint_pixels > 0:
                pass
            elif exterior.get('confirmed') is True and exterior.get('load_geometry_ready') is True:
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
                and runtime.math.isfinite(room_area_px2)
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
            if runtime.HAS_ENERGY_CALC:
                res = runtime.energy_calc.calculate_energy(calc_params)
                if runtime.HAS_DESIGN_LOAD_CALC:
                    res["design_loads"] = runtime.design_load_calc.calculate_design_loads(calc_params)
                res["recognition_source"] = "recognition_json"
                res["recognition_model_version"] = recognition.get("model", {}).get("version")
                res["recognition_preprocessing"] = recognition.get("preprocessing", {})
                res["floor_area_source"] = floor_area_source
                res["scale_source"] = scale_source

                # 将计算结果、输入参数、几何信息持久化到 SQLite 数据库中
                conn = runtime.get_db_connection()
                try:
                    conn.execute('''
                        INSERT INTO reports (
                            username, report_number, status, updated_at,
                            geometry_used, params, results
                        )
                        VALUES (?, ?, 'calculated', datetime('now'), ?, ?, ?)
                        ON CONFLICT(username, report_number) DO UPDATE SET
                            status = 'calculated',
                            updated_at = datetime('now'),
                            geometry_used = excluded.geometry_used,
                            params = excluded.params,
                            results = excluded.results
                    ''', (
                        context.owner_username,
                        report_number,
                        runtime.json.dumps(calc_params.get('geometry', {})),
                        runtime.json.dumps(calc_params),
                        runtime.json.dumps(res),
                    ))
                    conn.commit()
                finally:
                    conn.close()

                return runtime.jsonify(res)
            else:
                return runtime.jsonify({'error': 'Energy calculation engine missing on server'}), 500

        except runtime._REPORT_STORAGE_EXCEPTIONS as exc:
            return runtime._report_storage_error_response(exc)
        except runtime.ExteriorGenerationConflict as exc:
            return runtime.jsonify({'error': str(exc)}), 409
        except ValueError as e:
            return runtime.jsonify({'error': str(e)}), 400
        except Exception as exc:
            return runtime._internal_report_error_response('AI simulate error', exc)
        finally:
            if exterior_lock_acquired:
                exterior_lock_context.__exit__(None, None, None)

    app.register_blueprint(blueprint)
    return {'ai_recognize': ai_recognize, 'ai_status': ai_status, 'save_scale_calibration': save_scale_calibration, 'ai_simulate': ai_simulate}
