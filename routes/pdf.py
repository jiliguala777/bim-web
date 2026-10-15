"""Pdf HTTP routes with explicit runtime dependencies."""
from flask import Blueprint


def register(app, runtime):
    blueprint = Blueprint("pdf", __name__)

    @blueprint.route('/energy/pdf_prepare', methods=['POST'])
    @runtime.login_required
    def prepare_energy_pdf():
        """Persist one PDF upload and return its page count for page selection."""
        report_number = runtime.request.form.get('report_number', 'default')
        try:
            context = runtime._resolved_energy_report_context(
                report_number,
                runtime._requested_owner_username(runtime.request.form),
                create=True,
            )
        except runtime._REPORT_STORAGE_EXCEPTIONS as exc:
            return runtime._report_storage_error_response(exc)
        runtime._upsert_report_status(context.owner_username, context.report_number, 'created')
        uploaded = runtime.request.files.get('raster_file')
        if not uploaded or not uploaded.filename:
            return runtime.jsonify({'error': 'No PDF file provided'}), 400

        original_filename = runtime.secure_filename(uploaded.filename)
        extension = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''
        if extension != 'pdf':
            return runtime.jsonify({'error': 'Only PDF files can be prepared'}), 400

        target_dir = context.report_dir
        stored_filename = f'building_plan_prepared_{runtime.uuid.uuid4().hex}.pdf'
        stored_path = runtime.resolve_report_child_destination(target_dir, stored_filename)

        def discard_partial_upload():
            try:
                stored_path.unlink(missing_ok=True)
            except OSError:
                runtime.logger.error('Cannot remove partial prepared PDF', exc_info=True)

        try:
            runtime._atomic_save_upload(uploaded, stored_path)
        except Exception as exc:
            discard_partial_upload()
            runtime._update_report_status_if_exists(
                context.owner_username, context.report_number, 'failed',
            )
            return runtime._internal_report_error_response('Cannot save prepared PDF', exc)

        try:
            import pdfplumber

            with pdfplumber.open(stored_path) as pdf:
                page_count = len(pdf.pages)
            if page_count < 1:
                raise ValueError('PDF contains no pages')
        except Exception as exc:
            discard_partial_upload()
            runtime._update_report_status_if_exists(
                context.owner_username, context.report_number, 'failed',
            )
            runtime.logger.warning('Cannot read prepared PDF: %s', exc)
            return runtime.jsonify({'error': 'Cannot read PDF'}), 400

        runtime._upsert_report_status(context.owner_username, context.report_number, 'uploaded')

        return runtime.jsonify({
            'success': True,
            'upload_token': runtime._make_pdf_upload_token(
                context.owner_username,
                context.report_number,
                stored_filename,
                page_count,
            ),
            'page_count': page_count,
            'filename': original_filename,
        })

    @blueprint.route('/energy/pdf_page_preview', methods=['POST'])
    @runtime.login_required
    def preview_energy_pdf_page():
        """Render a selected prepared-PDF page for client-side region selection."""
        try:
            context = runtime._resolved_energy_report_context(
                runtime.request.form.get('report_number', 'default'),
                runtime._requested_owner_username(runtime.request.form),
            )
            pdf_path, page_number, page_count = runtime._validated_prepared_pdf(
                context.owner_username,
                context.report_number,
                runtime.request.form.get('pdf_upload_token', '').strip(),
                runtime.request.form.get('pdf_page_number', '1'),
            )
        except runtime._REPORT_STORAGE_EXCEPTIONS as exc:
            return runtime._report_storage_error_response(exc)
        except ValueError as exc:
            return runtime.jsonify({'error': str(exc)}), 400

        try:
            preview_bgr = runtime.render_pdf_page_preview(
                pdf_path,
                page_number,
                page_count,
                runtime._resolve_poppler_path(),
            )
            encoded, jpeg = runtime.cv2.imencode(
                '.jpg',
                preview_bgr,
                [runtime.cv2.IMWRITE_JPEG_QUALITY, 85],
            )
            if not encoded:
                raise ValueError('Cannot encode PDF preview')
        except Exception as exc:
            return runtime._internal_report_error_response('PDF preview failed', exc)

        height, width = preview_bgr.shape[:2]
        return runtime.jsonify({
            'success': True,
            'pdf_page_number': page_number,
            'pdf_page_count': page_count,
            'image_size': [width, height],
            'image': runtime.base64.b64encode(jpeg).decode('utf-8'),
        })

    @blueprint.route('/energy/crop_region.js', methods=['GET'])
    @runtime.login_required
    def energy_crop_region_helper():
        """Serve the energy crop-selection helper."""
        return runtime.send_from_directory(
            runtime.os.path.join(runtime.BASE_DIR, 'static', 'energy'),
            'crop_region.js',
            mimetype='application/javascript',
        )

    @blueprint.route('/energy/pdf_recognition_state.js', methods=['GET'])
    @runtime.login_required
    def energy_pdf_recognition_state_helper():
        """Serve the PDF recognition request-state helper."""
        return runtime.send_from_directory(
            runtime.os.path.join(runtime.BASE_DIR, 'static', 'energy'),
            'pdf_recognition_state.js',
            mimetype='application/javascript',
        )

    @blueprint.route('/energy/vector_pdf_fusion', methods=['POST'])
    @runtime.login_required
    @runtime._serialized_exterior_report
    def vector_pdf_fusion():
        """Publish isolated native-PDF/model diagnostics without energy geometry."""
        try:
            context = runtime._energy_report_context(
                runtime.request.form.get('report_number', 'default'),
                runtime._requested_owner_username(runtime.request.form),
            )
        except runtime._REPORT_STORAGE_EXCEPTIONS as exc:
            return runtime._report_storage_error_response(exc)
        runtime._upsert_report_status(
            context.owner_username, context.report_number, 'recognizing',
        )

        @runtime.after_this_request
        def update_vector_pdf_fusion_status(response):
            status = 'recognized' if 200 <= response.status_code < 300 else 'failed'
            runtime._update_report_status_if_exists(
                context.owner_username, context.report_number, status,
            )
            return response

        if not runtime.HAS_VECTOR_PDF_FUSION or runtime._vector_pdf_fusion_config is None:
            return runtime.jsonify({'error': 'Vector PDF fusion module is not configured'}), 501
        report_number = context.report_number
        try:
            pdf_path, page_number, page_count = runtime._validated_prepared_pdf(
                context.owner_username,
                report_number,
                runtime.request.form.get('pdf_upload_token', '').strip(),
                runtime.request.form.get('pdf_page_number', '1'),
            )
            region_request = runtime.parse_crop_region_request(
                runtime.request.form.get('recognition_mode', 'full_page'),
                runtime.request.form.get('crop_bbox_px'),
                runtime.request.form.get('crop_preview_size'),
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
                crop_bbox_page_px = runtime.map_crop_bbox_to_page(
                    region_request['crop_bbox_px'],
                    region_request['preview_size'],
                    render_size,
                )
        except runtime._REPORT_STORAGE_EXCEPTIONS as exc:
            return runtime._report_storage_error_response(exc)
        except ValueError as exc:
            return runtime.jsonify({'error': str(exc)}), 400

        report_dir = context.report_dir
        try:
            target_dir = runtime._validated_report_child_directory(
                report_dir,
                report_dir / 'vector_pdf_fusion',
                create=True,
            )
        except runtime._REPORT_STORAGE_EXCEPTIONS as exc:
            return runtime._report_storage_error_response(exc)
        generation_marker = runtime._new_exterior_generation(report_number)
        runtime._write_exterior_generation(report_dir, generation_marker)
        try:
            result = runtime.analyze_vector_pdf_page(
                pdf_path,
                page_number,
                target_dir,
                runtime._vector_pdf_fusion_config,
                crop_bbox_page_px=crop_bbox_page_px,
            )
            original_path = runtime.resolve_report_child_file(
                target_dir, 'pdf_vector_model_input.png', required=False,
            )
            overlay_path = runtime.resolve_report_child_file(
                target_dir, 'pdf_vector_fusion_overlay.png', required=False,
            )
            exterior_overlay_path = runtime.resolve_report_child_file(
                target_dir, 'pdf_exterior_overlay.png', required=False,
            )
            component_overlay_path = runtime.resolve_report_child_file(
                target_dir, 'pdf_component_overlay.png', required=False,
            )
            original = runtime.cv2.imread(str(original_path), runtime.cv2.IMREAD_COLOR)
            overlay = runtime.cv2.imread(str(overlay_path), runtime.cv2.IMREAD_COLOR)
            exterior_overlay = runtime.cv2.imread(str(exterior_overlay_path), runtime.cv2.IMREAD_COLOR)
            component_overlay = runtime.cv2.imread(str(component_overlay_path), runtime.cv2.IMREAD_COLOR)
            images = {}
            if original is not None:
                encoded, buffer = runtime.cv2.imencode('.png', original)
                if encoded:
                    images['original'] = runtime.base64.b64encode(buffer).decode('utf-8')
            if overlay is not None:
                encoded, buffer = runtime.cv2.imencode('.png', overlay)
                if encoded:
                    images['overlay'] = runtime.base64.b64encode(buffer).decode('utf-8')
            if exterior_overlay is not None:
                encoded, buffer = runtime.cv2.imencode('.png', exterior_overlay)
                if encoded:
                    images['exterior_overlay'] = runtime.base64.b64encode(buffer).decode('utf-8')
            if component_overlay is not None:
                encoded, buffer = runtime.cv2.imencode('.png', component_overlay)
                if encoded:
                    images['component_overlay'] = runtime.base64.b64encode(buffer).decode('utf-8')
            topology_path = runtime.resolve_report_child_file(
                target_dir, 'pdf_exterior_topology.json', required=False,
            )
            topology_sha256 = runtime._sha256_file(topology_path)
        except ValueError as exc:
            runtime._update_exterior_generation_if_current(
                report_dir, generation_marker['generation'], status='failed',
            )
            return runtime.jsonify({'error': str(exc)}), 400
        except Exception as exc:
            runtime._update_exterior_generation_if_current(
                report_dir, generation_marker['generation'], status='failed',
            )
            return runtime._internal_report_error_response('Vector PDF fusion failed', exc)

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
            'door_total_width_px': runtime.math.fsum(
                float(opening.get('width_px') or 0.0)
                for opening in accepted_openings
                if opening.get('kind') == 'door'
            ),
            'window_count': sum(
                opening.get('kind') == 'window' for opening in accepted_openings
            ),
            'window_total_width_px': runtime.math.fsum(
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

        runtime._update_exterior_generation_if_current(
            report_dir,
            generation_marker['generation'],
            status='ready',
            topology_sha256=topology_sha256,
        )
        return runtime.jsonify({
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

    @blueprint.route('/energy/vector_pdf_exterior_confirm', methods=['POST'])
    @runtime.login_required
    @runtime._serialized_exterior_report
    def vector_pdf_exterior_confirm():
        """Confirm only the current, hashed server-side exterior artifacts."""
        payload = runtime.request.get_json(silent=True)
        if not isinstance(payload, dict):
            return runtime.jsonify({'error': 'JSON request body is required'}), 400
        if payload.get('confirmed') is not True:
            return runtime.jsonify({'error': 'confirmed must be boolean true'}), 400
        context = None
        try:
            calibration_method = payload.get(
                'calibration_method', 'manual_exterior_confirmation',
            )
            if calibration_method not in {'manual_two_point', 'manual_exterior_confirmation'}:
                raise ValueError('calibration_method is invalid')
            scale = runtime._positive_finite_number(payload.get('scale_m_per_px'), 'scale_m_per_px')
            request_page = payload.get('page_number')
            if isinstance(request_page, bool) or not isinstance(request_page, int) or request_page < 1:
                raise ValueError('page_number must be a positive integer')
            if 'crop_bbox_page_px' not in payload:
                raise ValueError('crop_bbox_page_px is required (null for a full page)')
            request_crop = runtime._normalized_crop_bbox(
                payload.get('crop_bbox_page_px'), 'crop_bbox_page_px',
            )
            supplied_hash = payload.get('topology_sha256')
            if not isinstance(supplied_hash, str) or len(supplied_hash) != 64:
                raise ValueError('topology_sha256 must be a SHA-256 hex digest')
            try:
                int(supplied_hash, 16)
            except ValueError:
                raise ValueError('topology_sha256 must be a SHA-256 hex digest') from None
            context = runtime._energy_report_context(
                payload.get('report_number'),
                runtime._requested_owner_username(payload),
            )
            report_number = context.report_number
            report_dir, artifact_dir = runtime._require_exterior_report_dir(context)
            topology_raw, topology = runtime._read_exterior_artifact(
                artifact_dir, 'pdf_exterior_topology.json',
            )
            opening_raw, opening_artifact = runtime._read_exterior_artifact(
                artifact_dir, 'pdf_opening_candidates.json',
            )
        except runtime._REPORT_STORAGE_EXCEPTIONS as exc:
            return runtime._finalize_exterior_confirmation_response(
                context, runtime._report_storage_error_response(exc),
            )
        except FileNotFoundError as exc:
            return runtime._finalize_exterior_confirmation_response(
                context, (runtime.jsonify({'error': str(exc)}), 404),
            )
        except ValueError as exc:
            return runtime._finalize_exterior_confirmation_response(
                context, (runtime.jsonify({'error': str(exc)}), 400),
            )

        current_hash = runtime.hashlib.sha256(topology_raw).hexdigest()
        if not runtime.hmac.compare_digest(current_hash, supplied_hash.lower()):
            return runtime._finalize_exterior_confirmation_response(
                context,
                (runtime.jsonify({'error': 'Exterior topology has changed; review it again'}), 409),
            )

        try:
            generation_marker = runtime._require_current_exterior_generation(
                report_dir, report_number, topology_sha256=current_hash,
            )
        except runtime._REPORT_STORAGE_EXCEPTIONS as exc:
            return runtime._finalize_exterior_confirmation_response(
                context, runtime._report_storage_error_response(exc),
            )
        except runtime.ExteriorGenerationConflict as exc:
            return runtime._finalize_exterior_confirmation_response(
                context, (runtime.jsonify({'error': str(exc)}), 409),
            )

        opening_hash = topology.get('opening_artifact_sha256')
        if (
            not isinstance(opening_hash, str)
            or len(opening_hash) != 64
            or any(character not in '0123456789abcdef' for character in opening_hash)
            or not runtime.hmac.compare_digest(runtime.hashlib.sha256(opening_raw).hexdigest(), opening_hash)
        ):
            return runtime._finalize_exterior_confirmation_response(
                context,
                (runtime.jsonify({'error': 'Opening artifact has changed; review it again'}), 409),
            )

        try:
            provenance, artifact_page, artifact_crop, image_size = runtime._artifact_provenance(
                topology, opening_artifact,
            )
            if request_page != artifact_page or request_crop != artifact_crop:
                raise RuntimeError('Confirmed page or crop does not match current artifacts')
            (
                polygon_points, area_px2, perimeter_px,
                inferred_edges, inferred_length_px, inferred_perimeter_ratio,
            ) = runtime._validate_confirmable_topology(topology, opening_artifact, scale, image_size)
            openings = runtime._validate_opening_bindings(topology, opening_artifact)

            from vector_pdf_energy_geometry import apply_scale_to_exterior
            normalized_topology = runtime.copy.deepcopy(topology)
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

            walls = runtime._validated_real_exterior_wall_geometry(
                polygon_points, topology, inferred_edges,
            )
            doors = []
            windows = []
            for opening in scaled_openings:
                _, _, _, start, end = runtime._axis_segment(opening, 'opening')
                geometry_opening = runtime.copy.deepcopy(opening)
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
            door_width = runtime.math.fsum(float(opening['width_m']) for opening in doors)
            window_width = runtime.math.fsum(float(opening['width_m']) for opening in windows)
            if not runtime.math.isfinite(door_width) or not runtime.math.isfinite(window_width):
                raise RuntimeError('Opening width totals must be finite')
            geometry = {'walls': walls, 'windows': windows, 'doors': doors}
            recognition = {
                'schema_version': runtime.RECOGNITION_SCHEMA_VERSION,
                'owner_username': context.owner_username,
                'report_number': report_number,
                'exterior_generation': {
                    'generation': generation_marker['generation'],
                    'topology_sha256': current_hash,
                },
                'created_at': runtime.time.strftime('%Y-%m-%dT%H:%M:%SZ', runtime.time.gmtime()),
                'model': {
                    'backend': runtime.VECTOR_PYTORCH_BACKEND,
                    'name': 'vector_pdf_fusion',
                    'version': current_hash,
                },
                'preprocessing': {'requested': 'vector_pdf_fusion', 'use_preprocessing': False},
                'source_image': {'path': 'vector_pdf_fusion/pdf_vector_model_input.png'},
                'source_page': runtime.copy.deepcopy(provenance),
                'image_size': image_size,
                'stats': {},
                'geometry_summary': runtime._geometry_summary(geometry),
                'geometry': geometry,
                'contours': runtime.copy.deepcopy(geometry),
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
                    'wall_px': runtime._polyline_total_length(walls),
                    'window_px': runtime._polyline_total_length(windows),
                },
                'artifacts': {
                    'original': 'vector_pdf_fusion/pdf_vector_model_input.png',
                    'overlay': 'vector_pdf_fusion/pdf_exterior_overlay.png',
                    'exterior_topology': 'vector_pdf_fusion/pdf_exterior_topology.json',
                    'opening_candidates': 'vector_pdf_fusion/pdf_opening_candidates.json',
                    'topology_sha256': current_hash,
                },
            }
            runtime._require_current_exterior_generation(
                report_dir,
                report_number,
                topology_sha256=current_hash,
                expected_generation=generation_marker['generation'],
            )
            runtime._save_recognition_payload(report_dir, recognition)
            runtime._require_current_exterior_generation(
                report_dir,
                report_number,
                recognition=recognition,
                topology_sha256=current_hash,
                expected_generation=generation_marker['generation'],
            )
        except runtime._REPORT_STORAGE_EXCEPTIONS as exc:
            return runtime._finalize_exterior_confirmation_response(
                context, runtime._report_storage_error_response(exc),
            )
        except runtime.ExteriorGenerationConflict as exc:
            return runtime._finalize_exterior_confirmation_response(
                context, (runtime.jsonify({'error': str(exc)}), 409),
            )
        except (RuntimeError, ValueError) as exc:
            return runtime._finalize_exterior_confirmation_response(
                context, (runtime.jsonify({'error': str(exc)}), 409),
            )
        except Exception as exc:
            return runtime._finalize_exterior_confirmation_response(
                context, runtime._internal_report_error_response('Cannot persist exterior recognition', exc),
            )

        return runtime._finalize_exterior_confirmation_response(context, runtime.jsonify({
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
        }))

    app.register_blueprint(blueprint)
    return {'prepare_energy_pdf': prepare_energy_pdf, 'preview_energy_pdf_page': preview_energy_pdf_page, 'energy_crop_region_helper': energy_crop_region_helper, 'energy_pdf_recognition_state_helper': energy_pdf_recognition_state_helper, 'vector_pdf_fusion': vector_pdf_fusion, 'vector_pdf_exterior_confirm': vector_pdf_exterior_confirm}
