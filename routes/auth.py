"""Auth HTTP routes with explicit runtime dependencies."""
from flask import Blueprint


def register(app, runtime):
    blueprint = Blueprint("auth", __name__)

    @blueprint.route('/login', methods=['GET', 'POST'])
    def login():
        if runtime.request.method == 'POST':
            data = runtime.request.get_json(silent=True)
            if not isinstance(data, dict):
                return runtime.jsonify({'status': 'fail', 'message': '请求必须是 JSON 对象'}), 400
            username = data.get('username')
            password = data.get('password')
            if (not isinstance(username, str) or not isinstance(password, str)
                    or not username or not password or len(username) > 128 or len(password) > 1024):
                return runtime.jsonify({'status': 'fail', 'message': '用户名和密码格式无效'}), 400
            if runtime.app.config.get('AUTH_RATE_LIMIT_ENABLED', True):
                retry = runtime.consume_auth_attempt(runtime.DB_PATH, 'login', runtime.request.remote_addr or '', username)
                if retry:
                    return runtime.jsonify({'error': 'Too many authentication attempts'}), 429, {'Retry-After': str(retry)}

            # 特权管理员 (凭证从环境变量读取；未设置则不启用此后备登录)
            admin_user = runtime.os.environ.get('ADMIN_USER', 'admin')
            admin_password = runtime.os.environ.get('ADMIN_PASSWORD')
            if (admin_password and runtime.hmac.compare_digest(username.encode('utf-8'), admin_user.encode('utf-8'))
                    and runtime.hmac.compare_digest(password.encode('utf-8'), admin_password.encode('utf-8'))):
                runtime.session.clear()
                runtime.session['logged_in'] = True
                runtime.session['username'] = username
                runtime.session['is_admin'] = True
                runtime.session.permanent = True
                return runtime.jsonify({'status': 'success'})

            # 数据库查询
            conn = runtime.get_db_connection()
            user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
            conn.close()

            if user and runtime.check_password_hash(user['password_hash'], password):
                runtime.session.clear()
                runtime.session['logged_in'] = True
                runtime.session['username'] = username
                runtime.session['is_admin'] = False
                runtime.session.permanent = True
                return runtime.jsonify({'status': 'success'})

            return runtime.jsonify({'status': 'fail', 'message': '用户名或密码错误'}), 401
        return runtime.render_template('login.html')

    @blueprint.route('/register', methods=['GET', 'POST'])
    def register():
        if runtime.request.method == 'POST':
            data = runtime.request.get_json(silent=True)
            if not isinstance(data, dict):
                return runtime.jsonify({'status': 'fail', 'message': '请求必须是 JSON 对象'}), 400
            username = data.get('username')
            password = data.get('password')

            if (not isinstance(username, str) or not isinstance(password, str)
                    or not username or len(username) > 128 or not runtime.valid_password(password)):
                return runtime.jsonify({'status': 'fail', 'message': '密码需为 8–128 位，包含字母和数字或符号'}), 400
            if runtime.app.config.get('AUTH_RATE_LIMIT_ENABLED', True):
                retry = runtime.consume_auth_attempt(runtime.DB_PATH, 'register', runtime.request.remote_addr or '')
                if retry:
                    return runtime.jsonify({'error': 'Too many registration attempts'}), 429, {'Retry-After': str(retry)}
            try:
                runtime.validate_canonical_username(username)
            except ValueError:
                return runtime.jsonify({'status': 'fail', 'message': '用户名格式无效'}), 400

            hashed_pw = runtime.generate_password_hash(password)
            conn = None
            try:
                conn = runtime.get_db_connection()
                conn.execute('BEGIN IMMEDIATE')
                runtime.ensure_unique_user_storage_keys([
                    *runtime._authentication_principals(conn),
                    username,
                ])
                conn.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, hashed_pw))
                conn.commit()
                return runtime.jsonify({'status': 'success'})
            except (runtime.sqlite3.IntegrityError, runtime.StorageIdentityCollision):
                if conn is not None:
                    conn.rollback()
                return runtime.jsonify({'status': 'fail', 'message': '该用户名已被占用'}), 400
            except Exception:
                if conn is not None:
                    conn.rollback()
                runtime.logger.exception("Registration failed")
                return runtime.jsonify({'status': 'fail', 'message': '注册失败'}), 500
            finally:
                if conn is not None:
                    conn.close()

        return runtime.render_template('register.html')

    @blueprint.route('/logout', methods=['GET', 'POST'])
    def logout():
        if runtime.request.method == 'GET':
            return runtime.render_template('logout.html')
        runtime.session.clear()
        return runtime.redirect(runtime.url_for('auth.login'))

    app.register_blueprint(blueprint)
    return {'login': login, 'register': register, 'logout': logout}
