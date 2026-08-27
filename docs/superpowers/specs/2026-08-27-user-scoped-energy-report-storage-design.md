# 用户隔离的能耗报告存储设计

日期：2026-08-27

状态：已批准，待实施

## 1. 背景

当前平台把所有用户的图纸、识别产物和能耗报告文件写入：

```text
<UPLOAD_FOLDER>/energy/<report_number>/
```

SQLite 历史记录虽然保存 `username`，但部分查询和更新只使用
`report_number`。不同用户使用相同报告编号时，磁盘文件和数据库记录都可能互相
覆盖；部分读取接口也无法证明报告属于当前登录用户。

服务器上的唯一旧报告目录已经由用户删除。本设计不提供旧平面目录兼容和自动
迁移，直接启用新的用户隔离结构。

## 2. 目标

1. 所有能耗上传和识别文件使用以下结构：

   ```text
   /var/lib/bim-web/uploads/energy/<安全用户名目录>/<报告编号>/
   ```

2. 普通用户只能创建、读取、识别、标定、计算和查看自己的报告。
3. 管理员可以显式选择并管理任意用户的报告。
4. 不同用户可以安全使用相同的报告编号。
5. PDF 上传令牌、识别锁、SQLite 记录和磁盘目录使用同一报告所有权。
6. 拒绝路径穿越、符号链接逃逸和客户端伪造所有者。

## 3. 非目标

- 不迁移或读取旧的 `<UPLOAD_FOLDER>/energy/<report_number>/` 平面目录。
- 不增加磁盘配额、自动过期、对象存储或加密存储。
- 不改变模型推理、外轮廓识别和能耗计算算法。
- 不允许普通用户共享报告给其他普通用户。
- 不把用户名目录作为授权依据；授权始终来自登录 session 和服务端数据库。

## 4. 安全用户名目录

新增独立模块 `energy_report_storage.py`。用户名目录键由规范化后的可读部分和稳定
短哈希组成：

```text
<可读用户名>-<SHA-256 前 8 位>
```

算法固定如下：

1. 对原始用户名执行 Unicode NFKC 规范化并去除首尾空白。
2. 哈希输入是规范化后的完整 UTF-8 用户名。
3. 可读部分保留 Unicode 字母、数字、点、下划线和连字符。
4. 其他字符序列替换为单个连字符；去除首尾点和连字符。
5. 可读部分最多保留 48 个 Unicode 字符；若为空则使用 `user`。
6. 拼接小写 SHA-256 十六进制摘要的前 8 位。
7. 最终目录键必须是一个非空路径组件，不能等于 `.` 或 `..`。

示例：

```text
张三-1d841bc0/
alice-2bd806c9/
```

短哈希用于避免不同原始用户名清洗后发生目录冲突。页面和数据库继续显示完整原始
用户名，不把目录键显示成账号名称。

## 5. 身份与管理员权限

登录成功后 session 必须同时保存：

```text
username: 原始用户名
is_admin: 布尔值
```

只有通过环境变量 `ADMIN_USER` / `ADMIN_PASSWORD` 后备管理员登录成功时，
`is_admin` 才为 `true`。数据库普通用户即使用户名与管理员名称相同，也不能获得
管理员权限。普通数据库登录明确写入 `is_admin=false`。退出登录清空整个 session，
避免残留用户名或管理员标记。

普通用户的有效报告所有者永远是 session 中的 `username`。客户端提交
`owner_username` 时：

- 普通用户提交自己的用户名可以接受；
- 普通用户提交其他用户名返回 HTTP 403；
- 管理员可以提交任意已存在的数据库用户名；管理员自己的后备账号不要求存在于
  `users` 表；
- 管理员未提交时默认使用管理员自己的用户名。

管理员访问其他用户时必须显式携带 `owner_username`，不能通过猜测磁盘目录键
访问。服务端每次请求都重新验证 `is_admin`，不信任前端隐藏字段。

## 6. 统一报告上下文

`energy_report_storage.py` 提供以下职责清晰的接口：

```python
def user_storage_key(username: str) -> str: ...

def validate_report_number(report_number: str) -> str: ...

def resolve_report_owner(
    session_username: str,
    is_admin: bool,
    requested_owner: str | None,
) -> str: ...

def resolve_energy_report_context(
    upload_root: str | Path,
    session_username: str,
    is_admin: bool,
    report_number: str,
    requested_owner: str | None = None,
    *,
    create: bool = False,
) -> EnergyReportContext: ...
```

`EnergyReportContext` 至少包含：

```text
owner_username
owner_storage_key
report_number
energy_root
owner_root
report_dir
```

报告编号沿用 `secure_filename` 语义，但增加非空、单路径组件和长度上限校验。
创建目录时按 `energy_root → owner_root → report_dir` 顺序检查：

- 路径必须位于配置的 energy 根目录内；
- 已存在组件不能是符号链接；
- 解析后的父目录必须与预期父目录完全一致；
- 新目录由服务进程创建，不接受客户端提供绝对路径或目录键。

所有上传、预览、识别、标定、计算、产物下载和历史读取路由必须使用报告上下文。
生产代码中不得继续直接拼接：

```python
os.path.join(app.config['UPLOAD_FOLDER'], 'energy', report_number)
```

## 7. 请求与产物流

统一数据流为：

```text
登录 session
  → 解析有效 owner_username
  → 生成 owner_storage_key
  → 校验 report_number
  → 解析或创建 report_dir
  → 上传 / PDF 预览 / 识别 / 标定 / 计算 / 读取
```

报告目录内部文件名保持现有契约，例如：

```text
recognition.json
ai_overlay.jpg
ai_mask.png
building_plan_prepared_*.pdf
vector_pdf_fusion/pdf_component_overlay.png
vector_pdf_fusion/pdf_exterior_topology.json
```

本次只增加用户名父目录，不改变报告内部产物名称。

PDF 上传令牌增加 `owner_username` 字段。验证令牌时必须同时匹配：

- 当前有效报告所有者；
- 报告编号；
- 已保存文件名；
- 页数。

令牌不能跨用户、跨报告复用。外轮廓生成锁继续使用解析后的绝对 `report_dir` 作为
锁键，因此不同用户的同名报告互不阻塞。

## 8. 数据库

继续使用现有 `reports` 表，不新增重复报告表。报告身份从单独 `report_number`
改为复合身份：

```text
(username, report_number)
```

启动迁移执行：

1. 如缺少 `status` 和 `updated_at` 列，则用兼容 SQLite 的 `ALTER TABLE` 添加；
2. 查找重复 `(username, report_number)`，保留 `created_at` 最新、再以 `id` 最大
   决胜的一条；
3. 建立唯一索引 `reports_username_report_number_uq`；
4. 保留用户、几何、参数和结果字段；
5. 迁移在事务内完成，失败时回滚并阻止服务带着半迁移结构继续写入。

首次成功创建报告目录时即插入或更新当前用户的报告记录，状态至少覆盖：

```text
created
uploaded
recognizing
recognized
calculated
failed
```

后续更新必须使用：

```sql
WHERE username = ? AND report_number = ?
```

不得再用单独 `report_number` 更新或读取。

## 9. 历史列表和管理员界面

`GET /energy/reports`：

- 普通用户只返回自己的记录；
- 管理员返回全部用户记录；
- 每条记录包含 `username`、`report_number`、状态和原有能耗摘要。

`GET /energy/report/<report_number>`：

- 普通用户按 session 用户名查询；
- 管理员通过 `owner_username` 查询指定用户；
- 找不到该复合身份时返回 404；
- 普通用户指定其他用户时返回 403。

管理员历史卡片显示 `用户名 / 报告编号`，打开卡片后将目标用户名保存在页面当前
报告状态中。上传、识别、标定和计算请求继续携带该目标用户名。普通用户页面不
显示用户选择控件。

## 10. 错误处理

- 未登录：沿用现有登录拦截。
- 非管理员跨用户访问：HTTP 403。
- 非法用户名、报告编号、绝对路径或路径穿越：HTTP 400。
- 报告不存在：HTTP 404。
- 路径组件是符号链接或解析后逃逸根目录：拒绝请求并记录安全日志。
- 同一所有者同一报告正在执行外轮廓任务：沿用 HTTP 409。
- 不同所有者同名报告：允许并行执行。
- 管理员目标用户名不存在：HTTP 404，不自动创建幽灵用户目录。

错误响应不暴露服务器绝对路径、目录键、数据库语句或其他用户是否拥有某个具体
文件。

## 11. 测试

采用测试驱动开发，至少覆盖：

1. 中文、英文、空白和特殊字符用户名生成稳定安全目录键；
2. 清洗后可读部分相同的用户名仍因哈希不同而隔离；
3. 报告编号中的 `..`、斜杠、反斜杠和绝对路径被拒绝；
4. 两个普通用户使用相同报告编号时得到不同目录和数据库记录；
5. 普通用户不能上传、预览、识别、标定、计算或读取其他用户报告；
6. 管理员可以显式访问存在的用户报告；
7. 数据库普通用户不能通过使用管理员同名账号获得管理员权限；
8. PDF 上传令牌绑定所有者，跨用户使用返回错误；
9. 符号链接 owner 目录和报告目录被拒绝；
10. 图片、PDF、识别 JSON 和矢量融合产物写入用户目录；
11. 复合唯一索引迁移和按复合键 upsert 正确；
12. 管理员历史列表包含用户名，普通列表不泄露其他用户；
13. 搜索生产代码，确保报告业务不再直接拼接旧平面路径；
14. 全量现有单元测试继续通过。

## 12. 部署

生产部署步骤：

1. 备份 `/var/lib/bim-web/users.db` 和 `/var/lib/bim-web/uploads`；
2. 停止写入或短暂停止 `bim-web` 服务；
3. 检查 `/var/lib/bim-web/uploads/energy/` 下不存在旧平面报告目录；
4. 若发现旧目录，停止部署并报告，不自动移动或删除；
5. 拉取包含本功能的 GitHub `main`；
6. 启动服务并完成数据库事务迁移；
7. 用两个普通测试账号创建同名报告，验证磁盘和数据库隔离；
8. 验证普通账号跨用户请求返回 403；
9. 验证管理员可以访问两个用户的报告；
10. 验证 PDF 识别、两点比例尺和能耗计算完整流程。

回滚时恢复代码、数据库备份和上传目录备份。由于本次不兼容旧平面目录，部署前
检查是强制门禁，不能静默回退到旧路径。
