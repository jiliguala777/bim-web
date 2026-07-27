# bim-web 项目说明

`bim-web` 是一个建筑能耗计算平台原型。平台通过网页上传/识别建筑图纸，提取建筑几何信息，然后让用户设置围护结构热工参数、空调系统、照明、通风等参数，最后进行建筑能耗估算或仿真计算。

当前我们的主要工作重点不是图纸识别本身，而是第二步：**围护结构参数库建设**。也就是在网页参数选项窗口中，为外墙、外墙保温、外窗、屋面、地面、遮阳等围护结构提供可选择的真实热工参数。

## 运行方式

```powershell
$env:ADMIN_USER="admin"
$env:ADMIN_PASSWORD="你的密码"
$env:SECRET_KEY="随机串"

pip install flask werkzeug
python web_server_server.py
```

默认访问地址：

```text
http://localhost:5000
```

## 当前网页功能链路

能耗页面在：

```text
templates/energy.html
```

后端主要入口在：

```text
web_server_server.py
energy_library.py
energy_calc.py
```

围护结构参数选择窗口会调用这些接口：

```text
/energy/library/walls
/energy/library/envelope/<category>
```

这些接口最终读取：

```text
E:\bim-web\data\envelope_databases
```

## 围护结构数据库

网页选项窗口当前连接的是：

```text
E:\bim-web\data\envelope_databases
```

主要数据库包括：

```text
wall_insulation.db       外墙外保温系统
windows.db               外窗传热系数、太阳得热系数
exterior_wall.db         基层外墙
roof.db                  屋面
floor.db                 地面/楼板
door.db                  外门
shading.db               遮阳/窗帘
floor_contact.db         地面接触类型
window_air_tightness.db  外窗气密性
```

目前真实数据主要集中在：

- `wall_insulation.db`：已经导入一批检测报告结构化结果，重点字段包括热阻、传热系数、导热系数。
- `windows.db`：已经导入一批外窗检测报告中抽取的传热系数。

其他库目前仍以样例数据为主，后续需要继续补充真实来源。

## 数据来源

围护结构参数库的数据来源有三类：

1. **自有检测报告**
   - 原始报告是 Word `.doc` 文件。
   - 通过分类、结构化、参数抽取后导入 SQLite 数据库。
   - 重点关注导热系数、传热系数、热阻、太阳得热系数等和负荷计算相关的热工参数。

2. **网络公开资料**
   - 存放在：

     ```text
     E:\bim-web\data\external_sources
     ```

   - 包括重庆、湘潭等公开发布的建筑节能材料或门窗幕墙热工参数资料。
   - 当前多为候选参数和待复核 CSV，尚未全部导入正式数据库。

3. **样例/人工补充数据**
   - 用于网页下拉框在真实库不完整时仍能运行。
   - 后续应逐步被真实数据替代。

## 原始检测报告与分类目录

完整检测报告位于：

```text
E:\shujuku\work_report\work_report
```

这是原始报告库，原则上**不移动、不改名、不直接修改**。

派生分类目录位于：

```text
E:\shujuku\organized_reports
```

主要分为：

```text
organized_reports\not_useful      封面、临时文件、低优先级等不需要内容
organized_reports\review_later    暂时无法判断或未知类别
organized_reports\useful          有用报告
```

`useful` 下进一步分为：

```text
useful\envelope   围护结构相关
useful\hvac       暖通/空调相关
useful\lighting   照明相关
```

注意：`organized_reports` 是派生索引层，不是原始报告库。后续处理应以 manifest/CSV 记录为准，不要把它当作唯一原始来源。

## 当前报告处理进度

原始报告清单来自：

```text
E:\shujuku\output_experiment\work_report_manifest.csv
```

截至当前整理结果：

```text
原始报告总数：98,999
主内容筛选队列：50,827
封面/临时/跳过类：约 48,000+
```

之前已经完成内容筛选：

```text
offset 0 - 20000
```

近期新完成：

```text
offset 20000 - 20100   100 份试跑
offset 20100 - 21100   1000 份试跑
```

当前正在继续运行剩余批次：

```text
offset 21100
limit 29727
输出批次：21101_50827
```

当前批次输出文件：

```text
E:\shujuku\output_experiment\content_screening_21101_50827.csv
E:\shujuku\organized_reports\_organized_manifest_21101_50827.csv
```

其中 `content_screening_21101_50827.csv` 会边运行边追加；`_organized_manifest_21101_50827.csv` 要等整个批次完成后才生成。

## 已生成的审计清单

为了区分哪些报告已经处理、哪些还没处理，已经生成过一次审计目录：

```text
E:\bim-web\data\report_processing_audit
```

重要文件：

```text
summary.csv
unprocessed_main_queue.csv
processed_main_queue_inventory.csv
known_not_useful_or_skipped.csv
useful_organized_not_structured.csv
```

其中：

- `unprocessed_main_queue.csv`：当时尚未内容筛选的报告清单。
- `processed_main_queue_inventory.csv`：已内容筛选的报告清单。
- `useful_organized_not_structured.csv`：已经判定 useful，但还未结构化的报告。

## 当前批处理脚本

分类脚本位于：

```text
E:\shujuku\report_pipeline_v2
```

核心脚本：

```text
work_report_manifest.py          建原始报告 manifest
work_report_queue.py             建主队列，排除封面/临时文件
content_screening.py             打开 Word，按内容关键词分类
organize_files.py                将分类结果组织到 organized_reports
batch_screen_and_organize.py     筛选 + 组织的批处理入口
batch_watchdog.py                带卡死检测和自动恢复的入口
useful_structuring.py            对 useful 报告做结构化
```

目前已经对批处理脚本做过调整：

- 支持 `--source-root`，解决队列中 `work_report\...` 相对路径和真实目录不一致的问题。
- 默认不再每批重复组织全量 skipped 文件。
- watchdog 会等子进程完整退出，不会在 CSV 满行后提前杀掉组织阶段。
- 默认组织模式改为 `copy`；`hardlink` 在当前路径上出现过 `OSError(22, 函数不正确。)`，暂不作为默认。

## 正确的继续筛选命令

从 `E:\shujuku` 执行：

```powershell
cd E:\shujuku
$env:PYTHONPATH="E:\shujuku"

python -m report_pipeline_v2.batch_watchdog `
  --queue E:\shujuku\output_experiment\work_report_queues\main_queue.csv `
  --offset 21100 `
  --limit 29727 `
  --skipped E:\shujuku\output_experiment\work_report_queues\skipped.csv `
  --organized-root E:\shujuku\organized_reports `
  --output-root E:\shujuku\output_experiment `
  --skip-sources E:\shujuku\output_experiment\skip_sources.txt `
  --log-dir E:\shujuku\output_experiment\logs `
  --source-root E:\shujuku\work_report
```

说明：

- `--source-root E:\shujuku\work_report` 很重要，因为队列里的路径是 `work_report\...`。
- 不传 `--mode` 时默认是 `copy`。
- 支持 `--resume`，watchdog 启动子进程时会自动传给 `batch_screen_and_organize`。
- 如果 Word 卡住，watchdog 会把卡住文件加入 `skip_sources.txt` 并重启。

## 当前阶段要做什么

当前还不是结构化和入库阶段。

当前阶段的目标是：

```text
补齐剩余报告的大类内容筛选。
```

也就是把剩余未处理报告分成：

```text
useful/envelope
useful/hvac
useful/lighting
review_later/unknown
not_useful/low_priority
skipped
```

完成大类分类后，下一步才是围绕 `useful/envelope` 做更细筛选。

## 后续工作链路

推荐后续顺序：

1. **完成剩余 29,727 份报告的大类分类**
   - 目标是补齐 `organized_reports` 和 screening CSV。

2. **从新增的 `useful/envelope` 中做细分类**
   - 对应网页选项窗口中的围护结构类型：
     - 外墙
     - 外墙外保温
     - 外窗
     - 屋面
     - 地面/楼板
     - 外门
     - 遮阳

3. **筛出真正含热工参数的报告**
   - 重点参数：
     - 导热系数
     - 传热系数
     - 热阻
     - 太阳得热系数
     - 遮阳系数
     - 气密性等级

4. **结构化报告内容**
   - 保留 raw trace。
   - 提取 title/meta/items/category。
   - 不只保留最终参数，要保留来源证据。

5. **导入网页连接的 SQLite 参数库**
   - 目标目录：

     ```text
     E:\bim-web\data\envelope_databases
     ```

6. **在网页选项窗口中验证**
   - 下拉框是否出现新增数据。
   - 数值是否能正确写入能耗计算参数。
   - 参数单位和含义是否适合直接参与负荷计算。

## 注意事项

- 不要移动或修改 `E:\shujuku\work_report\work_report` 原始报告。
- 大批量任务必须保持可恢复：使用 CSV 追加、`--resume`、`skip_sources.txt`。
- `organized_reports` 是派生层，不是唯一事实来源；真正判断进度时优先看 CSV/manifest。
- 对已经分类过的报告，不要重新打开 Word 重跑，除非明确需要更新分类规则。
- 对已经判定 useful 但未结构化的报告，先不要急着结构化；后续要先做更精细的热工参数筛选。
- 涉及服务器、API key、SSH 私钥、部署密码的敏感文件不在 git 中，不要写入代码或 README。
