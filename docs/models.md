# 模型交付与可复现性

本仓库当前未公开发布识别权重，也没有可下载的模型 Release。
仅克隆源代码不能复现真实户型图推理；请通过仓库 Issues 联系维护者申请模型，
并确认数据、训练权重和再分发的授权。不要把本地模型路径当作公共下载地址。

| 后端 | 环境变量 | 默认文件 | 输入 |
|---|---|---|---|
| legacy_onnx | ONNX_MODEL_PATH | models/M2_pub_plus_user.onnx | float32 NCHW，1×3×512×512 |
| image_onnx | IMAGE_ONNX_MODEL_PATH | models/floorplan_image_dataset_5000.onnx | float32 NCHW，1×3×256×256 |

两个后端的预处理、输出头和类别顺序不同，不能互换文件或只改文件名。
获得模型后运行：

```powershell
python tools/check_models.py --backend legacy_onnx --path C:\models\M2_pub_plus_user.onnx
python tools/check_models.py --backend image_onnx --path C:\models\floorplan_image_dataset_5000.onnx
```

自检输出 SHA-256、输入/输出形状和一次零输入推理结果，失败返回非零退出码。
零输入推理只验证接口与可执行性，不代表真实识别精度。发布时应附模型版本、
SHA-256、训练/数据许可、类别映射、预处理、样例输入与参考结果。

缺失模型时 /energy/ai_status 会报告后端 unavailable，识别请求返回 501，且不会
创建新报告或空目录。其他材料库、计算、矢量解析和自动测试可以独立运行。
未发布实际模型前，不将“模型交付”记作全部完成。

2026-10-15 本机已有的两个文件已通过 CPU 零输入推理与上述输入/输出契约检查：

- legacy_onnx SHA-256: `ce855d661c12e7893aa6dbf7a184be641b6dfbba41b1c090fa7a8fd6a34d9d71`
- image_onnx SHA-256: `03181e083375d1a900355f11192b92065bcf89d6fceb667806c669dd0e841831`

这些校验值说明本机验证的具体版本，不提供下载能力，也不证明图纸识别精度。

## 可选引擎

旧 IFC、benchmark、CAD 和 EnergyPlus 适配器不随此开源仓库提供；默认关闭。
如部署者确有这些模块，可显式设置 BIM_WEB_IFC_MODULE、BIM_WEB_BENCHMARK_MODULE、
BIM_WEB_CAD_MODULE、BIM_WEB_ENERGYPLUS_MODULE 为可导入模块名。
配置的模块缺失或接口不完整时拒绝启动；没有配置时相关接口返回 501。
所需导出在 optional_engines 的调用处列出，不再启动时盲目导入缺失文件。
该旧 EnergyPlus 适配接口不等于其他开发分支中的 energyplus_bridge。
