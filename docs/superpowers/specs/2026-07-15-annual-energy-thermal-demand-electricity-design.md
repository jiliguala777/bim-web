# 全年冷热需求与总用电量分开展示设计

更新日期：2026-07-15

## 1. 目标

在不增加房间类型、不引入逐时气象文件、不新增独立设计负荷模块的前提下，提高现有 HDD/CDD 全年能耗计算的可解释性和完整性。

网站继续计算全年能耗，但把结果明确拆成两层：

1. 全年供暖、制冷热需求，单位 `kWh_th/a`；
2. 由冷热需求经季节性能系数换算，并叠加风机、照明、插座设备等直接用电后的全年总用电量，单位 `kWh/a`。

同时使用 HDD/CDD 计算等效峰值估算，作为与原冷热负荷计算书进行数量级校核的辅助结果。该值不命名为设计负荷，也不用于设备选型。

## 2. 范围

### 2.1 本次包含

- 第三步增加一套全局人员、新风、运行时段和季节效率参数；
- 不设置房间类型，不设置逐房间参数；
- 人数由计算面积和人均占用面积推导；
- 新风量由人数和人均新风量推导；
- 新风显热需求加入全年供暖、制冷热需求；
- 新排风机耗电单独计入全年用电量；
- 供暖、制冷季节性能系数由用户输入，并覆盖设备类型的内置默认值；
- 结果页把热需求和用电量分成两个独立区块；
- 增加 HDD/CDD 等效峰值估算和计算边界说明；
- 历史报告保存和重新加载新增参数及新增结果。

### 2.2 本次不包含

- 8760 小时气象模拟；
- 最大日或最大小时检索；
- 房间多边形、房间类型和分区计算；
- 新风潜热和除湿能耗；
- 人员、照明、设备内热对供暖和制冷需求的逐时耦合；
- 严格设计冷负荷、热负荷计算和设备选型。

人员、照明和设备仍分别用于推导新风量、计算照明用电和插座设备用电。由于 HDD18 本身是平衡温度方法，且当前没有供暖季、制冷季逐时运行分配，本次不再把全年内部得热直接从供暖需求中扣除或全部加入制冷需求，避免重复计算或错误分季。

## 3. 数据来源和默认值

文化宫降碳报告提供了常见房间的温度、人员密度、人均新风量、照明功率密度、设备功率密度、热回收方式、工作日新风时段和季节性能系数。

冷热负荷计算书提供了室外设计温度：

- 冬季室外供暖计算干球温度：`-5℃`；
- 夏季空调室外干球温度：`34.9℃`。

这两个室外设计温度只用于第 7 节的等效峰值校核，不参与全年 HDD/CDD 的累计计算。全年 HDD/CDD 以气候库的参考基准温度数据为起点，再根据用户填写的室内供暖、制冷设定温度进行调整。

第三步采用以下初始值，所有字段均可手动修改：

| 字段 | 默认值 | 单位 | 说明 |
|---|---:|---|---|
| 室内供暖设定温度 | 18 | ℃ | 用户确认的本次默认值 |
| 室内制冷设定温度 | 26 | ℃ | 文化宫报告 |
| 人均占用面积 | 10 | m²/人 | 对应 0.1 人/m² |
| 人均新风量 | 30 | m³/(h·人) | 文化宫常见房间 |
| 每日运行小时 | 12 | h/天 | 对应报告 7—18 时段的 12 个小时段 |
| 年运行天数 | 250 | 天/年 | 报告未明确，采用可编辑默认值 |
| 热回收效率 | 0 | % | 报告为“无” |
| 渗透风换气次数 | 0 | 次/h | 文化宫报告 |
| 风机单位风量功率 | 0.5 | W/(m³/h) | 报告未明确，沿用网站后端默认值 |
| 制热季节性能系数 | 1.90 | - | 文化宫设计建筑 HSPF |
| 制冷季节性能系数 | 2.30 | - | 文化宫设计建筑 SEER |
| 照明功率密度 | 8 | W/m² | 文化宫常见房间 |
| 设备功率密度 | 15 | W/m² | 文化宫常见房间 |
| 冬季室外设计温度 | -5 | ℃ | 文化宫热负荷计算书 |
| 夏季室外设计温度 | 34.9 | ℃ | 文化宫冷负荷计算书 |

## 4. 第三步页面设计

第三步继续使用现有左右两列卡片，不增加房间类型选择。

### 4.1 空调与供暖设备系统

保留：

- 建筑所在城市；
- 供暖系统类型；
- 室内供暖设定温度；
- 制冷系统类型；
- 室内制冷设定温度。

新增：

- 制热季节性能系数；
- 制冷季节性能系数；
- 冬季室外设计温度；
- 夏季室外设计温度。

设备类型改变时可更新推荐效率，但用户手动修改后的数值为最终计算值。

### 4.2 人员、新风和运行参数

新增或调整：

- 人均占用面积；
- 人均新风量；
- 每日运行小时；
- 年运行天数；
- 自动显示年运行小时；
- 热回收效率；
- 渗透风换气次数；
- 风机单位风量功率；
- 照明功率密度；
- 设备功率密度。

现有“换气次数 ACH”不再同时代表机械新风和渗透风。机械新风由人数和人均新风量确定；ACH 字段改为“渗透风换气次数”。

### 4.3 即时派生值

第三步显示只读摘要：

- 计算人数；
- 计算总新风量；
- 年运行小时；
- 热回收后的等效新风量。

## 5. 输入数据结构

前端向 `/energy/ai_simulate` 增加以下字段：

```json
{
  "area_per_person_m2": 10,
  "fresh_air_m3h_per_person": 30,
  "daily_operation_hours": 12,
  "annual_operation_days": 250,
  "heat_recovery_efficiency": 0,
  "infiltration_ach": 0,
  "fan_power_w_per_m3h": 0.5,
  "heating_seasonal_efficiency": 1.9,
  "cooling_seasonal_efficiency": 2.3,
  "winter_design_temperature_c": -5,
  "summer_design_temperature_c": 34.9
}
```

后端在 `calc_params` 中按职责组织：

- `occupancy`：人均面积、推导人数；
- `ventilation`：人均新风、总新风、热回收、渗透 ACH、风机功率；
- `schedule`：每日小时、年天数、年运行小时；
- `heating`、`cooling`：室内设定温度、季节性能系数、室外设计温度。

旧报告缺少新增字段时使用本设计的默认值，保证历史记录可重新加载。

## 6. 计算链

### 6.1 派生人员、风量和时间

```text
occupants = total_area / area_per_person_m2
fresh_air_flow = occupants × fresh_air_m3h_per_person
annual_operation_hours = daily_operation_hours × annual_operation_days
operation_fraction = annual_operation_hours / 8760
effective_fresh_air_flow = fresh_air_flow × (1 - heat_recovery_efficiency)
```

输入值必须进行边界处理：面积、人数、风量、小时和天数不得为负；每日运行小时不超过 24；年运行天数不超过 365；热回收效率限制在 0—100%。

### 6.2 围护结构年度热需求

当前西安气候库提供 `HDD18=2400 K·d` 和 `CDD26=200 K·d`。HDD/CDD 中的 18℃、26℃是度日数基准温度；严格来说是建筑开始需要供暖或制冷的平衡温度，通常接近但不必等于室内设定温度，因为内部得热、太阳得热和建筑热惰性会影响平衡温度。

在当前缺少逐日气象温度的条件下，继续采用现有网站的设定温度经验修正：

```text
HDD_adjusted = max(0, HDD18 + (indoor_heating_temperature - 18) × 120)
CDD_adjusted = max(0, CDD26 + (26 - indoor_cooling_temperature) × 90)
```

本次西安室内供暖设定温度为 18℃，因此网站采用：

```text
HDD_adjusted = 2400 + (18 - 18) × 120 = 2400 K·d
```

因此，室内供暖、制冷设定温度决定采用的调整后 HDD/CDD；冬季 `-5℃` 和夏季 `34.9℃` 不参与这一步。经验修正不等同于用逐日温度重新计算度日数，结果页需要保留这一模型边界说明。

继续使用现有简单模式或精确模式得到围护结构供暖、制冷需求：

```text
envelope_heating_kwh_th = UA × HDD_adjusted × 24 / 1000
envelope_cooling_kwh_th = UA × CDD_adjusted × 24 / 1000 + annual_solar_gain
```

精确模式继续保留按朝向墙、窗、门、屋面、地面和气密性计算。围护结构 HDD/CDD 结果暂不乘运行时段比例，因为当前基准模型默认建筑维持设定温度；运行时段只控制人员、新风、照明、设备和风机。这样可以避免把当前全年围护结构需求突然按 `3000/8760` 大幅缩小。

### 6.3 新风年度显热需求

新风换热能力：

```text
fresh_air_ua_w_per_k = 0.335 × effective_fresh_air_flow
```

年度显热需求：

```text
fresh_air_heating_kwh_th = fresh_air_ua × HDD_adjusted × 24 / 1000 × operation_fraction
fresh_air_cooling_kwh_th = fresh_air_ua × CDD_adjusted × 24 / 1000 × operation_fraction
```

本次不计算湿空气焓差和除湿潜热，因此结果中必须注明“新风制冷需求仅含显热”。

### 6.4 渗透风年度显热需求

```text
building_volume = total_area × floor_height
infiltration_flow = building_volume × infiltration_ach
infiltration_ua = 0.335 × infiltration_flow
```

渗透风按围护结构全年边界计算，不乘人员运行时段比例：

```text
infiltration_heating_kwh_th = infiltration_ua × HDD_adjusted × 24 / 1000
infiltration_cooling_kwh_th = infiltration_ua × CDD_adjusted × 24 / 1000
```

当显式提供 `infiltration_ach` 时，以该值为准，不再叠加当前详细模式根据窗气密性推导的简化渗透风 UA，避免重复计算。旧报告未提供该字段时，继续兼容原气密性推导逻辑。

### 6.5 全年冷热需求

```text
annual_heating_demand_kwh_th =
    envelope_heating_kwh_th
  + fresh_air_heating_kwh_th
  + infiltration_heating_kwh_th

annual_cooling_demand_kwh_th =
    envelope_cooling_kwh_th
  + fresh_air_cooling_kwh_th
  + infiltration_cooling_kwh_th
```

太阳得热继续保留在现有围护结构制冷需求中。内部得热本次不与冷热需求耦合，原因见 2.2。

### 6.6 设备与直接用电

对于热泵或电制冷设备：

```text
heating_electricity_kwh = annual_heating_demand_kwh_th / heating_seasonal_efficiency
cooling_electricity_kwh = annual_cooling_demand_kwh_th / cooling_seasonal_efficiency
```

直接用电：

```text
fan_electricity_kwh = fresh_air_flow × fan_power × annual_operation_hours / 1000
lighting_electricity_kwh = LPD × total_area × annual_operation_hours / 1000
equipment_electricity_kwh = EPD × total_area × annual_operation_hours / 1000
```

全年总用电量：

```text
total_electricity_kwh =
    heating_electricity_kwh
  + cooling_electricity_kwh
  + fan_electricity_kwh
  + lighting_electricity_kwh
  + equipment_electricity_kwh
  + electric_dhw_kwh
```

如果用户选择燃气锅炉、集中供热等非电供暖系统，供暖购入能耗不得混入“总用电量”；结果应单列对应燃料或热量。文化宫默认多联机热泵按电力处理。

## 7. HDD/CDD 等效峰值校核

### 7.1 供暖

```text
heating_delta_t_design = indoor_heating_temperature - winter_design_temperature
heating_equivalent_full_load_hours = HDD_adjusted × 24 / heating_delta_t_design
heating_peak_estimate_kw_th = annual_heating_demand_kwh_th / heating_equivalent_full_load_hours
```

### 7.2 制冷

```text
cooling_delta_t_design = summer_design_temperature - indoor_cooling_temperature
cooling_equivalent_full_load_hours = CDD_adjusted × 24 / cooling_delta_t_design
cooling_peak_estimate_kw_th = annual_cooling_demand_kwh_th / cooling_equivalent_full_load_hours
```

CDD 峰值估算受太阳辐射、内部得热和新风潜热影响更大，结果页需标注其不确定性高于供暖估算。

当温差、HDD 或 CDD 为零时，不输出峰值数字，显示“当前气候或设定温度下无法估算”。

## 8. 结果页设计

结果页分为两个主区块，不把热需求和用电量放在同一张总表中。

### 8.1 全年冷热需求

单位统一为 `kWh_th/a`：

- 全年供暖热需求；
- 全年制冷热需求；
- 围护结构供暖、制冷需求；
- 新风供暖、制冷显热需求；
- 渗透风供暖、制冷显热需求；
- HDD 供暖等效峰值估算，单位 `kW_th`；
- CDD 制冷等效峰值估算，单位 `kW_th`。

### 8.2 全年用电量

单位统一为 `kWh/a`：

- 供暖设备用电；
- 制冷设备用电；
- 新排风机用电；
- 照明用电；
- 插座设备用电；
- 电生活热水用电；
- 全年总用电量；
- 单位面积总用电量，`kWh/(m²·a)`。

结果页保留现有能效评级，但评级依据改为全年总用电量 EUI，并明确不以 `kWh_th` 参与总用电量直接相加。

### 8.3 计算边界提示

结果页固定显示：

- 当前为 HDD/CDD 简化年度模型；
- 新风制冷仅含显热，不含除湿潜热；
- 未进行 8760 小时逐时计算；
- 等效峰值仅用于数量级校核，不是严格设计负荷；
- 人员、照明和设备内部得热尚未与冷热需求逐时耦合。

## 9. 后端返回结构

新增：

```json
{
  "annual_thermal_demand_kwh_th": {
    "heating_total": 0,
    "cooling_total": 0,
    "heating_envelope": 0,
    "cooling_envelope": 0,
    "heating_fresh_air": 0,
    "cooling_fresh_air_sensible": 0,
    "heating_infiltration": 0,
    "cooling_infiltration": 0
  },
  "annual_electricity_kwh": {
    "heating": 0,
    "cooling": 0,
    "fan": 0,
    "lighting": 0,
    "equipment": 0,
    "dhw": 0,
    "total": 0
  },
  "equivalent_peak_check_kw_th": {
    "heating": 0,
    "cooling": 0,
    "heating_equivalent_full_load_hours": 0,
    "cooling_equivalent_full_load_hours": 0
  },
  "derived_inputs": {
    "occupants": 0,
    "fresh_air_flow_m3h": 0,
    "annual_operation_hours": 0,
    "operation_fraction": 0
  },
  "model_boundaries": []
}
```

现有 `breakdown_kwh` 在过渡期继续返回，避免历史前端或报告读取失败；其值映射到新的年度用电分项，不再混入热需求量。

## 10. 错误处理

- 人均面积、季节性能系数为零时返回明确的 400 参数错误；
- 运行小时、运行天数、热回收效率超出范围时前端提示并阻止提交，后端再次校验；
- 供暖或制冷系统为“无”时，对应设备用电为零，但仍可显示建筑对应的热需求；
- 新风量为零时，新风冷热需求和风机用电均为零；
- HDD/CDD 峰值估算分母为零时返回 `null`，前端显示不可估算；
- 旧报告字段缺失时采用兼容默认值，不修改原始历史 JSON。

## 11. 验证与验收

### 11.1 后端单元测试

- 人数、总新风量和年运行小时推导正确；
- 热回收效率为 0% 和 50% 时，新风显热需求关系正确；
- 新风量为零时新风需求和风机用电为零；
- 显式渗透 ACH 不与窗气密性推导值重复叠加；
- 供暖、制冷热需求不除以季节性能系数；
- 供暖、制冷设备用电正确除以 HSPF/SEER；
- 总用电量只汇总电力分项；
- HDD/CDD 等效峰值公式和零分母处理正确；
- 简单模式和精确围护模式均可返回新结构；
- 旧参数负载仍可计算。

### 11.2 前端测试

- 第三步包含全部新增字段且默认值正确；
- 修改人均面积、新风量、运行小时和天数时，派生摘要即时更新；
- 历史报告重新加载新增字段；
- 结果页存在两个独立区块，单位分别为 `kWh_th/a` 和 `kWh/a`；
- 峰值估算标注为校核值；
- 无系统或零 CDD/HDD 时不显示 `NaN`、`Infinity`。

### 11.3 浏览器验收

用文化宫四层当前围护参数完成一次精确模式计算，确认：

1. 全年供暖、制冷热需求独立显示；
2. 热需求经 1.90/2.30 换算为设备用电；
3. 风机、照明、设备用电独立显示；
4. 全年总用电量等于各电力分项之和；
5. HDD 等效供暖峰值可与四层计算书约 `39.43 kW` 做数量级比较；
6. 页面和控制台无错误，历史报告保存、加载正常。

## 12. 实施边界

本设计是现有 HDD/CDD 模型的可解释性和参数完整性升级，不声称达到 EnergyPlus 或逐时动态模拟精度。后续若获得西安或运城 8760 小时气象文件，可在保持第三步输入结构的前提下替换计算内核，并继续沿用热需求、设备用电分层的结果结构。
