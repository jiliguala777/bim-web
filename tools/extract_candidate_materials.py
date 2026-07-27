import csv
import re
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
SQL_PATH = DATA_DIR / "db_test.sql"
CSV_PATH = DATA_DIR / "candidate_material_parameters.csv"
SUMMARY_PATH = DATA_DIR / "candidate_material_parameters_summary.txt"


META_FIELDS = {
    "样品名称": "sample_name",
    "生产单位": "manufacturer",
    "规格型号": "spec_model",
    "检验依据": "test_basis",
    "判定依据": "criteria_basis",
    "检验项目": "test_items",
    "检验结论": "test_conclusion",
    "商 标": "brand",
}

RELEVANT_KEYWORDS = [
    "热阻",
    "导热系数",
    "传热系数",
    "热惰性",
    "蓄热",
    "密度",
    "比热",
    "单位面积质量",
    "厚度",
    "遮阳",
    "太阳得热",
    "透射比",
    "SHGC",
    "COP",
    "EER",
    "LPD",
    "EPD",
    "功率",
    "效率",
]


def split_insert_tuples(values_text):
    tuples = []
    current = []
    depth = 0
    in_string = False
    escape = False

    for char in values_text:
        if in_string:
            current.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == "'":
                in_string = False
            continue

        if char == "'":
            in_string = True
            current.append(char)
        elif char == "(":
            if depth > 0:
                current.append(char)
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                tuples.append("".join(current))
                current = []
            else:
                current.append(char)
        else:
            if depth > 0:
                current.append(char)

    return tuples


def split_tuple_fields(tuple_text):
    fields = []
    current = []
    in_string = False
    escape = False

    for char in tuple_text:
        if in_string:
            current.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == "'":
                in_string = False
            continue

        if char == "'":
            in_string = True
            current.append(char)
        elif char == ",":
            fields.append(parse_sql_value("".join(current)))
            current = []
        else:
            current.append(char)

    fields.append(parse_sql_value("".join(current)))
    return fields


def parse_sql_value(raw):
    value = raw.strip()
    if value.upper() == "NULL":
        return ""
    if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        value = value[1:-1]
        value = value.replace("\\'", "'").replace("\\\\", "\\")
        value = value.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
    return value


def iter_insert_rows(sql_text):
    pattern = re.compile(r"INSERT INTO `([^`]+)` VALUES (.*?);", re.S)
    for match in pattern.finditer(sql_text):
        table = match.group(1)
        values_text = match.group(2)
        for tuple_text in split_insert_tuples(values_text):
            yield table, split_tuple_fields(tuple_text)


def normalize_text(value):
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split())


def metadata_key(field_name):
    field = normalize_text(field_name)
    for prefix, key in META_FIELDS.items():
        if prefix in field:
            return key
    return None


def parameter_type(text):
    if "导热系数" in text:
        return "thermal_conductivity_lambda"
    if "热阻" in text:
        return "thermal_resistance_r"
    if "传热系数" in text:
        return "u_value"
    if "密度" in text:
        return "density"
    if "比热" in text:
        return "specific_heat"
    if "单位面积质量" in text:
        return "mass_per_area"
    if "厚度" in text:
        return "thickness"
    if "太阳得热" in text or "SHGC" in text.upper():
        return "shgc"
    if "透射比" in text:
        return "transmittance"
    if "气密" in text:
        return "airtightness"
    if "COP" in text.upper():
        return "cop"
    if "EER" in text.upper():
        return "eer"
    return "supporting_property"


def recommended_use(sample_name, item_text, param_type):
    text = f"{sample_name} {item_text}"
    if "窗" in text or "玻璃" in text or "幕墙" in text:
        return "window_system"
    if "屋面" in text:
        return "roof_assembly"
    if "空调" in text or "热泵" in text or "锅炉" in text or param_type in {"cop", "eer"}:
        return "equipment"
    if "外墙" in text or "墙体" in text or "保温系统" in text or param_type in {"thermal_resistance_r", "u_value"}:
        return "wall_assembly"
    if "保温" in text or "聚苯" in text or "聚氨酯" in text or "岩棉" in text:
        return "material"
    return "review"


def extract_number_and_unit(text):
    value = normalize_text(text)
    number_match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    number = number_match.group(0) if number_match else ""

    unit = ""
    unit_patterns = [
        r"W/[（(]?m[·.]?K[）)]?",
        r"W/[（(]?m²[·.]?K[）)]?",
        r"W/[（(]?m2[·.]?K[）)]?",
        r"[（(]?m²[·.]?K[）)]?/W",
        r"[（(]?m2[·.]?K[）)]?/W",
        r"kg/m²",
        r"kg/m2",
        r"g/m²",
        r"g/m2",
        r"kN",
        r"MPa",
        r"%",
    ]
    for pattern in unit_patterns:
        unit_match = re.search(pattern, value, re.I)
        if unit_match:
            unit = unit_match.group(0)
            break
    return number, unit


def is_relevant(*values):
    text = " ".join(normalize_text(v) for v in values)
    upper_text = text.upper()
    return any(keyword in text or keyword in upper_text for keyword in RELEVANT_KEYWORDS)


def main():
    if not SQL_PATH.exists():
        raise FileNotFoundError(f"SQL file not found: {SQL_PATH}")

    sql_text = SQL_PATH.read_text(encoding="utf-8-sig")

    report_meta = defaultdict(dict)
    eav_items = {}
    parsed_rows = 0

    for table, row in iter_insert_rows(sql_text):
        parsed_rows += 1
        if table.endswith("_meta") and len(row) >= 6:
            report_id = normalize_text(row[1])
            key = metadata_key(row[4])
            value = normalize_text(row[5])
            if report_id and key and value and value != "------":
                existing = report_meta[report_id].get(key, "")
                if value not in existing:
                    report_meta[report_id][key] = f"{existing}; {value}".strip("; ")
        elif table.endswith("_items_eav") and len(row) >= 9:
            report_id = normalize_text(row[1])
            table_no = normalize_text(row[2])
            row_no = normalize_text(row[3])
            seq = normalize_text(row[5])
            item_path = normalize_text(row[6])
            attr_name = normalize_text(row[7])
            attr_value = normalize_text(row[8])
            key = (table, report_id, table_no, row_no, seq, item_path)
            item = eav_items.setdefault(
                key,
                {
                    "source_table": table,
                    "report_id": report_id,
                    "table_no": table_no,
                    "row_no": row_no,
                    "seq": seq,
                    "item_path": item_path,
                    "requirement": "",
                    "result": "",
                    "conclusion": "",
                    "other_attrs": [],
                },
            )
            if "技术要求" in attr_name:
                item["requirement"] = attr_value
            elif "检测结果" in attr_name:
                item["result"] = attr_value
            elif "单项判定" in attr_name:
                item["conclusion"] = attr_value
            else:
                item["other_attrs"].append(f"{attr_name}: {attr_value}")

    candidates = []
    for item in eav_items.values():
        meta = report_meta.get(item["report_id"], {})
        sample_name = meta.get("sample_name", "")
        parameter_text = " ".join(
            [
                item["item_path"],
                item["requirement"],
                item["result"],
            ]
        )
        context_text = " ".join([sample_name, meta.get("test_items", ""), parameter_text])
        if not is_relevant(parameter_text):
            continue

        param_type = parameter_type(parameter_text)
        number, unit = extract_number_and_unit(item["result"] or item["requirement"])
        candidates.append(
            {
                "review_status": "",
                "recommended_use": recommended_use(sample_name, context_text, param_type),
                "parameter_type": param_type,
                "numeric_value": number,
                "unit": unit,
                "source_report_id": item["report_id"],
                "sample_name": sample_name,
                "manufacturer": meta.get("manufacturer", ""),
                "brand": meta.get("brand", ""),
                "spec_model": meta.get("spec_model", ""),
                "test_basis": meta.get("test_basis", ""),
                "test_items": meta.get("test_items", ""),
                "item_path": item["item_path"],
                "requirement": item["requirement"],
                "result": item["result"],
                "conclusion": item["conclusion"],
                "source_table": item["source_table"],
                "table_no": item["table_no"],
                "row_no": item["row_no"],
                "seq": item["seq"],
                "notes": "; ".join(item["other_attrs"]),
            }
        )

    candidates.sort(
        key=lambda row: (
            row["recommended_use"],
            row["source_report_id"],
            row["parameter_type"],
            row["item_path"],
        )
    )

    fieldnames = [
        "review_status",
        "recommended_use",
        "parameter_type",
        "numeric_value",
        "unit",
        "source_report_id",
        "sample_name",
        "manufacturer",
        "brand",
        "spec_model",
        "test_basis",
        "test_items",
        "item_path",
        "requirement",
        "result",
        "conclusion",
        "source_table",
        "table_no",
        "row_no",
        "seq",
        "notes",
    ]

    with CSV_PATH.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)

    by_use = defaultdict(int)
    by_param = defaultdict(int)
    for row in candidates:
        by_use[row["recommended_use"]] += 1
        by_param[row["parameter_type"]] += 1

    summary_lines = [
        f"SQL file: {SQL_PATH}",
        f"Parsed INSERT rows: {parsed_rows}",
        f"Reports with metadata: {len(report_meta)}",
        f"Raw EAV items: {len(eav_items)}",
        f"Candidate rows: {len(candidates)}",
        "",
        "By recommended_use:",
        *[f"- {key}: {by_use[key]}" for key in sorted(by_use)],
        "",
        "By parameter_type:",
        *[f"- {key}: {by_param[key]}" for key in sorted(by_param)],
    ]
    SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Wrote {len(candidates)} candidates to {CSV_PATH}")
    print(f"Wrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
