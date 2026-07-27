from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "external_sources" / "manifests" / "material_source_search_2026-07-07_round2.csv"
CSV_DIR = ROOT / "data" / "external_sources" / "intermediate" / "xls_csv"
REVIEW = ROOT / "data" / "external_sources" / "review"
OUT_PRODUCTS = REVIEW / "material_product_index_expanded_2026-07-07.csv"
OUT_SOURCES = REVIEW / "material_list_source_review_2026-07-07.csv"


PRODUCT_COLUMNS = [
    "screen_status",
    "region",
    "batch",
    "category",
    "seq",
    "product_name",
    "standard",
    "public_id",
    "company",
    "production_address",
    "valid_until",
    "source_title",
    "source_url",
    "local_path",
    "source_file",
    "has_thermal_values",
    "parameter_followup",
]

SOURCE_COLUMNS = [
    "region",
    "source_title",
    "url",
    "local_path",
    "source_type",
    "review_status",
    "product_row_status",
    "thermal_value_status",
    "notes",
]


def clean(value: object) -> str:
    text = str(value or "").replace("\ufeff", "").replace("\u3000", " ")
    return re.sub(r"\s+", " ", text).strip().strip(",")


def read_manifest() -> list[dict[str, str]]:
    if not MANIFEST.exists():
        return []
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def infer_region(text: str) -> str:
    for region in ("常德", "岳阳", "衡阳", "邵阳", "湘潭", "上海", "重庆"):
        if region in text:
            return region
    return ""


def infer_batch(text: str) -> str:
    match = re.search(r"(\d{4}年[^，,、]*?第[一二三四五六七八九十]+批)", text)
    if match:
        return match.group(1)
    match = re.search(r"(第[一二三四五六七八九十]+批)", text)
    if match:
        return match.group(1)
    return ""


def source_url_for_file(rows: list[dict[str, str]], stem: str) -> tuple[str, str]:
    best_title = ""
    best_url = ""
    for row in rows:
        local = clean(row.get("local_path"))
        if not local:
            continue
        if Path(local).stem in stem or stem in Path(local).stem:
            return clean(row.get("title")), clean(row.get("url"))
        title = clean(row.get("title"))
        if title and title in stem:
            best_title = title
            best_url = clean(row.get("url"))
    return best_title, best_url


def read_excel_csv(path: Path) -> list[list[str]]:
    for encoding in ("utf-8-sig", "gb18030", "mbcs"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                rows = [[clean(c) for c in row] for row in csv.reader(f)]
            return [row for row in rows if any(row)]
        except UnicodeDecodeError:
            continue
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return [[clean(c) for c in row] for row in csv.reader(f) if any(row)]


def normalize_headers(row: list[str]) -> list[str]:
    return [clean(c).replace("（", "(").replace("）", ")") for c in row]


def header_index(headers: list[str], names: list[str]) -> int | None:
    for i, header in enumerate(headers):
        for name in names:
            if name in header:
                return i
    return None


def extract_rows_from_csv(path: Path, manifest_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = read_excel_csv(path)
    if len(rows) < 3:
        return []
    title = next((clean(row[0]) for row in rows[:4] if len(row) == 1 or "公示名单" in clean(row[0]) or "汇总表" in clean(row[0])), "")
    source_title, source_url = source_url_for_file(manifest_rows, path.stem)
    source_title = title or source_title or path.stem

    header_at = None
    for i, row in enumerate(rows[:8]):
        headers = normalize_headers(row)
        if any("材料" in h and "名称" in h for h in headers) and any("公示编号" in h for h in headers):
            header_at = i
            break
    if header_at is None:
        return []

    headers = normalize_headers(rows[header_at])
    idx = {
        "category": header_index(headers, ["类别"]),
        "seq": header_index(headers, ["序号"]),
        "product_name": header_index(headers, ["材料(产品)名称", "材料（产品）名称", "产品名称"]),
        "standard": header_index(headers, ["执行标准"]),
        "public_id": header_index(headers, ["公示编号"]),
        "company": header_index(headers, ["企业名称"]),
        "production_address": header_index(headers, ["生产地址"]),
        "valid_until": header_index(headers, ["有效期至"]),
        "batch": header_index(headers, ["批次"]),
    }

    current: dict[str, str] = {}
    product_rows: list[dict[str, str]] = []
    for row in rows[header_at + 1 :]:
        def cell(name: str) -> str:
            i = idx.get(name)
            return clean(row[i]) if i is not None and i < len(row) else ""

        public_id = cell("public_id")
        company = cell("company")
        seq = cell("seq")
        if not (public_id or company or seq):
            continue

        for key in ("category", "product_name", "standard"):
            value = cell(key)
            if value:
                current[key] = value

        batch = cell("batch") or infer_batch(source_title) or infer_batch(path.stem)
        product_rows.append(
            {
                "screen_status": "product_index_no_thermal_value",
                "region": infer_region(source_title) or infer_region(path.stem),
                "batch": batch,
                "category": current.get("category", ""),
                "seq": seq,
                "product_name": current.get("product_name", ""),
                "standard": current.get("standard", ""),
                "public_id": public_id,
                "company": company,
                "production_address": cell("production_address"),
                "valid_until": cell("valid_until"),
                "source_title": source_title,
                "source_url": source_url,
                "local_path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "source_file": path.name,
                "has_thermal_values": "no",
                "parameter_followup": "反查型式检验报告、企业产品资料或软件材料库，补导热系数/传热系数/热阻",
            }
        )
    return product_rows


def source_review_rows(manifest_rows: list[dict[str, str]], product_count_by_source: dict[str, int]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in manifest_rows:
        local = clean(item.get("local_path"))
        title = clean(item.get("title"))
        source_type = clean(item.get("source_type"))
        count = product_count_by_source.get(clean(item.get("url")), 0)
        if not count:
            count = product_count_by_source.get(local, 0)
        if count:
            row_status = f"parsed_product_rows:{count}"
        elif source_type == "official_product_list_image":
            row_status = "needs_ocr_image"
        elif source_type == "official_product_list_pdf":
            row_status = "needs_ocr_pdf_or_render"
        elif source_type == "official_product_notice_html" and "岳阳" in title:
            row_status = "notice_has_embedded_images_or_pdf"
        elif source_type == "official_product_notice_html" and "衡阳" in title:
            row_status = "notice_page_collected_needs_manual_or_ocr_check"
        elif clean(item.get("review_status")) == "fetch_failed":
            row_status = "fetch_failed"
        else:
            row_status = "source_collected_no_direct_rows"
        rows.append(
            {
                "region": infer_region(title + " " + local),
                "source_title": title,
                "url": clean(item.get("url")),
                "local_path": local,
                "source_type": source_type,
                "review_status": clean(item.get("review_status")),
                "product_row_status": row_status,
                "thermal_value_status": "no_direct_thermal_values_found" if count else clean(item.get("thermal_value_status")),
                "notes": clean(item.get("notes")),
            }
        )
    return rows


def dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, ...]] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = (
            row["region"],
            row["public_id"],
            row["company"],
            row["product_name"],
            row["valid_until"],
            row["source_title"],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)
    manifest_rows = read_manifest()
    products: list[dict[str, str]] = []
    current_csv_paths: list[Path] = []
    for row in manifest_rows:
        if clean(row.get("source_type")) != "official_product_list_spreadsheet":
            continue
        local = clean(row.get("local_path"))
        if not local:
            continue
        current_csv_paths.append(CSV_DIR / (Path(local).stem + ".csv"))
    if not current_csv_paths:
        current_csv_paths = sorted(CSV_DIR.glob("material-list-*.csv"))
    for path in sorted(set(current_csv_paths)):
        if not path.exists():
            continue
        products.extend(extract_rows_from_csv(path, manifest_rows))
    products = dedupe(products)

    with OUT_PRODUCTS.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PRODUCT_COLUMNS)
        writer.writeheader()
        writer.writerows(products)

    product_count_by_source: dict[str, int] = {}
    for row in products:
        product_count_by_source[row["local_path"]] = product_count_by_source.get(row["local_path"], 0) + 1
        source_url = row.get("source_url", "")
        if source_url:
            product_count_by_source[source_url] = product_count_by_source.get(source_url, 0) + 1
    with OUT_SOURCES.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SOURCE_COLUMNS)
        writer.writeheader()
        writer.writerows(source_review_rows(manifest_rows, product_count_by_source))

    print(f"product_rows={len(products)}")
    print(f"sources_reviewed={len(manifest_rows)}")
    print(f"products={OUT_PRODUCTS}")
    print(f"sources={OUT_SOURCES}")


if __name__ == "__main__":
    main()
