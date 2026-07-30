"""Side-effect-free subprocess entry for one PDF page's vector extraction."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 4:
        return 2
    result_path = Path(arguments[0])
    source = arguments[1]
    page_index = int(arguments[2])
    dpi = int(arguments[3])
    try:
        from vector_pdf_scale import extract_vector_page

        page_data = extract_vector_page(
            source,
            page_index=page_index,
            dpi=dpi,
        )
        payload = {"ok": True, "page_data": page_data}
    except BaseException as exc:
        payload = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
    except BaseException:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
