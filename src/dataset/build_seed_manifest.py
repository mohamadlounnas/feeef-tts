"""
Validate / merge seed JSONL manifests with rights fields.

Expects rows already downloaded into data/seed/*/manifest.jsonl (see M1 scripts).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.dataset.preprocess import reject_empty_transcripts, require_rights_fields

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="Input JSONL manifests")
    p.add_argument("--output", required=True, help="Merged seed JSONL")
    p.add_argument(
        "--commercial-only",
        action="store_true",
        help="Keep only research_and_commercial_with_attribution rows",
    )
    args = p.parse_args(argv)

    rows: List[Dict[str, Any]] = []
    for path in args.inputs:
        rows.extend(load_jsonl(Path(path)))
    rows = reject_empty_transcripts(rows)
    for row in rows:
        require_rights_fields(row)
    if args.commercial_only:
        rows = [r for r in rows if r.get("allowed_use") == "research_and_commercial_with_attribution"]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    hours = sum(float(r.get("duration") or 0) for r in rows) / 3600
    logger.info("Wrote %d rows (%.2f h) -> %s", len(rows), hours, out)


if __name__ == "__main__":
    main()
