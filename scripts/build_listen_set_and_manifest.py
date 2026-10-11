#!/usr/bin/env python3
"""
Build listen_set.json (Casablanca DZ refs for research A/B) + merge seed manifests.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "seed"
LISTEN = SEED / "listen_set"


def load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    LISTEN.mkdir(parents=True, exist_ok=True)
    casa = load_jsonl(SEED / "casablanca_dz" / "manifest.jsonl")
    # Prefer 3–8s clips with solid Arabic text for F5 refs
    refs = []
    for row in casa:
        d = float(row.get("duration") or 0)
        t = (row.get("text") or "").strip()
        if 3.0 <= d <= 8.0 and len(t) >= 8:
            refs.append(row)
        if len(refs) >= 12:
            break
    if len(refs) < 12:
        refs = casa[:12]

    listen_items = []
    for i, row in enumerate(refs):
        src = Path(row["audio_filepath"])
        dest = LISTEN / f"ref_{i+1:02d}.wav"
        if src.exists():
            shutil.copy2(src, dest)
        listen_items.append(
            {
                "id": f"L{i+1:02d}",
                "ref_audio": str(dest),
                "text": row.get("text"),
                "license": row.get("license"),
                "allowed_use": row.get("allowed_use"),
                "source": row.get("source"),
                "duration": row.get("duration"),
                "note": "Casablanca Algeria — research_eval_only; not for commercial train",
            }
        )
    (LISTEN / "listen_set.json").write_text(
        json.dumps(listen_items, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Prefer CAFE v2 if present
    cafe = SEED / "processed" / "cafe_manifest_v2.jsonl"
    if not cafe.exists():
        cafe = SEED / "processed" / "cafe_manifest.jsonl"
    parts = [
        cafe,
        SEED / "dvoice_ary" / "manifest.jsonl",
        SEED / "casablanca_dz" / "manifest.jsonl",
        SEED / "moroccan_wiki_ary" / "manifest.jsonl",
    ]
    all_rows = []
    for p in parts:
        all_rows.extend(load_jsonl(p))
    all_rows = [r for r in all_rows if (r.get("text") or "").strip()]
    out = SEED / "seed_manifest_v1.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hours = sum(float(r.get("duration") or 0) for r in all_rows) / 3600
    print(f"listen refs={len(listen_items)} seed_rows={len(all_rows)} hours={hours:.3f} -> {out}")


if __name__ == "__main__":
    main()
