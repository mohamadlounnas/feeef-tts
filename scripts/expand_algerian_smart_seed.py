#!/usr/bin/env python3
"""
Expand Mac SILMA FT seed with Algerian-preferring clips under tight disk budget.

Strategy (disk-aware):
  1. Add ``fares5462/darija-dz-tts-v1`` (~0.6 h casual DZ dialogue; labeled text).
  2. Keep existing CAFE Algerian segments (CC-BY).
  3. Cap Moroccan DVoice hours so Algerian hours dominate the train mix.
  4. Hard-ban empty / tiny garbage transcripts.
  5. Abort if free disk would fall below ``--min-free-gb`` after the download.

Writes:
  - ``data/seed/darija_dz_tts_v1/manifest.jsonl`` + ``wavs/``
  - ``data/seed/smart_train_manifest.jsonl`` (+ ``.summary.json``)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import librosa
import soundfile as sf
from huggingface_hub import hf_hub_download

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.dataset.commercial_seed import (  # noqa: E402
    assign_splits,
    hours_of,
    load_jsonl,
    summarize_by_source,
    write_jsonl,
)
from src.dataset.preprocess import reject_empty_transcripts  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DZ_REPO = "fares5462/darija-dz-tts-v1"
DZ_DIR = REPO / "data/seed/darija_dz_tts_v1"


def free_gb(path: Path | None = None) -> float:
    """Return free disk space in GiB for the volume containing ``path``."""
    st = os.statvfs(str(path or REPO))
    return (st.f_bavail * st.f_frsize) / (1024**3)


def _text_ok(text: str, *, min_chars: int = 4) -> bool:
    """Reject empty / tiny / non-letter garbage (empty transcripts are banned)."""
    t = (text or "").strip()
    if len(t) < min_chars:
        return False
    # Require at least one Arabic or Latin letter.
    return any("\u0600" <= c <= "\u06FF" or c.isalpha() for c in t)


def download_darija_dz(
    *,
    max_hours: float,
    min_free_gb: float,
    min_sec: float,
    max_sec: float,
) -> Tuple[Path, float, int]:
    """
    Download darija-dz-tts-v1 clips until ``max_hours`` or disk headroom stops us.

    Returns (manifest_path, hours, clip_count).
    """
    out_wav = DZ_DIR / "wavs"
    out_wav.mkdir(parents=True, exist_ok=True)
    meta_csv = Path(
        hf_hub_download(DZ_REPO, "metadata.csv", repo_type="dataset")
    )
    rows_out: List[Dict[str, Any]] = []
    total_sec = 0.0
    skipped = 0

    with open(meta_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if total_sec / 3600.0 >= max_hours:
                break
            if free_gb() < min_free_gb + 0.35:
                logger.warning(
                    "Stopping darija-dz download early: free=%.2f GiB < headroom+0.35",
                    free_gb(),
                )
                break
            rel = (row.get("file_name") or "").strip()
            text = (row.get("text") or "").strip()
            if not rel or not _text_ok(text):
                skipped += 1
                continue
            dest = out_wav / Path(rel).name
            try:
                if not dest.exists() or dest.stat().st_size < 1000:
                    local = hf_hub_download(DZ_REPO, rel, repo_type="dataset")
                    if not dest.exists():
                        shutil.copy2(local, dest)
                audio, sr = librosa.load(str(dest), sr=24000, mono=True)
                dur = float(len(audio) / sr)
                if dur < min_sec or dur > max_sec:
                    skipped += 1
                    continue
                # Normalize peak lightly; rewrite 24 kHz mono for F5 consistency.
                peak = float(max(abs(audio.max()), abs(audio.min()), 1e-8))
                audio = audio * (0.9 / peak)
                sf.write(str(dest), audio, 24000)
            except Exception as e:
                logger.warning("Skip %s: %s", rel, e)
                skipped += 1
                continue

            rows_out.append(
                {
                    "audio_filepath": str(dest),
                    "text": text,
                    "duration": round(dur, 3),
                    "speaker_id": "darija_dz_tts_v1_multi",
                    "dialect": "Algerian_Darja",
                    "dialect_note": "Casual Algerian Darja dialogue podcast-style clips",
                    "license": "unlisted_on_hub",
                    "allowed_use": "research_prototype_only",
                    "source": DZ_REPO,
                    "source_url": f"https://huggingface.co/datasets/{DZ_REPO}",
                    "transcript_status": "dataset_label",
                    "sample_rate": 24000,
                    "qc_pass": True,
                    "consent": "dataset_card_unclear_prototype",
                }
            )
            total_sec += dur
            if (i + 1) % 40 == 0:
                logger.info(
                    "darija-dz %d rows, %.2f min, free=%.2f GiB",
                    len(rows_out),
                    total_sec / 60.0,
                    free_gb(),
                )

    man = DZ_DIR / "manifest.jsonl"
    rows_out = reject_empty_transcripts(rows_out)
    write_jsonl(man, rows_out)
    hours = hours_of(rows_out)
    logger.info(
        "darija-dz manifest: %d clips / %.3f h (skipped %d) -> %s",
        len(rows_out),
        hours,
        skipped,
        man,
    )
    return man, hours, len(rows_out)


def _cap_hours(
    rows: Sequence[Dict[str, Any]],
    *,
    max_hours: float,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Deterministic subsample so total duration ≤ max_hours."""
    import random

    if max_hours <= 0:
        return []
    if hours_of(rows) <= max_hours + 1e-6:
        return list(rows)
    rng = random.Random(seed)
    pool = list(rows)
    rng.shuffle(pool)
    kept: List[Dict[str, Any]] = []
    sec = 0.0
    for r in pool:
        d = float(r.get("duration") or 0.0)
        if sec + d > max_hours * 3600.0 and kept:
            continue
        kept.append(r)
        sec += d
        if sec / 3600.0 >= max_hours:
            break
    return kept


def build_smart_manifest(
    *,
    cafe_manifest: Path,
    dvoice_manifest: Path,
    darija_manifest: Path,
    out: Path,
    moroccan_cap_hours: float,
    product_lines: Path | None,
) -> Dict[str, Any]:
    """
    Merge Algerian-first sources; cap Moroccan DVoice; ban empty text.

    Accepts both commercial CC-BY and research_prototype_only (Mac smart FT).
    """
    cafe = reject_empty_transcripts(load_jsonl(cafe_manifest))
    dz = reject_empty_transcripts(load_jsonl(darija_manifest)) if darija_manifest.exists() else []
    ma = reject_empty_transcripts(load_jsonl(dvoice_manifest))
    ma = _cap_hours(ma, max_hours=moroccan_cap_hours)

    # Drop rows missing rights; keep research_prototype + commercial.
    allowed = {
        "research_and_commercial_with_attribution",
        "research_prototype_only",
    }
    merged: List[Dict[str, Any]] = []
    for r in list(cafe) + list(dz) + list(ma):
        if not _text_ok(r.get("text") or ""):
            continue
        if r.get("allowed_use") not in allowed:
            continue
        if not r.get("qc_pass", True):
            continue
        for key in ("license", "allowed_use", "source"):
            if not str(r.get(key) or "").strip():
                raise ValueError(f"missing {key} on {r.get('audio_filepath')}")
        merged.append(r)

    holdout: List[str] = []
    if product_lines and product_lines.exists():
        holdout = json.loads(product_lines.read_text(encoding="utf-8"))
    merged = assign_splits(merged, val_fraction=0.08, listen_holdout_texts=holdout)
    write_jsonl(out, merged)

    algerian = [r for r in merged if "Algerian" in str(r.get("dialect") or "")]
    summary = {
        "mode": "mac_smart_algerian_prefer",
        "clips": len(merged),
        "hours_total": round(hours_of(merged), 4),
        "hours_train": round(hours_of([r for r in merged if r["split"] == "train"]), 4),
        "hours_val": round(hours_of([r for r in merged if r["split"] == "val"]), 4),
        "hours_algerian": round(hours_of(algerian), 4),
        "algerian_fraction": round(hours_of(algerian) / max(hours_of(merged), 1e-9), 3),
        "moroccan_cap_hours": moroccan_cap_hours,
        "by_source": summarize_by_source(merged),
        "free_gb_after": round(free_gb(), 2),
        "empty_transcripts": 0,
        "note": "Empty transcripts banned; darija-dz is research_prototype_only",
    }
    out.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Smart manifest: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-dz-hours", type=float, default=0.65)
    p.add_argument("--moroccan-cap-hours", type=float, default=0.30)
    p.add_argument("--min-free-gb", type=float, default=12.5)
    p.add_argument("--min-sec", type=float, default=1.5)
    p.add_argument("--max-sec", type=float, default=12.0)
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    free0 = free_gb()
    logger.info("Free disk before expand: %.2f GiB (min headroom %.2f)", free0, args.min_free_gb)
    if free0 < args.min_free_gb:
        logger.error("Already below min free headroom; refusing expand")
        return 2

    if not args.skip_download:
        download_darija_dz(
            max_hours=args.max_dz_hours,
            min_free_gb=args.min_free_gb,
            min_sec=args.min_sec,
            max_sec=args.max_sec,
        )
    else:
        logger.info("Skip download; using existing %s", DZ_DIR)

    summary = build_smart_manifest(
        cafe_manifest=REPO / "data/seed/processed/cafe_manifest_v2.jsonl",
        dvoice_manifest=REPO / "data/seed/dvoice_ary/manifest.jsonl",
        darija_manifest=DZ_DIR / "manifest.jsonl",
        out=REPO / "data/seed/smart_train_manifest.jsonl",
        moroccan_cap_hours=args.moroccan_cap_hours,
        product_lines=REPO / "data/seed/listen_set/product_lines.json",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if free_gb() < args.min_free_gb:
        logger.error("Post-expand free %.2f GiB < min %.2f", free_gb(), args.min_free_gb)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
