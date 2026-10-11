#!/usr/bin/env python3
"""
Optional Maghrebi commercial supplement: Moroccan Darija Wiki Audio (CC-BY-4.0).

Dialect gap is documented on every row — NOT Algerian. Used only to grow
commercial Maghrebi hours when Algerian CC-BY speech is scarce.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO = "atlasia/Moroccan-Darija-Wiki-Audio-Dataset"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/seed/moroccan_wiki_ary")
    p.add_argument("--target-minutes", type=float, default=20.0)
    args = p.parse_args()

    out = Path(args.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)

    files = list_repo_files(REPO, repo_type="dataset")
    # Prefer metadata + wavs if present
    meta_candidates = [f for f in files if f.endswith((".csv", ".json", ".jsonl", ".tsv"))]
    audio_files = [f for f in files if f.lower().endswith((".wav", ".flac", ".mp3"))]
    logger.info("meta=%s audio_files=%d", meta_candidates[:10], len(audio_files))

    # Try datasets loader first
    rows = []
    total = 0.0
    try:
        from datasets import load_dataset
        import soundfile as sf
        import numpy as np

        ds = load_dataset(REPO, split="train")
        for i, ex in enumerate(ds):
            if total >= args.target_minutes * 60:
                break
            text = (
                (ex.get("text") or ex.get("sentence") or ex.get("transcription") or "")
            ).strip()
            if not text or len(text) < 3:
                continue
            audio = ex.get("audio")
            if not audio:
                continue
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])
            dur = len(arr) / sr
            if dur < 2.0 or dur > 12.0:
                continue
            fname = f"wiki_{i:05d}.wav"
            path = out / "wavs" / fname
            sf.write(str(path), arr, sr)
            rows.append(
                {
                    "audio_filepath": str(path),
                    "text": text,
                    "duration": round(dur, 3),
                    "speaker_id": "moroccan_wiki_ary",
                    "dialect": "Moroccan_Darija",
                    "dialect_note": (
                        "NOT Algerian; CC-BY Maghrebi wiki-read speech. "
                        "Dialect gap vs Algiers Darja documented."
                    ),
                    "license": "CC-BY-4.0",
                    "allowed_use": "research_and_commercial_with_attribution",
                    "source": REPO,
                    "source_url": f"https://huggingface.co/datasets/{REPO}",
                    "transcript_status": "dataset_label",
                    "sample_rate": sr,
                    "split": "train",
                }
            )
            total += dur
    except Exception as e:
        logger.error("datasets load failed: %s — falling back to file copy if possible", e)
        # Fallback: download a handful of wavs without text is useless; abort honestly.
        if not rows:
            logger.error("Could not assemble Moroccan wiki with transcripts; skipping.")
            (out / "manifest.jsonl").write_text("", encoding="utf-8")
            return

    man = out / "manifest.jsonl"
    with open(man, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Moroccan wiki: %d clips, %.1f min -> %s", len(rows), total / 60, man)


if __name__ == "__main__":
    main()
