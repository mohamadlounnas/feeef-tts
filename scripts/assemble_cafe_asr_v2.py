#!/usr/bin/env python3
"""
CAFE Algerian segments with improved ASR (faster-whisper small) + light QC.

Replaces M1 Whisper-tiny drafts for commercial train seed.
Parent long-form CSV transcripts are retained as hints only (not 1:1 aligned).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from src.dataset.asr_qc import SeedAsrTranscriber, qc_row

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cafe-dir", default="data/seed/cafe_raw")
    p.add_argument("--out-wav", default="data/seed/processed/cafe_segments_v2")
    p.add_argument("--out-manifest", default="data/seed/processed/cafe_manifest_v2.jsonl")
    p.add_argument("--model-size", default="small")
    p.add_argument("--target-minutes", type=float, default=45.0)
    p.add_argument("--min-sec", type=float, default=2.0)
    p.add_argument("--max-sec", type=float, default=10.0)
    args = p.parse_args()

    cafe_dir = Path(args.cafe_dir)
    out_wav = Path(args.out_wav)
    out_wav.mkdir(parents=True, exist_ok=True)

    text_map = {}
    csvs = list(cafe_dir.glob("*.csv"))
    if csvs:
        with open(csvs[0], newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                base = os.path.basename(row.get("path") or "")
                text_map[base] = (row.get("text") or "").strip()

    asr = SeedAsrTranscriber(model_size=args.model_size, device="cpu", compute_type="int8")
    manifest = []
    total = 0.0
    target_cap = args.target_minutes * 60.0
    rejected = 0

    for wav_path in sorted(cafe_dir.glob("*.wav")):
        if total >= target_cap:
            break
        parent_text = text_map.get(wav_path.name, "")
        audio, sr = librosa.load(str(wav_path), sr=24000, mono=True)
        peak = float(np.max(np.abs(audio)) or 1.0)
        audio = audio * (0.9 / peak)
        intervals = librosa.effects.split(audio, top_db=30)

        def emit(seg: np.ndarray, fname: str) -> None:
            nonlocal total, rejected
            seg_dur = len(seg) / sr
            if seg_dur < args.min_sec or seg_dur > args.max_sec:
                return
            seg16 = librosa.resample(seg, orig_sr=sr, target_sr=16000)
            result = asr.transcribe_array(seg16)
            text = result.text
            path = out_wav / fname
            row = {
                "audio_filepath": str(path),
                "text": text,
                "duration": round(seg_dur, 3),
                "speaker_id": "cafe_dz_multi",
                "dialect": "Algerian_Darja",
                "dialect_note": (
                    "CAFE spontaneous Algerian Arabic/French/English code-switch; "
                    "multi-speaker; not single Algiers talent"
                ),
                "license": "CC-BY-4.0",
                "allowed_use": "research_and_commercial_with_attribution",
                "source": "FatimahEmadEldin/cafe-algerian-codeswitch-speech",
                "source_url": "https://huggingface.co/datasets/FatimahEmadEldin/cafe-algerian-codeswitch-speech",
                "upstream": "https://doi.org/10.5281/zenodo.16964503",
                "transcript_status": f"{result.model_name}_draft",
                "parent_file": wav_path.name,
                "parent_transcript_available": bool(parent_text),
                "sample_rate": 24000,
                "split": "train",
            }
            row = qc_row(row, result)
            if not row["qc_pass"]:
                rejected += 1
                logger.debug("QC reject %s: %s", fname, row["qc_reasons"])
                return
            sf.write(str(path), seg, sr)
            manifest.append(row)
            total += seg_dur

        for idx, (start, end) in enumerate(intervals):
            if total >= target_cap:
                break
            dur = (end - start) / sr
            if args.min_sec <= dur <= args.max_sec:
                emit(audio[start:end], f"{wav_path.stem}_seg_{idx:03d}.wav")
            elif dur > args.max_sec:
                max_samples = int(8.0 * sr)
                for sub in range(start, end, max_samples):
                    if total >= target_cap:
                        break
                    sub_end = min(sub + max_samples, end)
                    emit(audio[sub:sub_end], f"{wav_path.stem}_seg_{idx:03d}_{sub}.wav")

        logger.info("%s: %.1f min so far, clips=%d rejected=%d", wav_path.name, total / 60, len(manifest), rejected)

    man_path = Path(args.out_manifest)
    man_path.parent.mkdir(parents=True, exist_ok=True)
    with open(man_path, "w", encoding="utf-8") as f:
        for e in manifest:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    logger.info(
        "CAFE v2: %d clips, %.2f min (rejected %d) -> %s",
        len(manifest),
        total / 60,
        rejected,
        man_path,
    )


if __name__ == "__main__":
    main()
