#!/usr/bin/env python3
"""
Restore commercial train wavs for M3 on a machine that only has manifests.

Downloads:
  - DVoice Maghrebi clips listed in data/seed/dvoice_ary/manifest.jsonl
  - CAFE-small-no-overlap parents, then re-segments with the same VAD rules as
    assemble_cafe_asr_v2, attaching texts from cafe_manifest_v2 when stems match,
    otherwise using parent CSV text / empty-skip.

Rewrites absolute /workspace paths to the local repo root.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download, list_repo_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
DVOICE_REPO = "BrunoHays/DVOICEv2.0-Darija"
CAFE_REPO = "FatimahEmadEldin/cafe-algerian-codeswitch-speech"


def _rewrite_path(p: str, repo: Path) -> str:
    """Map old cloud absolute paths onto this checkout."""
    if p.startswith("/workspace/repos/feeef-tts/"):
        return str(repo / p[len("/workspace/repos/feeef-tts/") :])
    if not os.path.isabs(p):
        return str(repo / p)
    return p


def restore_dvoice(manifest_path: Path, repo: Path) -> int:
    rows = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    wav_dir = repo / "data/seed/dvoice_ary/wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    for i, row in enumerate(rows):
        basename = Path(row["audio_filepath"]).name
        dest = wav_dir / basename
        audio_id = row.get("audio_id") or basename.replace(".wav", "")
        # DVoice layout: data/train/<uuid>.wav — augment prefixes vary; try common roots.
        candidates = [
            f"data/train/{basename}",
            f"data/train/{audio_id}.wav",
            f"data/test/{basename}",
            f"data/dev/{basename}",
        ]
        if dest.exists() and dest.stat().st_size > 1000:
            row["audio_filepath"] = str(dest)
            ok += 1
            continue
        local = None
        for rel in candidates:
            try:
                local = hf_hub_download(DVOICE_REPO, rel, repo_type="dataset")
                break
            except Exception:
                continue
        if local is None:
            logger.warning("DVoice miss %s", basename)
            continue
        if not dest.exists():
            shutil.copy2(local, dest)
        row["audio_filepath"] = str(dest)
        ok += 1
        if (i + 1) % 25 == 0:
            logger.info("DVoice %d/%d", i + 1, len(rows))

    with open(manifest_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("DVoice restored %d/%d -> %s", ok, len(rows), wav_dir)
    return ok


def _cafe_basename_aliases(name: str) -> List[str]:
    """Undo M2 flatten prefixes (clean_/large_) for HF lookup."""
    stem = name
    aliases = [stem]
    for pref in ("clean_", "large_"):
        if stem.startswith(pref):
            aliases.append(stem[len(pref) :])
    # large_chunk_50v2_9.wav -> chunk_50v2_9.wav / chunk50v2_9.wav
    m = re.match(r"large_chunk_(.+)", stem)
    if m:
        aliases.append(f"chunk_{m.group(1)}")
        aliases.append(f"chunk{m.group(1)}")
    return list(dict.fromkeys(aliases))


def restore_cafe_segments(
    cafe_manifest: Path,
    out_wav: Path,
    repo: Path,
    target_minutes: float,
    min_sec: float = 2.0,
    max_sec: float = 10.0,
) -> int:
    """Download CAFE parents and rebuild segments; reuse old texts when possible."""
    old_by_name: Dict[str, dict] = {}
    if cafe_manifest.exists():
        for line in cafe_manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            old_by_name[Path(r["audio_filepath"]).name] = r

    files = list_repo_files(CAFE_REPO, repo_type="dataset")
    wavs = [f for f in files if f.endswith(".wav")]
    by_base: Dict[str, List[str]] = {}
    for f in wavs:
        by_base.setdefault(Path(f).name, []).append(f)

    # Prefer no-overlap / clean / small parents referenced by old manifest.
    parents = sorted({r.get("parent_file") for r in old_by_name.values() if r.get("parent_file")})
    cafe_raw = repo / "data/seed/cafe_raw"
    cafe_raw.mkdir(parents=True, exist_ok=True)

    # CSV parent texts as fallback
    text_map: Dict[str, str] = {}
    for csv_path in cafe_raw.glob("*.csv"):
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                base = os.path.basename(row.get("path") or "")
                text = (row.get("text") or row.get("transcription") or "").strip()
                if base and text:
                    text_map[base] = text

    downloaded: Dict[str, Path] = {}
    for parent in parents:
        aliases = _cafe_basename_aliases(parent)
        rel = None
        for a in aliases:
            if a in by_base:
                # Prefer no-overlap paths
                cands = by_base[a]
                cands = sorted(
                    cands,
                    key=lambda x: (
                        0 if "no-overlap" in x else 1 if "clean" in x else 2,
                        len(x),
                    ),
                )
                rel = cands[0]
                break
        if rel is None:
            continue
        dest = cafe_raw / Path(rel).name
        if not dest.exists():
            local = hf_hub_download(CAFE_REPO, rel, repo_type="dataset")
            shutil.copy2(local, dest)
        downloaded[parent] = dest
        # also index by dest name
        downloaded[dest.name] = dest

    logger.info("CAFE parents downloaded: %d / %d referenced", len(set(downloaded.values())), len(parents))

    out_wav.mkdir(parents=True, exist_ok=True)
    manifest: List[dict] = []
    total = 0.0
    target_cap = target_minutes * 60.0

    # If old segments exist as text map, try to reproduce by same split naming.
    for parent, wav_path in sorted({(p, downloaded[p]) for p in parents if p in downloaded}):
        if total >= target_cap:
            break
        audio, sr = librosa.load(str(wav_path), sr=24000, mono=True)
        peak = float(np.max(np.abs(audio)) or 1.0)
        audio = audio * (0.9 / peak)
        intervals = librosa.effects.split(audio, top_db=30)
        parent_text = text_map.get(wav_path.name, "") or text_map.get(parent, "")

        def emit(seg: np.ndarray, fname: str) -> None:
            nonlocal total
            seg_dur = len(seg) / sr
            if seg_dur < min_sec or seg_dur > max_sec:
                return
            old = old_by_name.get(fname)
            text = (old or {}).get("text") or parent_text
            if not text or not str(text).strip():
                return
            path = out_wav / fname
            sf.write(str(path), seg, sr)
            row = {
                "audio_filepath": str(path),
                "text": str(text).strip(),
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
                "transcript_status": (old or {}).get("transcript_status", "restored_or_parent_csv"),
                "parent_file": parent,
                "sample_rate": 24000,
                "split": "train",
                "qc_pass": True,
                "qc_reasons": [],
            }
            if old:
                for k in ("asr_model", "asr_avg_logprob", "asr_no_speech_prob", "asr_language"):
                    if k in old:
                        row[k] = old[k]
            manifest.append(row)
            total += seg_dur

        for idx, (start, end) in enumerate(intervals):
            if total >= target_cap:
                break
            dur = (end - start) / sr
            # Match M2 naming: prefer parent stem as stored in old manifest.
            stem = Path(parent).stem
            if min_sec <= dur <= max_sec:
                emit(audio[start:end], f"{stem}_seg_{idx:03d}.wav")
            elif dur > max_sec:
                max_samples = int(8.0 * sr)
                for sub in range(start, end, max_samples):
                    if total >= target_cap:
                        break
                    sub_end = min(sub + max_samples, end)
                    emit(audio[sub:sub_end], f"{stem}_seg_{idx:03d}_{sub}.wav")

        logger.info("%s: %.1f min so far, clips=%d", parent, total / 60, len(manifest))

    cafe_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(cafe_manifest, "w", encoding="utf-8") as f:
        for row in manifest:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("CAFE segments restored: %d clips / %.2f min", len(manifest), total / 60)
    return len(manifest)


def rewrite_commercial_manifest(path: Path, repo: Path) -> Tuple[int, int]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    exist = 0
    for r in rows:
        r["audio_filepath"] = _rewrite_path(r["audio_filepath"], repo)
        if Path(r["audio_filepath"]).exists():
            exist += 1
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return exist, len(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=str(REPO))
    p.add_argument("--cafe-minutes", type=float, default=25.0)
    p.add_argument("--skip-dvoice", action="store_true")
    p.add_argument("--skip-cafe", action="store_true")
    args = p.parse_args()
    repo = Path(args.repo)

    if not args.skip_dvoice:
        restore_dvoice(repo / "data/seed/dvoice_ary/manifest.jsonl", repo)
    if not args.skip_cafe:
        restore_cafe_segments(
            repo / "data/seed/processed/cafe_manifest_v2.jsonl",
            repo / "data/seed/processed/cafe_segments_v2",
            repo,
            target_minutes=args.cafe_minutes,
        )
    exist, n = rewrite_commercial_manifest(repo / "data/seed/commercial_train_manifest.jsonl", repo)
    logger.info("commercial_train_manifest audio exist %d/%d (will rebuild via f5_finetune prep)", exist, n)


if __name__ == "__main__":
    main()
