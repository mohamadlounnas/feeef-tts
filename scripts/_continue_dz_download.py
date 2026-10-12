#!/usr/bin/env python3
"""Continue darija-dz download under disk headroom."""
from __future__ import annotations
import csv, json, os, shutil, sys
from pathlib import Path
import librosa, soundfile as sf
from huggingface_hub import hf_hub_download

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.dataset.commercial_seed import write_jsonl, hours_of
from src.dataset.preprocess import reject_empty_transcripts

DZ = "fares5462/darija-dz-tts-v1"
out_wav = REPO / "data/seed/darija_dz_tts_v1/wavs"
out_wav.mkdir(parents=True, exist_ok=True)
man_path = REPO / "data/seed/darija_dz_tts_v1/manifest.jsonl"
existing = {}
if man_path.exists():
    for line in man_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            existing[Path(r["audio_filepath"]).name] = r

def free_gb() -> float:
    st = os.statvfs(str(REPO))
    return (st.f_bavail * st.f_frsize) / (1024**3)

def text_ok(t: str) -> bool:
    t = (t or "").strip()
    return len(t) >= 4 and any("\u0600" <= c <= "\u06FF" or c.isalpha() for c in t)

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
meta = Path(hf_hub_download(DZ, "metadata.csv", repo_type="dataset"))
rows = list(existing.values())
total = sum(float(r["duration"]) for r in rows)
print(f"start clips={len(rows)} hours={total/3600:.3f} free={free_gb():.2f}", flush=True)
target_h = float(os.environ.get("DZ_TARGET_H", "0.50"))
min_free = float(os.environ.get("MIN_FREE_GB", "13.0"))
skipped = 0
with open(meta, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if total / 3600 >= target_h:
            break
        if free_gb() < min_free:
            print("stop disk", free_gb(), flush=True)
            break
        rel = (row.get("file_name") or "").strip()
        text = (row.get("text") or "").strip()
        if not rel or not text_ok(text):
            skipped += 1
            continue
        name = Path(rel).name
        if name in existing:
            continue
        dest = out_wav / name
        try:
            local = hf_hub_download(DZ, rel, repo_type="dataset")
            if not dest.exists():
                try:
                    os.link(local, dest)
                except OSError:
                    shutil.copy2(local, dest)
            audio, _sr = librosa.load(str(dest), sr=24000, mono=True)
            dur = len(audio) / 24000.0
            if dur < 1.5 or dur > 12.0:
                skipped += 1
                continue
            peak = float(max(abs(audio.max()), abs(audio.min()), 1e-8))
            sf.write(str(dest), audio * (0.9 / peak), 24000)
            r = {
                "audio_filepath": str(dest),
                "text": text,
                "duration": round(dur, 3),
                "speaker_id": "darija_dz_tts_v1_multi",
                "dialect": "Algerian_Darja",
                "dialect_note": "Casual Algerian Darja dialogue podcast-style clips",
                "license": "unlisted_on_hub",
                "allowed_use": "research_prototype_only",
                "source": DZ,
                "source_url": f"https://huggingface.co/datasets/{DZ}",
                "transcript_status": "dataset_label",
                "sample_rate": 24000,
                "qc_pass": True,
                "consent": "dataset_card_unclear_prototype",
            }
            rows.append(r)
            existing[name] = r
            total += dur
        except Exception as e:
            print("skip", rel, e, flush=True)
            skipped += 1
        if len(rows) % 40 == 0:
            print(f"progress clips={len(rows)} h={total/3600:.3f} free={free_gb():.2f}", flush=True)
            write_jsonl(man_path, rows)  # checkpoint manifest

rows = reject_empty_transcripts(rows)
write_jsonl(man_path, rows)
print(json.dumps({"clips": len(rows), "hours": round(hours_of(rows), 4), "skipped": skipped, "free_gb": round(free_gb(), 2)}, indent=2), flush=True)
