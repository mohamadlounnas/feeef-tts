"""Tests for commercial seed filter, ASR QC heuristics, and F5 prep wiring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from src.dataset.asr_qc import AsrResult, light_qc
from src.dataset.commercial_seed import (
    COMMERCIAL_ALLOWED_USE,
    assign_splits,
    export_f5_csv_wavs,
    filter_commercial,
    hours_of,
)
from src.training.f5_finetune import build_finetune_command, estimate_gpu_needs


def test_filter_excludes_casablanca_ncnd():
    rows = [
        {
            "audio_filepath": "a.wav",
            "text": "واش راك",
            "duration": 2.0,
            "license": "CC-BY-4.0",
            "allowed_use": COMMERCIAL_ALLOWED_USE,
            "source": "cafe",
        },
        {
            "audio_filepath": "b.wav",
            "text": "صح السلاح",
            "duration": 3.0,
            "license": "CC-BY-NC-ND-4.0",
            "allowed_use": "research_eval_only",
            "source": "casablanca",
        },
        {
            "audio_filepath": "c.wav",
            "text": "",
            "duration": 2.0,
            "license": "CC-BY-4.0",
            "allowed_use": COMMERCIAL_ALLOWED_USE,
            "source": "cafe",
        },
    ]
    kept = filter_commercial(rows)
    assert len(kept) == 1
    assert kept[0]["source"] == "cafe"
    assert hours_of(kept) == 2.0 / 3600.0


def test_assign_splits_holdout_listen():
    rows = [
        {
            "audio_filepath": f"{i}.wav",
            "text": t,
            "duration": 2.0,
            "license": "CC-BY-4.0",
            "allowed_use": COMMERCIAL_ALLOWED_USE,
            "source": "x",
        }
        for i, t in enumerate(["واش راك؟ لبّاس؟", "مرحبا", "يعطيك الصحة", "فتح التطبيق", "بزاف شباب"])
    ]
    out = assign_splits(rows, val_fraction=0.25, listen_holdout_texts=["واش راك؟ لبّاس؟"])
    splits = {r["text"]: r["split"] for r in out}
    assert splits["واش راك؟ لبّاس؟"] == "listen"
    assert "train" in splits.values()


def test_light_qc_rejects_garbage():
    ok, reasons = light_qc("واش راك خويا", 3.0)
    assert ok
    bad, reasons = light_qc("....", 3.0)
    assert not bad
    assert "non_linguistic" in reasons or "too_short_text" in reasons
    asr = AsrResult("music", "en", -2.0, 0.9, 1.0, "test")
    bad2, reasons2 = light_qc("music", 3.0, asr, min_ar_ratio=0.25)
    assert not bad2


def test_export_f5_csv_wavs(tmp_path):
    wav = tmp_path / "src.wav"
    sr = 24000
    sf.write(str(wav), 0.1 * np.sin(2 * np.pi * 220 * np.linspace(0, 2, sr * 2, endpoint=False)), sr)
    rows = [
        {
            "audio_filepath": str(wav),
            "text": "واش راك",
            "duration": 2.0,
            "split": "train",
            "license": "CC-BY-4.0",
            "allowed_use": COMMERCIAL_ALLOWED_USE,
            "source": "unit",
        }
    ]
    meta, n, sec = export_f5_csv_wavs(rows, tmp_path / "f5raw", split="train")
    assert n == 1
    assert meta.exists()
    assert "|" in meta.read_text(encoding="utf-8")
    assert (tmp_path / "f5raw" / "wavs").exists()


def test_build_finetune_command_has_real_module():
    cmd = build_finetune_command("darja_commercial", tokenizer_path="/tmp/vocab.txt")
    assert "f5_tts.train.finetune_cli" in " ".join(cmd)
    assert "--finetune" in cmd
    est = estimate_gpu_needs(0.5, epochs=50)
    assert "recommended_gpu" in est
