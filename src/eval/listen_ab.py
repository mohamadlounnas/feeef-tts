"""
Fixed listen-set A/B harness: MMS-TTS-ara vs F5-TTS zero-shot.

Writes WAV pairs and a JSON score sheet template. Does not claim automatic
native/MOS scores — humans (or a later rater tool) fill the sheet.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * 0.9
    sf.write(str(path), audio.astype(np.float32), sr)


def run_listen_ab(
    listen_json: str,
    product_lines_json: str,
    output_dir: str,
    run_mms: bool = True,
    run_f5: bool = True,
    f5_ckpt: Optional[str] = None,
    f5_vocab: Optional[str] = None,
    max_lines: Optional[int] = None,
    f5_tag: str = "f5zs",
) -> Dict[str, Any]:
    """
    Synthesize product Darja lines with MMS and/or F5 (Darja reference clips).

    Each product line is paired with a rotating Casablanca/CAFE reference for F5.
    ``f5_tag`` controls output stem suffix (``f5zs`` zero-shot, ``f5ft`` fine-tuned).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(listen_json, encoding="utf-8") as f:
        refs: List[Dict[str, Any]] = json.load(f)
    with open(product_lines_json, encoding="utf-8") as f:
        lines: List[str] = json.load(f)
    if max_lines:
        lines = lines[:max_lines]

    mms_engine = None
    f5_engine = None
    if run_mms:
        from src.models.tts_engine import AlgerianTTSEngine

        mms_engine = AlgerianTTSEngine(backend_model="mms-ara", allow_harmonic_fallback=False)
    if run_f5:
        from src.models.tts_engine import AlgerianTTSEngine

        f5_engine = AlgerianTTSEngine(
            backend_model="f5tts",
            f5_ckpt=f5_ckpt,
            f5_vocab=f5_vocab,
            allow_harmonic_fallback=False,
        )

    results = []
    for i, text in enumerate(lines):
        ref = refs[i % len(refs)]
        item: Dict[str, Any] = {
            "id": f"P{i+1:02d}",
            "text": text,
            "ref_id": ref.get("id"),
            "ref_audio": ref.get("ref_audio"),
            "ref_text": ref.get("text"),
            "ref_license": ref.get("license"),
            "mms_wav": None,
            "f5_wav": None,
            "f5_tag": f5_tag,
            "f5_ckpt": f5_ckpt,
            "errors": [],
            "rater": {"native_feel": None, "msa_or_robotic": None, "prefer": None, "notes": ""},
        }
        stem = f"{item['id']}"

        if mms_engine is not None:
            try:
                res = mms_engine.synthesize(text)
                mms_path = out / f"{stem}_mms.wav"
                _write_wav(mms_path, res["audio"], res["sample_rate"])
                item["mms_wav"] = str(mms_path)
                item["mms_duration"] = res["duration"]
            except Exception as e:
                logger.exception("MMS failed for %s", stem)
                item["errors"].append(f"mms: {e}")

        if f5_engine is not None:
            try:
                res = f5_engine.synthesize(
                    text,
                    reference_audio_path=ref["ref_audio"],
                    reference_text=ref.get("text") or "",
                )
                f5_path = out / f"{stem}_{f5_tag}.wav"
                _write_wav(f5_path, res["audio"], res["sample_rate"])
                item["f5_wav"] = str(f5_path)
                item["f5_duration"] = res["duration"]
            except Exception as e:
                logger.exception("F5 failed for %s", stem)
                item["errors"].append(f"f5: {e}")

        # Also copy reference for ABX
        if ref.get("ref_audio") and os.path.exists(ref["ref_audio"]):
            ref_out = out / f"{stem}_ref.wav"
            audio, sr = sf.read(ref["ref_audio"])
            _write_wav(ref_out, np.asarray(audio), sr)
            item["ref_wav_copy"] = str(ref_out)

        results.append(item)
        logger.info("Done %s", stem)

    vs = "fine-tuned" if f5_tag == "f5ft" else "zero-shot"
    sheet = {
        "task": f"MMS (MSA baseline) vs F5-TTS {vs} with Darja reference",
        "f5_tag": f5_tag,
        "f5_ckpt": f5_ckpt,
        "instructions": (
            f"For each id, listen to *_mms.wav vs *_{f5_tag}.wav (and optional *_ref.wav). "
            "Score native_feel 1-5 (5=sounds like casual Algerian talking). "
            "Set msa_or_robotic true/false. prefer = 'mms'|'f5'|'neither'|'tie'."
        ),
        "items": results,
    }
    sheet_path = out / "score_sheet.json"
    with open(sheet_path, "w", encoding="utf-8") as f:
        json.dump(sheet, f, ensure_ascii=False, indent=2)
    logger.info("Wrote score sheet %s", sheet_path)
    return sheet


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Darja listen-set MMS vs F5 A/B")
    p.add_argument("--listen-json", required=True)
    p.add_argument("--product-lines", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--skip-mms", action="store_true")
    p.add_argument("--skip-f5", action="store_true")
    p.add_argument("--f5-ckpt", default=None)
    p.add_argument("--f5-vocab", default=None)
    p.add_argument("--max-lines", type=int, default=None)
    p.add_argument(
        "--f5-tag",
        default="f5zs",
        help="Output suffix: f5zs (zero-shot) or f5ft (fine-tuned ckpt)",
    )
    args = p.parse_args(argv)
    run_listen_ab(
        listen_json=args.listen_json,
        product_lines_json=args.product_lines,
        output_dir=args.output_dir,
        run_mms=not args.skip_mms,
        run_f5=not args.skip_f5,
        f5_ckpt=args.f5_ckpt,
        f5_vocab=args.f5_vocab,
        max_lines=args.max_lines,
        f5_tag=args.f5_tag,
    )


if __name__ == "__main__":
    main()
