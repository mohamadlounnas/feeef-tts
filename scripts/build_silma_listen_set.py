#!/usr/bin/env python3
"""
Build silma-ft listen set: P01–P12 ref / silma_zs / silma_ft / mms + L01 same-words.

Uses F5TTS_v1_Small (SILMA arch) — not SWivid Base.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
STORE_MEDIA = Path(
    "/Users/mohamad/Library/Application Support/Cursor/AgentStores/"
    "cursor_agent_stores/bc-53f9e0f4-71e6-4778-b239-98f1504109bb/files/media/silma-ft-listen-set"
)
# Prefer smart-seed merged vocab when present (mac-smart FT); else commercial.
DEFAULT_VOCAB_FT = REPO / "data/f5_datasets/darja_smart_char/vocab_silma_merged.txt"
if not DEFAULT_VOCAB_FT.exists():
    DEFAULT_VOCAB_FT = REPO / "data/f5_datasets/darja_commercial_char/vocab_silma_merged.txt"


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * 0.9
    sf.write(str(path), audio, sr)


def _export_infer_ckpt(src: Path, dest: Path, *, use_ema_blob: bool) -> Path:
    """Write a lean checkpoint load_checkpoint can open with weights_only=True."""
    blob = torch.load(str(src), map_location="cpu", weights_only=False)
    if use_ema_blob:
        if "ema_model_state_dict" in blob:
            out = {"ema_model_state_dict": blob["ema_model_state_dict"]}
        else:
            out = {"ema_model_state_dict": blob.get("model_state_dict") or blob}
    else:
        sd = blob.get("model_state_dict") or blob
        out = {"model_state_dict": sd}
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dest)
    return dest


def _load_f5_small(ckpt: Path, vocab: Path, *, use_ema: bool, device: str):
    from f5_tts.api import F5TTS

    return F5TTS(
        model="F5TTS_v1_Small",
        ckpt_file=str(ckpt),
        vocab_file=str(vocab),
        use_ema=use_ema,
        device=device,
    )


def _infer(f5, ref_wav: str, ref_text: str, gen_text: str, speed: float = 1.0):
    wav, sr, _ = f5.infer(
        ref_file=ref_wav,
        ref_text=ref_text,
        gen_text=gen_text,
        speed=speed,
    )
    return np.asarray(wav, dtype=np.float32), int(sr)


def _duration_ratio(synth: np.ndarray, sr: int, ref_path: Path) -> float:
    ref, ref_sr = sf.read(str(ref_path), always_2d=False)
    ref = np.asarray(ref, dtype=np.float32)
    if ref.ndim > 1:
        ref = ref.mean(axis=1)
    ref_dur = len(ref) / float(ref_sr)
    syn_dur = len(synth) / float(sr)
    return syn_dur / ref_dur if ref_dur > 0 else float("nan")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--listen-json", default=str(REPO / "data/seed/listen_set/listen_set.json"))
    p.add_argument("--product-lines", default=str(REPO / "data/seed/listen_set/product_lines.json"))
    p.add_argument("--out", default=str(STORE_MEDIA))
    p.add_argument("--silma-pt", default=str(REPO / "ckpts/silma_weights/model.pt"))
    p.add_argument(
        "--ft-pt",
        "--ckpt",
        dest="ft_pt",
        default=str(REPO / "ckpts/silma_darja/model_last.pt"),
        help="Fine-tuned SILMA checkpoint (alias: --ckpt)",
    )
    p.add_argument(
        "--vocab-ft",
        default=str(DEFAULT_VOCAB_FT),
        help="Vocab used during fine-tune (merged SILMA+local)",
    )
    p.add_argument(
        "--vocab-zs",
        default=str(REPO / "data/f5_datasets/silma_vocab.txt"),
        help="Original SILMA vocab (must match pretrained embed rows)",
    )
    p.add_argument("--max-lines", type=int, default=12)
    p.add_argument("--skip-mms", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    refs: List[Dict[str, Any]] = json.loads(Path(args.listen_json).read_text(encoding="utf-8"))
    lines: List[str] = json.loads(Path(args.product_lines).read_text(encoding="utf-8"))[: args.max_lines]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    vocab_ft = Path(args.vocab_ft)
    vocab_zs = Path(args.vocab_zs)
    if not vocab_zs.exists():
        from huggingface_hub import hf_hub_download

        vocab_zs = Path(hf_hub_download("silma-ai/silma-tts", "vocab.txt"))
        Path(REPO / "data/f5_datasets/silma_vocab.txt").write_text(
            vocab_zs.read_text(encoding="utf-8"), encoding="utf-8"
        )
        vocab_zs = REPO / "data/f5_datasets/silma_vocab.txt"
    if not vocab_ft.exists():
        vocab_ft = vocab_zs

    zs_ckpt = _export_infer_ckpt(Path(args.silma_pt), out / "_ckpts" / "silma_zs_infer.pt", use_ema_blob=True)
    ft_ckpt = _export_infer_ckpt(Path(args.ft_pt), out / "_ckpts" / "silma_ft_infer.pt", use_ema_blob=False)
    # Pull train_meta when present so score_sheet matches this ckpt (e.g. 7000 / 0.4554).
    train_meta: Dict[str, Any] = {}
    meta_path = Path(args.ft_pt).resolve().parent / "train_meta.json"
    if meta_path.exists():
        try:
            train_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            train_meta = {}

    logger.info("Loading SILMA zero-shot (F5TTS_v1_Small + ema, SILMA vocab)…")
    f5_zs = _load_f5_small(zs_ckpt, vocab_zs, use_ema=True, device=device)
    logger.info("Loading SILMA-ft (merged vocab)…")
    f5_ft = _load_f5_small(ft_ckpt, vocab_ft, use_ema=False, device=device)

    mms = None
    if not args.skip_mms:
        import sys

        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from src.models.tts_engine import AlgerianTTSEngine

        mms = AlgerianTTSEngine(backend_model="mms-ara", allow_harmonic_fallback=False)

    sheet: List[Dict[str, Any]] = []
    # Product lines
    for i, text in enumerate(lines):
        ref = refs[i % len(refs)]
        pid = f"P{i+1:02d}"
        ref_wav = Path(ref["ref_audio"])
        ref_text = ref["text"]
        # copy ref
        ref_out = out / f"{pid}_ref.wav"
        shutil.copy2(ref_wav, ref_out)
        item: Dict[str, Any] = {
            "id": pid,
            "text": text,
            "ref_id": ref.get("id"),
            "ref_audio": str(ref_out),
            "ref_text": ref_text,
            "ref_license": ref.get("license"),
        }
        try:
            wav, sr = _infer(f5_zs, str(ref_wav), ref_text, text)
            p_zs = out / f"{pid}_silma_zs.wav"
            _write_wav(p_zs, wav, sr)
            item["silma_zs"] = str(p_zs)
            item["silma_zs_duration"] = float(len(wav) / sr)
            item["silma_zs_duration_ratio_vs_ref"] = _duration_ratio(wav, sr, ref_wav)
        except Exception as e:
            item["silma_zs_error"] = str(e)
            logger.exception("zs %s", pid)
        try:
            wav, sr = _infer(f5_ft, str(ref_wav), ref_text, text)
            p_ft = out / f"{pid}_silma_ft.wav"
            _write_wav(p_ft, wav, sr)
            item["silma_ft"] = str(p_ft)
            item["silma_ft_duration"] = float(len(wav) / sr)
            item["silma_ft_duration_ratio_vs_ref"] = _duration_ratio(wav, sr, ref_wav)
        except Exception as e:
            item["silma_ft_error"] = str(e)
            logger.exception("ft %s", pid)
        if mms is not None:
            try:
                res = mms.synthesize(text)
                p_mms = out / f"{pid}_mms.wav"
                _write_wav(p_mms, res["audio"], res["sample_rate"])
                item["mms"] = str(p_mms)
                item["mms_duration"] = res["duration"]
            except Exception as e:
                item["mms_error"] = str(e)
        sheet.append(item)
        logger.info("done %s", pid)

    # L01 same-words reconstruct
    l01 = refs[0]
    ref_wav = Path(l01["ref_audio"])
    ref_text = l01["text"]
    shutil.copy2(ref_wav, out / "LISTEN_L01_ref_REAL.wav")
    recon: Dict[str, Any] = {"id": "L01", "text": ref_text, "ref": str(out / "LISTEN_L01_ref_REAL.wav")}
    for tag, eng, use_path in (
        ("silma_zs", f5_zs, out / "LISTEN_L01_silma_zs_SAME_WORDS.wav"),
        ("silma_ft", f5_ft, out / "LISTEN_L01_silma_ft_SAME_WORDS.wav"),
    ):
        try:
            wav, sr = _infer(eng, str(ref_wav), ref_text, ref_text)
            _write_wav(use_path, wav, sr)
            recon[tag] = str(use_path)
            recon[f"{tag}_duration_ratio"] = _duration_ratio(wav, sr, ref_wav)
        except Exception as e:
            recon[f"{tag}_error"] = str(e)
            logger.exception("L01 %s", tag)

    # Light rate judgment helper (not MOS)
    judgments = []
    for item in sheet:
        zs_r = item.get("silma_zs_duration_ratio_vs_ref")
        ft_r = item.get("silma_ft_duration_ratio_vs_ref")
        judgments.append(
            {
                "id": item["id"],
                "zs_ratio": zs_r,
                "ft_ratio": ft_r,
                "ft_near_natural_rate": bool(ft_r and 0.75 <= float(ft_r) <= 1.35),
                "paper_reading_suspect_ft": bool(ft_r and float(ft_r) > 1.5),
            }
        )
    l01_ft_r = recon.get("silma_ft_duration_ratio")
    summary = {
        "train_updates": int(train_meta.get("updates") or 0) or None,
        "hours_train": train_meta.get("hours_train"),
        "final_loss": train_meta.get("final_loss"),
        "ckpt": str(Path(args.ft_pt).resolve()),
        "l01_ft_duration_ratio": l01_ft_r,
        "l01_zs_duration_ratio": recon.get("silma_zs_duration_ratio"),
        "product_ft_near_natural": sum(1 for j in judgments if j["ft_near_natural_rate"]),
        "product_ft_paper_suspect": sum(1 for j in judgments if j["paper_reading_suspect_ft"]),
        "usable_for_voiceover_provisional": (
            "maybe_improved_vs_zs_but_not_product_claim"
            if (l01_ft_r and 0.75 <= float(l01_ft_r) <= 1.35)
            else "still_not_claiming_usable_VO"
        ),
        "note": (
            "Auto duration_ratio only — not native MOS. "
            "Native listen still required. "
            "Paper-reading = ratio ≫ 1 on same-words / product lines."
        ),
    }

    (out / "score_sheet.json").write_text(
        json.dumps({"product": sheet, "l01_recon": recon, "rate_judgment": judgments, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "README.md").write_text(
        "# SILMA-ft listen set\n\n"
        "Per line: `P##_ref.wav`, `P##_silma_zs.wav`, `P##_silma_ft.wav`, `P##_mms.wav`\n\n"
        "Hear first: `LISTEN_L01_ref_REAL.wav` vs `LISTEN_L01_silma_ft_SAME_WORDS.wav` "
        "(and `_silma_zs_`). Judge natural pace vs paper-reading.\n",
        encoding="utf-8",
    )
    logger.info("Wrote listen set -> %s", out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
