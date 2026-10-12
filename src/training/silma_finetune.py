"""
SILMA TTS (F5 Small Arabic) fine-tune on Apple MPS / CUDA — real CFM loss.

Init from ``silma-ai/silma-tts`` (``model.pt`` + SILMA vocab). Not SWivid Base.

Resume: every ``save_every`` updates writes ``model_last.pt`` + ``train_meta.json``
under ``ckpts/silma_darja/``. Pass ``--resume`` (or auto-detect last ckpt) after
sleep / disconnect to continue from ``updates`` already completed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.training.f5_finetune import (
    REPO_ROOT,
    prepare_commercial_manifest,
    prepare_f5_dataset_dir,
    select_train_device,
)
from src.dataset.commercial_seed import hours_of

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# SILMA V1 Small DiT (from silma-ai/silma-tts config.yaml)
SILMA_DIT_CFG = dict(
    dim=768,
    depth=18,
    heads=12,
    ff_mult=2,
    text_dim=512,
    text_mask_padding=True,
    conv_layers=4,
    pe_attn_head=None,
)

DEFAULT_SAVE_DIR = REPO_ROOT / "ckpts" / "silma_darja"
SILMA_HF = "silma-ai/silma-tts"


def resolve_silma_assets() -> Tuple[Path, Path]:
    """Download (or reuse cache) SILMA ``model.pt`` + ``vocab.txt``."""
    from huggingface_hub import hf_hub_download

    ckpt = Path(hf_hub_download(SILMA_HF, "model.pt"))
    vocab = Path(hf_hub_download(SILMA_HF, "vocab.txt"))
    local_vocab = REPO_ROOT / "data" / "f5_datasets" / "silma_vocab.txt"
    local_vocab.parent.mkdir(parents=True, exist_ok=True)
    local_vocab.write_text(vocab.read_text(encoding="utf-8"), encoding="utf-8")
    return ckpt, local_vocab


def _load_state_dict(path: Path) -> Dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    if str(path).endswith(".safetensors"):
        raw = load_file(str(path), device="cpu")
    else:
        raw = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        for key in ("ema_model_state_dict", "model_state_dict", "model", "state_dict"):
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break
    state: Dict[str, Any] = {}
    for k, v in raw.items():
        nk = k[len("ema_model.") :] if str(k).startswith("ema_model.") else k
        if str(nk).startswith("mel_spec."):
            continue
        state[nk] = v
    return state


def _write_meta(save_dir: Path, meta: Dict[str, Any]) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "train_meta.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_checkpoint(
    model,
    save_dir: Path,
    *,
    updates: int,
    losses: List[float],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Persist ``model_last.pt``, numbered ckpt, and ``train_meta.json``."""
    import torch

    save_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "updates": updates,
        "losses_tail": losses[-100:],
        "loss_start": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "init": "silma-ai/silma-tts",
        "arch": "SilmaTTS_V1_Small",
        **(extra or {}),
    }
    last = save_dir / "model_last.pt"
    # Atomic-ish replace: write temp then rename (resume-safe if lid closes mid-save).
    tmp = save_dir / "model_last.pt.tmp"
    torch.save(payload, tmp)
    tmp.replace(last)
    # Sparse numbered milestones under tight Mac disk (~650MB each).
    # Keep only the newest one so free headroom stays ≥12 GB.
    numbered_path = None
    if updates % 1000 == 0:
        numbered = save_dir / f"model_{updates}.pt"
        torch.save(payload, numbered)
        numbered_path = str(numbered)
        milestones = sorted(save_dir.glob("model_[0-9]*.pt"), key=lambda p: p.stat().st_mtime)
        for old in milestones[:-1]:
            try:
                old.unlink()
                logger.info("Removed old milestone %s (disk)", old.name)
            except OSError:
                pass
    meta = {
        "updates": updates,
        "loss_start": payload["loss_start"],
        "final_loss": payload["final_loss"],
        "losses_tail": payload["losses_tail"],
        "ckpt_last": str(last),
        "ckpt_numbered": numbered_path,
        "init": "silma-ai/silma-tts",
        "arch": "SilmaTTS_V1_Small",
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": extra.get("device") if extra else None,
        "hours_train": extra.get("hours_train") if extra else None,
        "max_updates_target": extra.get("max_updates") if extra else None,
        "resume_hint": (
            f"python -m src.training.silma_finetune --train-mps --resume "
            f"--max-updates {extra.get('max_updates', updates)} "
            f"--save-dir {save_dir}"
        ),
    }
    meta_path = _write_meta(save_dir, meta)
    # Best-effort refresh of user-facing live progress docs (Project store).
    try:
        import subprocess

        prog = REPO_ROOT / "scripts" / "write_train_progress.py"
        if prog.is_file():
            subprocess.run(
                [sys.executable, str(prog), "--meta", str(meta_path)],
                check=False,
                capture_output=True,
                timeout=30,
            )
    except Exception:
        pass
    logger.info("Saved SILMA ckpt updates=%d loss=%.4f -> %s", updates, losses[-1], last)
    return {"last": str(last), "numbered": numbered_path or "", "meta": str(meta_path)}


def find_resume_ckpt(save_dir: Path) -> Optional[Path]:
    last = save_dir / "model_last.pt"
    if last.is_file() and last.stat().st_size > 1_000_000:
        return last
    numbered = sorted(save_dir.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    for p in reversed(numbered):
        if p.name == "model_last.pt":
            continue
        if p.stat().st_size > 1_000_000:
            return p
    return None


def run_silma_mps_finetune(
    arrow_dir: Path,
    *,
    max_updates: int = 2000,
    batch_frames: int = 256,
    learning_rate: float = 1e-5,
    save_dir: Optional[Path] = None,
    save_every: int = 50,
    resume: bool = True,
    pretrain: Optional[str] = None,
    epochs: int = 20,
) -> Dict[str, Any]:
    """
    Real SILMA Small CFM fine-tune with periodic resume-safe checkpoints.
    """
    import torch
    import torchaudio
    from f5_tts.model import CFM, DiT
    from f5_tts.model.utils import get_tokenizer

    device = select_train_device()
    save_dir = Path(save_dir or DEFAULT_SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "ok": False,
        "device": device,
        "max_updates": max_updates,
        "batch_frames": batch_frames,
        "save_dir": str(save_dir),
        "init": SILMA_HF,
        "arch": "SilmaTTS_V1_Small",
    }
    if device == "cpu":
        report["error"] = "No CUDA/MPS — refusing full SILMA fine-tune on CPU"
        return report

    silma_ckpt, silma_vocab = resolve_silma_assets()
    # Prefer SILMA vocab for Arabic coverage; keep arrow vocab as fallback merge.
    vocab_file = silma_vocab
    if (arrow_dir / "vocab.txt").exists():
        # Ensure dataset chars are covered: merge arrow chars into SILMA vocab.
        base = silma_vocab.read_text(encoding="utf-8").splitlines()
        local = (arrow_dir / "vocab.txt").read_text(encoding="utf-8").splitlines()
        merged = list(dict.fromkeys(base + local))
        merged_path = arrow_dir / "vocab_silma_merged.txt"
        merged_path.write_text("\n".join(merged) + "\n", encoding="utf-8")
        vocab_file = merged_path
        report["vocab"] = str(vocab_file)
        report["vocab_size"] = len(merged)
    else:
        report["vocab"] = str(vocab_file)

    raw_arrow = arrow_dir / "raw.arrow"
    if not raw_arrow.exists():
        report["error"] = f"missing {raw_arrow}"
        return report

    import pyarrow as pa

    with pa.memory_map(str(raw_arrow), "r") as source:
        table = pa.ipc.RecordBatchFileReader(source).read_all()
        audio_paths = table.column("audio_path").to_pylist()
        texts = table.column("text").to_pylist()
        durations = [float(x) for x in table.column("duration").to_pylist()]

    pairs = [
        (ap, tx, dur)
        for ap, tx, dur in zip(audio_paths, texts, durations)
        if Path(ap).exists() and str(tx).strip() and 0.3 <= float(dur) <= 15.0
    ]
    if len(pairs) < 8:
        report["error"] = f"too few clips: {len(pairs)}"
        return report
    report["clips"] = len(pairs)
    report["hours"] = round(sum(p[2] for p in pairs) / 3600.0, 4)

    vocab_char_map, vocab_size = get_tokenizer(str(vocab_file), "custom")
    mel_spec_kwargs = dict(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
        mel_spec_type="vocos",
    )
    model = CFM(
        transformer=DiT(**SILMA_DIT_CFG, text_num_embeds=vocab_size, mel_dim=100),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    start_update = 0
    losses: List[float] = []
    resume_path = find_resume_ckpt(save_dir) if resume else None
    init_path = Path(pretrain) if pretrain else silma_ckpt

    if resume_path is not None:
        logger.info("Resuming from %s", resume_path)
        blob = torch.load(str(resume_path), map_location="cpu", weights_only=False)
        state = blob.get("model_state_dict") or blob
        if isinstance(state, dict) and "transformer.text_embed.text_embed.weight" in state:
            pass
        else:
            state = _load_state_dict(resume_path)
        # Resize text embed if vocab grew
        embed_key = "transformer.text_embed.text_embed.weight"
        target_rows = vocab_size + 1
        if embed_key in state and state[embed_key].shape[0] != target_rows:
            old = state[embed_key]
            new = old.new_zeros(target_rows, old.shape[1])
            n = min(old.shape[0], target_rows)
            new[:n] = old[:n]
            if target_rows > old.shape[0]:
                torch.nn.init.normal_(new[old.shape[0] :], mean=0.0, std=0.02)
            state[embed_key] = new
            report["text_embed_resized"] = [int(old.shape[0]), int(target_rows)]
        missing, unexpected = model.load_state_dict(state, strict=False)
        start_update = int(blob.get("updates", 0)) if isinstance(blob, dict) else 0
        losses = list(blob.get("losses_tail") or blob.get("losses") or [])
        report["resumed_from"] = str(resume_path)
        report["resume_updates"] = start_update
        report["load_missing"] = len(missing)
        report["load_unexpected"] = len(unexpected)
        init_path = resume_path
    else:
        logger.info("Cold start from SILMA weights %s", init_path)
        state = _load_state_dict(Path(init_path))
        embed_key = "transformer.text_embed.text_embed.weight"
        target_rows = vocab_size + 1
        if embed_key in state and state[embed_key].shape[0] != target_rows:
            old = state[embed_key]
            new = old.new_zeros(target_rows, old.shape[1])
            n = min(old.shape[0], target_rows)
            new[:n] = old[:n]
            if target_rows > old.shape[0]:
                torch.nn.init.normal_(new[old.shape[0] :], mean=0.0, std=0.02)
            state[embed_key] = new
            report["text_embed_resized"] = [int(old.shape[0]), int(target_rows)]
        missing, unexpected = model.load_state_dict(state, strict=False)
        report["load_missing"] = len(missing)
        report["load_unexpected"] = len(unexpected)
        report["pretrain"] = str(init_path)

    if start_update >= max_updates:
        report["ok"] = True
        report["updates"] = start_update
        report["note"] = "Already at or past max_updates; nothing to do"
        report["final_loss"] = losses[-1] if losses else None
        return report

    model = model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    hop = 256
    sr = 24000
    updates = start_update

    def _load_mel(path: str, max_sec: float = 8.0):
        wav, file_sr = torchaudio.load(path)
        wav = wav.mean(dim=0, keepdim=True)
        if file_sr != sr:
            wav = torchaudio.functional.resample(wav, file_sr, sr)
        max_samples = int(max_sec * sr)
        if wav.shape[-1] > max_samples:
            wav = wav[..., :max_samples]
        mel = model.mel_spec(wav.to(device))
        return mel.squeeze(0).permute(1, 0)

    extra = {
        "device": device,
        "hours_train": report["hours"],
        "max_updates": max_updates,
        "batch_frames": batch_frames,
        "learning_rate": learning_rate,
        "vocab": str(vocab_file),
        "pretrain": str(init_path),
    }
    # Persist a meta stub before the first step so lid-sleep still has resume hints.
    _write_meta(
        save_dir,
        {
            "updates": updates,
            "status": "running",
            "init": SILMA_HF,
            "arch": "SilmaTTS_V1_Small",
            "device": device,
            "hours_train": report["hours"],
            "max_updates_target": max_updates,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "resume_hint": (
                f"python -m src.training.silma_finetune --train-mps --resume "
                f"--max-updates {max_updates} --save-dir {save_dir}"
            ),
        },
    )

    rng = torch.Generator()
    rng.manual_seed(42 + updates)
    try:
        for epoch in range(epochs):
            order = torch.randperm(len(pairs), generator=rng).tolist()
            batch_mels: List[torch.Tensor] = []
            batch_texts: List[str] = []
            frame_count = 0
            for idx in order:
                if updates >= max_updates:
                    break
                ap, tx, _dur = pairs[idx]
                try:
                    mel = _load_mel(ap)
                except Exception as e:
                    logger.warning("skip %s: %s", ap, e)
                    continue
                t_frames = mel.shape[0]
                if t_frames < 16:
                    continue
                if batch_mels and frame_count + t_frames > batch_frames:
                    max_t = max(m.shape[0] for m in batch_mels)
                    n_mel = batch_mels[0].shape[1]
                    stacked = torch.zeros(len(batch_mels), max_t, n_mel, device=device)
                    for bi, m in enumerate(batch_mels):
                        stacked[bi, : m.shape[0]] = m
                    opt.zero_grad(set_to_none=True)
                    out = model(stacked, batch_texts)
                    loss = out[0] if isinstance(out, (tuple, list)) else out
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    losses.append(float(loss.detach().cpu()))
                    updates += 1
                    if updates % 10 == 0 or updates == start_update + 1:
                        logger.info(
                            "silma-mps update %d/%d epoch=%d loss=%.4f",
                            updates,
                            max_updates,
                            epoch,
                            losses[-1],
                        )
                    if updates % save_every == 0:
                        save_checkpoint(
                            model, save_dir, updates=updates, losses=losses, extra=extra
                        )
                    batch_mels, batch_texts, frame_count = [], [], 0
                    if device == "mps":
                        torch.mps.empty_cache()
                batch_mels.append(mel)
                batch_texts.append(tx)
                frame_count += t_frames
            if updates >= max_updates:
                break
            if batch_mels and updates < max_updates:
                max_t = max(m.shape[0] for m in batch_mels)
                n_mel = batch_mels[0].shape[1]
                stacked = torch.zeros(len(batch_mels), max_t, n_mel, device=device)
                for bi, m in enumerate(batch_mels):
                    stacked[bi, : m.shape[0]] = m
                opt.zero_grad(set_to_none=True)
                out = model(stacked, batch_texts)
                loss = out[0] if isinstance(out, (tuple, list)) else out
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu()))
                updates += 1
                if updates % save_every == 0:
                    save_checkpoint(
                        model, save_dir, updates=updates, losses=losses, extra=extra
                    )
    except KeyboardInterrupt:
        report["interrupted"] = True
        logger.warning("KeyboardInterrupt at update %d — saving resume ckpt", updates)
    except Exception as e:
        report["error"] = f"train loop failed: {e}"
        logger.exception("SILMA fine-tune failed")

    if updates > start_update:
        paths = save_checkpoint(
            model, save_dir, updates=updates, losses=losses, extra=extra
        )
        report["ckpt"] = paths["last"]
        report["meta"] = paths["meta"]

    report["ok"] = updates > start_update and "error" not in report
    report["updates"] = updates
    report["updates_this_run"] = updates - start_update
    report["final_loss"] = losses[-1] if losses else None
    report["loss_start"] = losses[0] if losses else None
    report["note"] = (
        "Real SILMA Small CFM fine-tune (not SWivid Base). "
        "Resume with --resume after sleep/disconnect."
    )
    (save_dir / "train_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="SILMA Small → Darja MPS/CUDA fine-tune")
    p.add_argument(
        "--manifests",
        nargs="+",
        default=[
            str(REPO_ROOT / "data/seed/processed/cafe_manifest_v2.jsonl"),
            str(REPO_ROOT / "data/seed/dvoice_ary/manifest.jsonl"),
        ],
    )
    p.add_argument(
        "--commercial-manifest",
        default=str(REPO_ROOT / "data/seed/commercial_train_manifest.jsonl"),
    )
    p.add_argument(
        "--prebuilt-manifest",
        action="store_true",
        help="Use --commercial-manifest as-is (smart Algerian seed; skip re-filter)",
    )
    p.add_argument(
        "--allow-research-prototype",
        action="store_true",
        help="Allow research_prototype_only rows (e.g. darija-dz-tts-v1)",
    )
    p.add_argument(
        "--product-lines",
        default=str(REPO_ROOT / "data/seed/listen_set/product_lines.json"),
    )
    p.add_argument("--dataset-root", default=str(REPO_ROOT / "data/f5_datasets"))
    p.add_argument("--dataset-name", default="darja_commercial")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--train-mps", action="store_true")
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--no-resume", action="store_true", help="Force cold start from SILMA")
    p.add_argument("--max-updates", type=int, default=2000)
    p.add_argument("--batch-frames", type=int, default=256)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--save-dir", default=str(DEFAULT_SAVE_DIR))
    p.add_argument("--pretrain", default=None, help="Override SILMA model.pt path")
    args = p.parse_args(argv)

    rows = prepare_commercial_manifest(
        args.manifests,
        Path(args.commercial_manifest),
        product_lines_json=args.product_lines if Path(args.product_lines).exists() else None,
        allow_research_prototype=bool(args.allow_research_prototype or args.prebuilt_manifest),
        prebuilt=bool(args.prebuilt_manifest),
    )
    train_h = hours_of([r for r in rows if r.get("split") == "train"])
    logger.info("Commercial train hours ≈ %.3f", train_h)
    arrow_dir = prepare_f5_dataset_dir(rows, Path(args.dataset_root), args.dataset_name)

    if args.prepare_only:
        # Still fetch SILMA assets so wake/resume has weights on disk.
        ckpt, vocab = resolve_silma_assets()
        logger.info("Prepared arrow=%s silma_ckpt=%s vocab=%s", arrow_dir, ckpt, vocab)
        return 0

    if not args.train_mps:
        logger.info("Nothing to do (pass --train-mps or --prepare-only)")
        return 0

    resume = args.resume or (not args.no_resume)
    report = run_silma_mps_finetune(
        arrow_dir,
        max_updates=args.max_updates,
        batch_frames=args.batch_frames,
        learning_rate=args.learning_rate,
        save_dir=Path(args.save_dir),
        save_every=args.save_every,
        resume=resume,
        pretrain=args.pretrain,
        epochs=args.epochs,
    )
    print(json.dumps({k: report[k] for k in report if k != "losses"}, indent=2))
    return 0 if report.get("ok") else 4


if __name__ == "__main__":
    raise SystemExit(main())
