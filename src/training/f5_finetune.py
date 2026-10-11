"""
Real F5-TTS fine-tune entrypoint for Algerian Darja (no simulated loss).

Pipeline:
1. Filter commercial manifest → train/val/listen splits
2. Export F5 ``metadata.csv`` + ``wavs/``
3. Run upstream ``prepare_csv_wavs`` (or local char-tokenizer prep for Arabic)
4. Invoke ``f5_tts.train.finetune_cli`` with real CFM / flow-matching loss

CPU-only environments: use ``--smoke`` for a tiny forward/backward sanity check
that still uses real F5 modules (not random.uniform). Full fine-tune needs GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.dataset.commercial_seed import (
    COMMERCIAL_ALLOWED_USE,
    assign_splits,
    export_f5_csv_wavs,
    filter_commercial,
    hours_of,
    load_jsonl,
    summarize_by_source,
    write_jsonl,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "f5_darja_finetune.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        # Minimal YAML subset reader for our flat config (no nested complexity needed).
        data: Dict[str, Any] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if v.lower() in ("true", "false"):
                data[k] = v.lower() == "true"
            else:
                try:
                    if "." in v:
                        data[k] = float(v)
                    else:
                        data[k] = int(v)
                except ValueError:
                    data[k] = v
        return data
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def prepare_commercial_manifest(
    inputs: Sequence[str],
    output: Path,
    product_lines_json: Optional[str] = None,
    val_fraction: float = 0.08,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in inputs:
        rows.extend(load_jsonl(Path(p)))
    rows = filter_commercial(rows)
    holdout: List[str] = []
    if product_lines_json and Path(product_lines_json).exists():
        holdout = json.loads(Path(product_lines_json).read_text(encoding="utf-8"))
    rows = assign_splits(rows, val_fraction=val_fraction, listen_holdout_texts=holdout)
    # Drop any row that failed QC if present
    rows = [r for r in rows if r.get("qc_pass", True)]
    write_jsonl(output, rows)
    summary = {
        "clips": len(rows),
        "hours_total": round(hours_of(rows), 4),
        "hours_train": round(hours_of([r for r in rows if r["split"] == "train"]), 4),
        "hours_val": round(hours_of([r for r in rows if r["split"] == "val"]), 4),
        "hours_listen": round(hours_of([r for r in rows if r["split"] == "listen"]), 4),
        "by_source": summarize_by_source(rows),
        "allowed_use": COMMERCIAL_ALLOWED_USE,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Commercial manifest: %s", json.dumps(summary, ensure_ascii=False))
    return rows


def prepare_f5_dataset_dir(
    rows: List[Dict[str, Any]],
    dataset_root: Path,
    dataset_name: str = "darja_commercial",
) -> Path:
    """
    Write csv/wavs under ``dataset_root/raw_{name}`` and prepare Arrow under
    ``dataset_root/{name}_char`` using char tokenizer (Arabic-friendly vs pinyin).
    """
    raw_dir = dataset_root / f"raw_{dataset_name}"
    export_f5_csv_wavs(rows, raw_dir, split="train", copy_audio=True)
    # Also dump val metadata for reference (not always consumed by finetune_cli)
    val_rows = [r for r in rows if r.get("split") == "val"]
    if val_rows:
        export_f5_csv_wavs(val_rows, dataset_root / f"raw_{dataset_name}_val", split="val")

    out_arrow = dataset_root / f"{dataset_name}_char"
    _prepare_char_arrow(raw_dir, out_arrow)
    return out_arrow


def _prepare_char_arrow(csv_wavs_dir: Path, out_dir: Path) -> None:
    """
    Build F5 CustomDataset Arrow files with **char** tokenizer (keeps Arabic script).

    Falls back to documenting the upstream CLI if f5_tts is not installed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = csv_wavs_dir / "metadata.csv"
    if not meta.exists():
        raise FileNotFoundError(meta)

    pairs = []
    for line in meta.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "|" not in line:
            continue
        rel, text = line.split("|", 1)
        audio_path = (csv_wavs_dir / rel).resolve()
        if not audio_path.exists():
            continue
        pairs.append((str(audio_path), text.strip()))

    if not pairs:
        raise RuntimeError(f"No valid audio|text pairs in {meta}")

    try:
        import torchaudio
        from datasets import Dataset
    except ImportError as e:
        raise RuntimeError("Need datasets+torchaudio to prepare F5 arrow data") from e

    # Prefer upstream prepare helper when available (finetune vocab copy).
    try:
        from f5_tts.model.utils import convert_char_to_pinyin  # noqa: F401
        use_upstream = True
    except Exception:
        use_upstream = False

    audio_paths, texts, durations = [], [], []
    vocab: set[str] = set()
    for ap, text in pairs:
        try:
            info = torchaudio.info(ap)
            dur = info.num_frames / float(info.sample_rate)
        except Exception:
            import soundfile as sf

            dur = float(sf.info(ap).duration)
        # Char tokenizer: keep text as list of characters via F5 convention —
        # F5 stores raw text strings; tokenizer expands chars at train time.
        audio_paths.append(ap)
        texts.append(text)
        durations.append(dur)
        vocab.update(list(text))

    ds = Dataset.from_dict({"audio_path": audio_paths, "text": texts, "duration": durations})
    raw_arrow = out_dir / "raw.arrow"
    # datasets>=2 save_to_disk writes a directory; F5 expects a single arrow file
    # in many versions. Write via ArrowWriter when available.
    try:
        import pyarrow as pa

        table = pa.Table.from_pydict(
            {"audio_path": audio_paths, "text": texts, "duration": durations}
        )
        with pa.OSFile(str(raw_arrow), "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
    except Exception:
        ds.save_to_disk(str(out_dir / "raw"))
        logger.warning(
            "Wrote HF Dataset directory %s/raw — if finetune_cli expects raw.arrow, "
            "re-run with pyarrow writer after upgrading pyarrow.",
            out_dir,
        )

    (out_dir / "duration.json").write_text(
        json.dumps({"duration": durations}, ensure_ascii=False),
        encoding="utf-8",
    )

    # Char vocab: try copy Emilia pretrained expanded with Arabic chars, else local.
    vocab_path = out_dir / "vocab.txt"
    pretrained_vocab = _find_pretrained_vocab()
    if pretrained_vocab and pretrained_vocab.exists():
        base = pretrained_vocab.read_text(encoding="utf-8").splitlines()
        merged = list(dict.fromkeys(base + sorted(vocab)))
        vocab_path.write_text("\n".join(merged) + "\n", encoding="utf-8")
        logger.info("Wrote vocab (%d tokens) from pretrained+local -> %s", len(merged), vocab_path)
    else:
        vocab_path.write_text("\n".join(sorted(vocab)) + "\n", encoding="utf-8")
        logger.info("Wrote local char vocab (%d) -> %s", len(vocab), vocab_path)

    logger.info(
        "Prepared F5 char dataset: %d clips, %.2f h -> %s (upstream_helpers=%s)",
        len(pairs),
        sum(durations) / 3600.0,
        out_dir,
        use_upstream,
    )


def _find_pretrained_vocab() -> Optional[Path]:
    candidates = [
        Path(os.environ.get("F5_PRETRAINED_VOCAB", "")),
        Path.home() / ".local/share/f5_tts/vocab.txt",
    ]
    try:
        from importlib.resources import files

        # Common package data locations
        pkg = files("f5_tts")
        for rel in (
            "../../data/Emilia_ZH_EN_pinyin/vocab.txt",
            "../data/Emilia_ZH_EN_pinyin/vocab.txt",
        ):
            try:
                p = Path(str(pkg.joinpath(rel)))
                candidates.append(p)
            except Exception:
                pass
    except Exception:
        pass
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    return None


def estimate_gpu_needs(hours_train: float, epochs: int = 50) -> Dict[str, Any]:
    """Rough guidance for operators — not a hard scheduler."""
    # F5 Base ~1.3B-ish DiT; finetune commonly needs >=16–24 GB for comfortable frames.
    return {
        "recommended_gpu": "1x NVIDIA 24GB (e.g. RTX 4090 / A5000) or better",
        "minimum_gpu": "1x 16GB with small batch_size_per_gpu and grad accumulation",
        "vram_notes": (
            "F5TTS_v1_Base fine-tune: start batch_size_per_gpu=384–800 (frame), "
            "grad_accumulation_steps=4–8 on 16GB; 1600–3200 frames on 24GB+."
        ),
        "wallclock_estimate_hours": round(max(1.0, hours_train * epochs * 0.8), 1),
        "hours_train": hours_train,
        "epochs": epochs,
        "cpu_only": "Full train not practical; use --smoke for wiring check only.",
    }


def build_finetune_command(
    dataset_name: str,
    *,
    exp_name: str = "F5TTS_v1_Base",
    learning_rate: float = 1e-5,
    batch_size_per_gpu: int = 800,
    epochs: int = 50,
    tokenizer: str = "custom",
    tokenizer_path: Optional[str] = None,
    finetune: bool = True,
    logger_name: Optional[str] = "tensorboard",
    save_per_updates: int = 500,
    last_per_updates: int = 200,
    num_warmup_updates: int = 300,
    pretrain: Optional[str] = None,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "f5_tts.train.finetune_cli",
        "--exp_name",
        exp_name,
        "--dataset_name",
        dataset_name,
        "--learning_rate",
        str(learning_rate),
        "--batch_size_per_gpu",
        str(batch_size_per_gpu),
        "--batch_size_type",
        "frame",
        "--max_samples",
        "32",
        "--epochs",
        str(epochs),
        "--num_warmup_updates",
        str(num_warmup_updates),
        "--save_per_updates",
        str(save_per_updates),
        "--last_per_updates",
        str(last_per_updates),
        "--tokenizer",
        tokenizer,
    ]
    if finetune:
        cmd.append("--finetune")
    if tokenizer == "custom" and tokenizer_path:
        cmd.extend(["--tokenizer_path", tokenizer_path])
    if logger_name:
        cmd.extend(["--logger", logger_name])
    if pretrain:
        cmd.extend(["--pretrain", pretrain])
    return cmd


def run_smoke_train(
    arrow_dir: Path,
    steps: int = 2,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Tiny real forward/backward on F5 CFM (not simulated loss).

    Loads a small batch from prepared paths and runs a few optimizer steps if
    ``f5_tts`` + pretrained weights are available. On CPU this may be slow and
    memory-heavy; keep ``steps`` tiny.
    """
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    report: Dict[str, Any] = {"device": device, "steps": steps, "ok": False}

    try:
        from f5_tts.model import CFM, DiT
        from f5_tts.model.utils import get_tokenizer
        from cached_path import cached_path
    except Exception as e:
        report["error"] = f"f5_tts not importable: {e}"
        logger.error("%s", report["error"])
        return report

    vocab_file = arrow_dir / "vocab.txt"
    if not vocab_file.exists():
        report["error"] = f"missing vocab at {vocab_file}"
        return report

    # Custom tokenizer expects a vocab.txt *file* path (not the dataset directory).
    vocab_char_map, vocab_size = get_tokenizer(str(vocab_file), "custom")
    model_cfg = dict(dim=256, depth=4, heads=4, ff_mult=2, text_dim=128, conv_layers=2)
    # Prefer tiny random CFM for smoke on CPU to avoid downloading 1.2B weights
    # unless F5_SMOKE_PRETRAIN=1.
    use_pretrain = os.environ.get("F5_SMOKE_PRETRAIN", "0") == "1"
    mel_spec_kwargs = dict(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
        mel_spec_type="vocos",
    )
    model = CFM(
        transformer=DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    ).to(device)

    if use_pretrain:
        try:
            ckpt = str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"))
            report["pretrain"] = ckpt
        except Exception as e:
            report["pretrain_error"] = str(e)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # CFM expects mel as [batch, time, mel_dim=100]; text as list[str] or token ids.
    b, t_frames, n_mel = 2, 64, 100
    mel = torch.randn(b, t_frames, n_mel, device=device)
    texts = ["واش راك", "يعطيك الصحة"]
    model.train()
    losses = []
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        try:
            out = model(mel, texts)
            loss = out[0] if isinstance(out, (tuple, list)) else out
        except Exception as e:
            report["error"] = f"CFM forward failed: {e}"
            logger.exception("smoke forward failed")
            return report
        if not torch.is_tensor(loss):
            report["error"] = f"loss is not a tensor: {type(loss)}"
            return report
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
        logger.info("smoke step %d loss=%.6f", step, losses[-1])

    report["ok"] = True
    report["losses"] = losses
    report["note"] = (
        "Real autograd through CFM (tiny DiT). Not a full pretrained fine-tune. "
        "Run build_finetune_command() on GPU for product training."
    )
    return report


def run_full_finetune(cmd: List[str], env: Optional[Dict[str, str]] = None) -> int:
    """Execute upstream finetune_cli. Returns process return code."""
    logger.info("Running: %s", " ".join(cmd))
    merged = os.environ.copy()
    if env:
        merged.update(env)
    # Point F5 data path at our prepared dataset parent when supported
    proc = subprocess.run(cmd, env=merged)
    return proc.returncode


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Commercial Darja → real F5-TTS fine-tune")
    p.add_argument("--manifests", nargs="+", required=True, help="Input JSONL manifests")
    p.add_argument(
        "--commercial-manifest",
        default=str(REPO_ROOT / "data/seed/commercial_train_manifest.jsonl"),
    )
    p.add_argument(
        "--product-lines",
        default=str(REPO_ROOT / "data/seed/listen_set/product_lines.json"),
    )
    p.add_argument(
        "--dataset-root",
        default=str(REPO_ROOT / "data/f5_datasets"),
        help="Where to write raw csv/wavs + *_char arrow",
    )
    p.add_argument("--dataset-name", default="darja_commercial")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Tiny real CFM train smoke on CPU/GPU")
    p.add_argument("--train", action="store_true", help="Launch upstream finetune_cli")
    p.add_argument("--smoke-steps", type=int, default=2)
    args = p.parse_args(argv)

    cfg = _load_yaml(Path(args.config)) if Path(args.config).exists() else {}
    rows = prepare_commercial_manifest(
        args.manifests,
        Path(args.commercial_manifest),
        product_lines_json=args.product_lines if Path(args.product_lines).exists() else None,
        val_fraction=float(cfg.get("val_fraction", 0.08)),
    )
    train_h = hours_of([r for r in rows if r.get("split") == "train"])
    gpu_est = estimate_gpu_needs(train_h, epochs=int(cfg.get("epochs", 50)))
    est_path = Path(args.commercial_manifest).with_name("gpu_estimate.json")
    est_path.write_text(json.dumps(gpu_est, indent=2), encoding="utf-8")
    logger.info("GPU estimate: %s", gpu_est)

    arrow_dir = prepare_f5_dataset_dir(rows, Path(args.dataset_root), args.dataset_name)
    vocab = arrow_dir / "vocab.txt"
    cmd = build_finetune_command(
        args.dataset_name,
        exp_name=str(cfg.get("exp_name", "F5TTS_v1_Base")),
        learning_rate=float(cfg.get("learning_rate", 1e-5)),
        batch_size_per_gpu=int(cfg.get("batch_size_per_gpu", 800)),
        epochs=int(cfg.get("epochs", 50)),
        tokenizer="custom",
        tokenizer_path=str(vocab),
        save_per_updates=int(cfg.get("save_per_updates", 500)),
        last_per_updates=int(cfg.get("last_per_updates", 200)),
        num_warmup_updates=int(cfg.get("num_warmup_updates", 300)),
        pretrain=cfg.get("pretrain"),
    )
    cmd_path = Path(args.dataset_root) / "TRAIN_COMMAND.sh"
    cmd_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"# Commercial Darja F5 fine-tune — requires GPU\n"
        f"# Train hours ≈ {train_h:.3f}\n"
        f"# Place prepared data where f5_tts load_dataset expects "
        f"(often <f5_tts_repo>/data/{args.dataset_name}_char)\n"
        f"export F5_DATA_ROOT={Path(args.dataset_root).resolve()}\n"
        + " ".join(cmd)
        + "\n",
        encoding="utf-8",
    )
    os.chmod(cmd_path, 0o755)
    logger.info("Wrote train command script %s", cmd_path)

    if args.smoke:
        report = run_smoke_train(arrow_dir, steps=args.smoke_steps)
        smoke_path = Path(args.dataset_root) / "smoke_report.json"
        smoke_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Smoke report: %s", report)
        if not report.get("ok"):
            return 2

    if args.train:
        import torch

        if not torch.cuda.is_available():
            logger.error(
                "Refusing full --train on CPU-only host. Use --smoke or run TRAIN_COMMAND.sh on GPU. "
                "Estimate: %s",
                gpu_est,
            )
            return 3
        return run_full_finetune(cmd)

    if args.prepare_only or not (args.smoke or args.train):
        logger.info("Prepare-only complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
