"""
Commercial train-seed helpers for Algerian Darja TTS.

Hard filter: only ``research_and_commercial_with_attribution`` (CC-BY-class)
rows may enter shipping training manifests. Casablanca NC-ND stays eval/ref.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.dataset.preprocess import reject_empty_transcripts, require_rights_fields

logger = logging.getLogger(__name__)

COMMERCIAL_ALLOWED_USE = "research_and_commercial_with_attribution"
EVAL_ONLY_ALLOWED_USE = "research_eval_only"
# Mac smart FT / research_scale may include prototype Algerian dialogue corpora.
RESEARCH_PROTOTYPE_ALLOWED_USE = "research_prototype_only"
TRAIN_ALLOWED_USES = frozenset(
    {
        COMMERCIAL_ALLOWED_USE,
        RESEARCH_PROTOTYPE_ALLOWED_USE,
    }
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_commercial(
    rows: Iterable[Dict[str, Any]],
    *,
    allow_research_prototype: bool = False,
) -> List[Dict[str, Any]]:
    """Keep train-eligible rows with non-empty text and rights fields."""
    allowed = TRAIN_ALLOWED_USES if allow_research_prototype else {COMMERCIAL_ALLOWED_USE}
    kept: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("allowed_use") not in allowed:
            continue
        require_rights_fields(row)
        kept.append(row)
    return reject_empty_transcripts(kept)


def hours_of(rows: Sequence[Dict[str, Any]]) -> float:
    return sum(float(r.get("duration") or 0.0) for r in rows) / 3600.0


def assign_splits(
    rows: List[Dict[str, Any]],
    *,
    val_fraction: float = 0.08,
    listen_holdout_texts: Optional[Sequence[str]] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Assign train/val/listen splits.

    Listen holdout: any row whose text exactly matches a product listen line,
    plus a small random val slice from the remainder. Never train on listen.
    """
    import random

    holdout = {t.strip() for t in (listen_holdout_texts or []) if t and t.strip()}
    out: List[Dict[str, Any]] = []
    pool: List[Dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        text = (r.get("text") or "").strip()
        if text in holdout:
            r["split"] = "listen"
            out.append(r)
        else:
            pool.append(r)

    rng = random.Random(seed)
    rng.shuffle(pool)
    n_val = max(1, int(len(pool) * val_fraction)) if pool else 0
    for i, r in enumerate(pool):
        r["split"] = "val" if i < n_val else "train"
        out.append(r)
    return out


def summarize_by_source(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        src = r.get("source") or "unknown"
        b = buckets.setdefault(
            src,
            {
                "source": src,
                "clips": 0,
                "hours": 0.0,
                "license": r.get("license"),
                "allowed_use": r.get("allowed_use"),
                "dialect": r.get("dialect"),
            },
        )
        b["clips"] += 1
        b["hours"] += float(r.get("duration") or 0.0) / 3600.0
    for b in buckets.values():
        b["hours"] = round(b["hours"], 4)
    return sorted(buckets.values(), key=lambda x: -x["hours"])


def export_f5_csv_wavs(
    rows: Sequence[Dict[str, Any]],
    out_dir: Path,
    *,
    split: str = "train",
    copy_audio: bool = True,
) -> Tuple[Path, int, float]:
    """
    Export F5-TTS csv+wavs layout:

        out_dir/
          metadata.csv   # relative_path|text
          wavs/*.wav
    """
    out_dir = Path(out_dir)
    wavs = out_dir / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    selected = [r for r in rows if r.get("split", "train") == split]
    selected = reject_empty_transcripts(selected)

    lines: List[str] = []
    total_sec = 0.0
    for i, row in enumerate(selected):
        src = Path(row["audio_filepath"])
        if not src.exists():
            logger.warning("Missing audio, skip: %s", src)
            continue
        dest_name = f"{split}_{i:05d}_{src.stem}.wav"
        dest = wavs / dest_name
        if copy_audio:
            if not dest.exists() or dest.stat().st_size == 0:
                shutil.copy2(src, dest)
        else:
            # symlink when possible
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(src.resolve())
        text = (row["text"] or "").replace("|", " ").replace("\n", " ").strip()
        lines.append(f"wavs/{dest_name}|{text}")
        total_sec += float(row.get("duration") or 0.0)

    meta = out_dir / "metadata.csv"
    meta.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    logger.info(
        "F5 csv/wavs: %d clips, %.2f min -> %s",
        len(lines),
        total_sec / 60.0,
        out_dir,
    )
    return meta, len(lines), total_sec
