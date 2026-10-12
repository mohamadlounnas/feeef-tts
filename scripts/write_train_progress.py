#!/usr/bin/env python3
"""Refresh Project-store train-progress.md from ckpts/silma_darja/train_meta.json."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_META = REPO / "ckpts" / "silma_darja" / "train_meta.json"
STORE_DOCS = Path(
    "/Users/mohamad/Library/Application Support/Cursor/AgentStores/"
    "cursor_agent_stores/bc-53f9e0f4-71e6-4778-b239-98f1504109bb/files/docs"
)
PROGRESS = STORE_DOCS / "train-progress.md"
STATUS = STORE_DOCS / "silma-ft-status.md"
SMART_SUMMARY = REPO / "data/seed/smart_train_manifest.summary.json"


def tmux_alive(session: str = "silma-ft") -> bool:
    return shutil.which("tmux") is not None and subprocess.run(
        ["tmux", "has-session", "-t", session],
        capture_output=True,
    ).returncode == 0


def train_proc_alive() -> bool:
    """True if a silma_finetune / caffeinate train python looks running."""
    try:
        out = subprocess.check_output(["pgrep", "-fl", "silma_finetune"], text=True)
        return bool(out.strip())
    except Exception:
        return False


def free_gb() -> float:
    st = os.statvfs(str(REPO))
    return (st.f_bavail * st.f_frsize) / (1024**3)


def load_hours() -> str:
    if SMART_SUMMARY.exists():
        try:
            s = json.loads(SMART_SUMMARY.read_text(encoding="utf-8"))
            return (
                f"{s.get('hours_train')} h train "
                f"(Algerian frac {s.get('algerian_fraction')}, "
                f"total {s.get('hours_total')} h / {s.get('clips')} clips)"
            )
        except Exception:
            pass
    return "~see train_meta / commercial summary"


def render(meta: dict) -> str:
    u = int(meta.get("updates") or 0)
    target = int(meta.get("max_updates_target") or meta.get("max_updates") or 2000)
    loss = meta.get("final_loss")
    pct = 100.0 * u / target if target else 0.0
    session_alive = tmux_alive()
    proc = train_proc_alive()
    running = "yes" if proc else ("tmux up, idle" if session_alive else "no")
    ts = meta.get("saved_at") or time.strftime("%Y-%m-%dT%H:%M:%S%z")
    ckpt = meta.get("ckpt_last") or str(REPO / "ckpts/silma_darja/model_last.pt")
    loss_s = f"{float(loss):.4f}" if loss is not None else "n/a"
    if u >= target and not proc:
        nxt = "Target reached — build `media/mac-smart-listen-set/` + status doc"
    elif proc:
        nxt = f"Training toward {target} (keep Mac awake / plugged)"
    else:
        nxt = f"Resume: tmux silma-ft + --resume --max-updates {target}"

    return f"""# Train progress (self-serve)

**Last updated:** {time.strftime("%Y-%m-%d %H:%M %z")}  
**Mode:** SILMA MPS fine-tune — Algerian-preferring smart continue  
**Status:** {"TRAINING" if proc else ("idle at target" if u >= target else "idle / not running")}

| Field | Value |
|---|---|
| Updates | **{u} / {target}** ({pct:.1f}%) |
| Loss | **{loss_s}** |
| Hours / seed | {load_hours()} |
| Device | {meta.get("device") or "mps"} |
| Ckpt | `{ckpt}` |
| Meta saved_at | {ts} |
| Train running | **{running}** |
| tmux | `silma-ft` (train) / `silma-progress` (watcher) |
| Disk free | **~{free_gb():.1f} GB** |

## Keep Mac awake

- Prefer **plugged in** + lid open (or clamshell with power + external display).
- Train launches under `caffeinate -dims` inside tmux `silma-ft`.
- Do **not** force sleep; closing the lid without power may stall MPS.
- Self-check: `tmux capture-pane -t silma-ft -p | tail -20`
- Meta: `cat ~/Desktop/projects/feeef-tts-m3/ckpts/silma_darja/train_meta.json`

**Next:** {nxt}
"""


def write_progress(meta_path: Path) -> dict:
    meta = (
        json.loads(meta_path.read_text(encoding="utf-8"))
        if meta_path.exists()
        else {
            "updates": 0,
            "max_updates_target": 2000,
            "final_loss": None,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    )
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(render(meta), encoding="utf-8")
    return meta


def patch_status_live(meta: dict) -> None:
    if not STATUS.exists():
        return
    text = STATUS.read_text(encoding="utf-8")
    u = int(meta.get("updates") or 0)
    target = int(meta.get("max_updates_target") or 2000)
    loss = meta.get("final_loss")
    loss_s = f"{float(loss):.4f}" if loss is not None else "n/a"
    alive = "yes" if train_proc_alive() else ("tmux only" if tmux_alive() else "no")
    ts = meta.get("saved_at") or time.strftime("%Y-%m-%dT%H:%M:%S%z")
    marker = "## Live train (honest)"
    if marker not in text:
        return
    before, rest = text.split(marker, 1)
    nxt = rest.find("\n## ")
    after = rest[nxt:] if nxt != -1 else ""
    live = f"""## Live train (honest)

| Item | Value |
| :--- | :--- |
| Session | tmux `silma-ft` — **alive {alive}** |
| Hours train | {load_hours()} |
| Progress | **{u} / {target}** ({(100*u/target if target else 0):.1f}%) |
| Last loss | **{loss_s}** |
| Ckpt | `/Users/mohamad/Desktop/projects/feeef-tts-m3/ckpts/silma_darja/model_last.pt` |
| last updated | **{ts}** |
| Usable for casual Algiers VO? | **Pending listen set** |

Also see always-current [`train-progress.md`](train-progress.md).

"""
    STATUS.write_text(before + live + after.lstrip("\n") if after else before + live, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--meta", default=str(DEFAULT_META))
    p.add_argument("--watch", action="store_true", help="Poll meta every N seconds")
    p.add_argument("--interval", type=float, default=30.0)
    args = p.parse_args()
    meta_path = Path(args.meta)

    def once() -> None:
        meta = write_progress(meta_path)
        patch_status_live(meta)
        print(
            f"updates={meta.get('updates')} loss={meta.get('final_loss')} "
            f"running={'yes' if train_proc_alive() else 'no'} -> {PROGRESS}"
        )

    once()
    if not args.watch:
        return 0

    while True:
        time.sleep(args.interval)
        once()
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            target = int(meta.get("max_updates_target") or 2000)
            u = int(meta.get("updates") or 0)
            # Exit watch only when target reached AND train process gone.
            if u >= target and not train_proc_alive():
                time.sleep(5)
                once()
                return 0
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
