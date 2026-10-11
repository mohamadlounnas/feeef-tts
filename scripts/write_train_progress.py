#!/usr/bin/env python3
"""Refresh Project-store train-progress.md from ckpts/silma_darja/train_meta.json."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_META = REPO / "ckpts" / "silma_darja" / "train_meta.json"
# Cursor Project store (feeef-arabic-tts)
STORE_DOCS = Path(
    "/Users/mohamad/Library/Application Support/Cursor/AgentStores/"
    "cursor_agent_stores/bc-53f9e0f4-71e6-4778-b239-98f1504109bb/files/docs"
)
PROGRESS = STORE_DOCS / "train-progress.md"
STATUS = STORE_DOCS / "silma-ft-status.md"


def tmux_alive(session: str = "silma-ft") -> bool:
    return shutil.which("tmux") is not None and __import__("subprocess").run(
        ["tmux", "has-session", "-t", session],
        capture_output=True,
    ).returncode == 0


def render(meta: dict) -> str:
    u = int(meta.get("updates") or 0)
    target = int(meta.get("max_updates_target") or 2000)
    loss = meta.get("final_loss")
    pct = 100.0 * u / target if target else 0.0
    alive = "yes" if tmux_alive() else "no"
    ts = meta.get("saved_at") or time.strftime("%Y-%m-%dT%H:%M:%S%z")
    ckpt = meta.get("ckpt_last") or str(REPO / "ckpts/silma_darja/model_last.pt")
    if u >= target:
        nxt = "Train done — rebuild/listen silma-ft listen set + native-rate judgment"
    else:
        nxt = f"Continue MPS train toward {target} (tmux silma-ft / --resume)"
    loss_s = f"{float(loss):.4f}" if loss is not None else "n/a"
    return (
        "# SILMA train progress (live)\n\n"
        "| Field | Value |\n"
        "| :--- | :--- |\n"
        f"| updates | **{u} / {target}** |\n"
        f"| loss | **{loss_s}** |\n"
        f"| % | **{pct:.1f}%** |\n"
        f"| tmux `silma-ft` | **{alive}** |\n"
        f"| last updated | **{ts}** |\n"
        f"| next | {nxt} |\n\n"
        f"Ckpt: `{ckpt}`\n"
    )


def write_progress(meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {
        "updates": 0,
        "max_updates_target": 2000,
        "final_loss": None,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text(render(meta), encoding="utf-8")
    return meta


def patch_status_live(meta: dict) -> None:
    """Best-effort refresh of the Live train table in silma-ft-status.md."""
    if not STATUS.exists():
        return
    text = STATUS.read_text(encoding="utf-8")
    u = int(meta.get("updates") or 0)
    target = int(meta.get("max_updates_target") or 2000)
    loss = meta.get("final_loss")
    loss_s = f"{float(loss):.4f}" if loss is not None else "n/a"
    alive = "yes" if tmux_alive() else "no"
    ts = meta.get("saved_at") or time.strftime("%Y-%m-%dT%H:%M:%S%z")
    marker = "## Live train (honest)"
    if marker not in text:
        return
    before, rest = text.split(marker, 1)
    # keep everything after the next ## if present
    rest_body = rest
    nxt = rest_body.find("\n## ")
    after = rest_body[nxt:] if nxt != -1 else ""
    live = f"""## Live train (honest)

| Item | Value |
| :--- | :--- |
| Session | tmux `silma-ft` — **alive {alive}** |
| Hours train | **~0.91 h** (CAFE DZ ~0.33 + DVoice MA ~0.66) |
| Progress | **{u} / {target}** ({100*u/target:.1f}%) |
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
            f"tmux={'yes' if tmux_alive() else 'no'} -> {PROGRESS}"
        )

    once()
    if not args.watch:
        return 0
    last_mtime = meta_path.stat().st_mtime if meta_path.exists() else 0.0
    while True:
        time.sleep(args.interval)
        mtime = meta_path.stat().st_mtime if meta_path.exists() else last_mtime
        once()
        # stop watching shortly after target reached and process idle
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if int(meta.get("updates") or 0) >= int(meta.get("max_updates_target") or 2000):
                if not tmux_alive() or True:
                    # one more write then exit after short grace
                    time.sleep(5)
                    once()
                    return 0
        except Exception:
            pass
        last_mtime = mtime


if __name__ == "__main__":
    raise SystemExit(main())
