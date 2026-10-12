#!/bin/zsh
# Poll train_meta; when updates >= MIN_UPDATES (default 2500) or train idle at target,
# build listen set into Project store media/mac-smart-listen-set/.
set -euo pipefail
export PATH="/opt/homebrew/bin:/bin:/usr/bin:/usr/sbin:/sbin:$PATH"
cd /Users/mohamad/Desktop/projects/feeef-tts-m3
source .venv/bin/activate

MIN_UPDATES="${MIN_UPDATES:-2500}"
META=ckpts/silma_darja/train_meta.json
STORE_MEDIA="/Users/mohamad/Library/Application Support/Cursor/AgentStores/cursor_agent_stores/bc-53f9e0f4-71e6-4778-b239-98f1504109bb/files/media/mac-smart-listen-set"
STATUS="/Users/mohamad/Library/Application Support/Cursor/AgentStores/cursor_agent_stores/bc-53f9e0f4-71e6-4778-b239-98f1504109bb/files/docs/mac-smart-train-status.md"

echo "Waiting for updates>=$MIN_UPDATES or train complete..."
while true; do
  if [[ -f "$META" ]]; then
    eval "$(python - <<'PY'
import json
m=json.load(open("ckpts/silma_darja/train_meta.json"))
print(f"U={int(m.get('updates') or 0)}")
print(f"T={int(m.get('max_updates_target') or 7000)}")
PY
)"
    running=0
    pgrep -f 'src.training.silma_finetune' >/dev/null && running=1 || running=0
    echo "$(date) updates=$U/$T running=$running"
    if (( U >= MIN_UPDATES )) || { (( U >= T )) && (( running == 0 )); }; then
      break
    fi
  fi
  sleep 60
done

# Avoid fighting MPS while train still running unless target done
if pgrep -f 'src.training.silma_finetune' >/dev/null; then
  echo "Train still running at u=$U — defer listen until idle or target"
  while pgrep -f 'src.training.silma_finetune' >/dev/null; do
    sleep 90
  done
fi

mkdir -p "$STORE_MEDIA"
echo "Building listen set -> $STORE_MEDIA"
python scripts/build_silma_listen_set.py \
  --out "$STORE_MEDIA" \
  --ckpt ckpts/silma_darja/model_last.pt \
  2>&1 | tee -a logs/mac_smart_listen.log

# Patch status
if [[ -f "$STATUS" ]]; then
  python - <<PY
from pathlib import Path
p = Path("$STATUS")
t = p.read_text(encoding="utf-8")
note = "\n\n## Listen set (written)\n\nPath: \`media/mac-smart-listen-set/\` — built after train idle.\n"
if "Listen set (written)" not in t:
    p.write_text(t.rstrip() + note, encoding="utf-8")
PY
fi
echo "LISTEN_DONE $(date)"
