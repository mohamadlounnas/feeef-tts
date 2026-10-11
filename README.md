# feeef-tts — Algerian Darja TTS (pipeline foundation)

**Status (M2):** Commercial-train-ready seed filter + improved CAFE ASR/QC + **real F5-TTS fine-tune entrypoint** (upstream CFM loss, no simulated training).  
**Product goal:** casual everyday Algiers Darja voiceover — *not* pro studio VO, *not* MSA news voice.

This branch adapts the Jules `algerian-darja-tts-framework` scaffold. Simulated fine-tune loops and “multi-backend already works” claims are removed.

## What actually works today

| Component | Reality |
| :--- | :--- |
| `facebook/mms-tts-ara` | Real MSA Arabic VITS baseline (smoke / A-side). **Not** Darja. |
| F5-TTS inference | Optional (`pip install f5-tts`); zero-shot clone with a Darja reference clip. |
| F5-TTS fine-tune | Real entrypoint: `python -m src.training.f5_finetune` → prepares csv/wavs + char arrow, writes `TRAIN_COMMAND.sh`, optional `--smoke` CFM autograd check. Full `--train` requires GPU. |
| Preprocess | Resample / VAD split / peak norm → JSONL with **required** `text`, `license`, `allowed_use`, `source`. |
| Commercial seed | Only `allowed_use=research_and_commercial_with_attribution` (CC-BY). Casablanca NC-ND is **eval/ref only**. |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[asr,f5]"

# Unit tests
pytest -q

# Build commercial train seed (after assembling source manifests)
python -m src.training.f5_finetune \
  --manifests data/seed/processed/cafe_manifest_v2.jsonl \
              data/seed/dvoice_ary/manifest.jsonl \
  --prepare-only --smoke

# GPU full fine-tune (see also data/f5_datasets/TRAIN_COMMAND.sh)
python -m src.training.f5_finetune --manifests ... --train

# Listen A/B
python -m src.eval.listen_ab \
  --listen-json data/seed/listen_set/listen_set.json \
  --product-lines data/seed/listen_set/product_lines.json \
  --output-dir data/seed/listen_set/ab_out
# After fine-tune, pass --f5-ckpt <path> --f5-tag f5ft
```

## Seed corpus

Assembled under `data/seed/`:

- **CAFE** (Algerian conversational, CC-BY-4.0) — commercial-friendly; M2 uses faster-whisper-small drafts + light QC (still not human-corrected).
- **Casablanca Algeria** (CC-BY-NC-ND-4.0) — research/eval / F5 reference only; **never** in commercial train.
- **DVoice Darija** (Moroccan, CC-BY-4.0) — Maghrebi gap documented.
- Optional **Moroccan Wiki Audio** (CC-BY-4.0) — same Maghrebi gap.

License table: `data/seed/SOURCES_LICENSE_MANIFEST.json`

**Do not** train shipping voices on Casablanca (NC-ND) or on scraped YouTube without rights.

## Project layout

```
configs/       F5 fine-tune YAML
scripts/       seed download / ASR / assemble helpers
src/dataset/   preprocess, commercial filter, ASR QC
src/models/    MMS + F5 engine
src/training/  f5_finetune (real) + dataset loader
src/eval/      listen-set A/B harness
data/seed/     open/licensed seed + listen set
tests/         unit tests (no quality claims)
```

## Hard bans

- Simulated training / empty `state_dict` checkpoints presented as success
- Empty transcripts in training manifests
- Scrape-to-ship without rights
- README fiction (F5/XTTS/emotion “done” when not)

## GPU needs (honest)

Full F5TTS_v1_Base fine-tune: **≥16 GB VRAM** (tight), **24 GB recommended**. CPU hosts: prepare data + `--smoke` only; run `data/f5_datasets/TRAIN_COMMAND.sh` on a GPU machine.
