# Experimental log (honest)

## M1 (2026-10-11)
- Seed v0 ~1.23 h (CAFE Whisper-tiny drafts + Casablanca NC-ND + DVoice).
- F5 zero-shot did **not** beat MMS on product Darja listen lines.
- Simulated trainer removed; push to GitHub blocked (403).

## M2 (2026-10-11)
- Recreated M1 code on new VM from framework branch + transcript recovery (`cursor/m2-commercial-finetune-31ad`).
- Commercial train filter: `allowed_use=research_and_commercial_with_attribution` only (Casablanca excluded from train).
- CAFE re-ASR with faster-whisper-small + light QC (`scripts/assemble_cafe_asr_v2.py`).
- Grew DVoice Maghrebi CC-BY hours; Moroccan Wiki Audio gated — skipped.
- Real F5 entrypoint: `src/training/f5_finetune.py` (csv/wavs + char arrow + `TRAIN_COMMAND.sh` + `--smoke` CFM autograd).
- Full `--train` refused on CPU-only host; GPU estimate documented.
- Listen A/B re-run under `media/m2-listen-set/` (see Project store docs).
