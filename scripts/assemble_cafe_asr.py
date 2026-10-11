"""
Recovered from transcript shell heredoc (msg 94).
Ephemeral M1 seed/ASR assembly — was not a committed scripts/ file.
"""

import os, json, csv, re
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import whisper

cafe_dir = Path('/workspace/repos/feeef-tts/data/seed/cafe_raw')
out_wav = Path('/workspace/repos/feeef-tts/data/seed/processed/cafe_segments')
out_wav.mkdir(parents=True, exist_ok=True)

# Map basename -> full transcript
text_map = {}
csv_path = list(cafe_dir.glob('*.csv'))[0]
with open(csv_path, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        base = os.path.basename(row['path'])
        text_map[base] = (row.get('text') or '').strip()

print('Loading whisper tiny...')
model = whisper.load_model('tiny')

manifest = []
total = 0.0
target_cap = 30 * 60  # 30 min of segments

for wav_path in sorted(cafe_dir.glob('*.wav')):
    if total >= target_cap:
        break
    parent_text = text_map.get(wav_path.name, '')
    audio, sr = librosa.load(str(wav_path), sr=24000, mono=True)
    # peak norm
    peak = np.max(np.abs(audio)) or 1.0
    audio = audio * (0.9 / peak)
    intervals = librosa.effects.split(audio, top_db=30)
    for idx, (start, end) in enumerate(intervals):
        dur = (end - start) / sr
        if dur < 2.0 or dur > 10.0:
            # chunk long intervals
            if dur > 10.0:
                max_samples = int(8.0 * sr)
                for sub in range(start, end, max_samples):
                    sub_end = min(sub + max_samples, end)
                    seg_dur = (sub_end - sub) / sr
                    if seg_dur < 2.0:
                        continue
                    seg = audio[sub:sub_end]
                    # ASR
                    # whisper expects 16k
                    seg16 = librosa.resample(seg, orig_sr=sr, target_sr=16000)
                    result = model.transcribe(seg16, language='ar', fp16=False)
                    text = (result.get('text') or '').strip()
                    if not text or len(text) < 3:
                        continue
                    fname = f"{wav_path.stem}_seg_{idx:03d}_{sub}.wav"
                    path = out_wav / fname
                    sf.write(str(path), seg, sr)
                    manifest.append({
                        'audio_filepath': str(path),
                        'text': text,
                        'duration': round(seg_dur, 3),
                        'speaker_id': 'cafe_dz_multi',
                        'dialect': 'Algerian_Darja',
                        'dialect_note': 'CAFE spontaneous Algerian Arabic/French/English code-switch podcast dialogue; multi-speaker; Algiers-media everyday register (not single Algiers talent)',
                        'license': 'CC-BY-4.0',
                        'allowed_use': 'research_and_commercial_with_attribution',
                        'source': 'FatimahEmadEldin/cafe-algerian-codeswitch-speech',
                        'source_url': 'https://huggingface.co/datasets/FatimahEmadEldin/cafe-algerian-codeswitch-speech',
                        'upstream': 'https://doi.org/10.5281/zenodo.16964503',
                        'transcript_status': 'whisper_tiny_draft',
                        'parent_file': wav_path.name,
                        'parent_transcript_available': bool(parent_text),
                        'sample_rate': 24000,
                    })
                    total += seg_dur
                    if total >= target_cap:
                        break
            continue
        seg = audio[start:end]
        seg16 = librosa.resample(seg, orig_sr=sr, target_sr=16000)
        result = model.transcribe(seg16, language='ar', fp16=False)
        text = (result.get('text') or '').strip()
        if not text or len(text) < 3:
            continue
        fname = f"{wav_path.stem}_seg_{idx:03d}.wav"
        path = out_wav / fname
        sf.write(str(path), seg, sr)
        manifest.append({
            'audio_filepath': str(path),
            'text': text,
            'duration': round(dur, 3),
            'speaker_id': 'cafe_dz_multi',
            'dialect': 'Algerian_Darja',
            'dialect_note': 'CAFE spontaneous Algerian Arabic/French/English code-switch podcast dialogue; multi-speaker; Algiers-media everyday register (not single Algiers talent)',
            'license': 'CC-BY-4.0',
            'allowed_use': 'research_and_commercial_with_attribution',
            'source': 'FatimahEmadEldin/cafe-algerian-codeswitch-speech',
            'source_url': 'https://huggingface.co/datasets/FatimahEmadEldin/cafe-algerian-codeswitch-speech',
            'upstream': 'https://doi.org/10.5281/zenodo.16964503',
            'transcript_status': 'whisper_tiny_draft',
            'parent_file': wav_path.name,
            'parent_transcript_available': bool(parent_text),
            'sample_rate': 24000,
        })
        total += dur
        if total >= target_cap:
            break
    print(f'{wav_path.name}: total so far {total/60:.1f} min, clips {len(manifest)}')

man_path = Path('/workspace/repos/feeef-tts/data/seed/processed/cafe_manifest.jsonl')
with open(man_path, 'w', encoding='utf-8') as f:
    for e in manifest:
        f.write(json.dumps(e, ensure_ascii=False)+'\n')
print(f'CAFE processed: {len(manifest)} clips, {total/60:.1f} min -> {man_path}')
