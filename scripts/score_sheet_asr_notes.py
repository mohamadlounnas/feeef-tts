"""
Recovered from transcript shell heredoc (msg 187).
Ephemeral M1 seed/ASR assembly — was not a committed scripts/ file.
"""

import json, soundfile as sf, whisper, numpy as np
from pathlib import Path
media = Path('/cursor/stores/self/media/m1-listen-set')
sheet = json.loads((media/'score_sheet.json').read_text())
model = whisper.load_model('tiny')
print('id | text | mms_s | f5_s | mms_asr | f5_asr')
for item in sheet['items'][:6]:
    text = item['text']
    mms = media/f"{item['id']}_mms.wav"
    f5 = media/f"{item['id']}_f5zs.wav"
    md = sf.info(str(mms)).duration if mms.exists() else None
    fd = sf.info(str(f5)).duration if f5.exists() else None
    mms_asr = whisper.load_model and model.transcribe(str(mms), language='ar', fp16=False)['text'].strip() if mms.exists() else ''
    f5_asr = model.transcribe(str(f5), language='ar', fp16=False)['text'].strip() if f5.exists() else ''
    print(f"{item['id']}|{text}|{md:.2f}|{fd:.2f}|{mms_asr}|{f5_asr}")
    item['asr_mms_tiny'] = mms_asr
    item['asr_f5_tiny'] = f5_asr
# provisional agent listen notes (not native MOS)
notes_path = media/'agent_listen_notes.json'
json.dump({'method':'whisper-tiny ASR + waveform presence; NOT native MOS','items':sheet['items']}, open(notes_path,'w'), ensure_ascii=False, indent=2)
print('wrote', notes_path)
