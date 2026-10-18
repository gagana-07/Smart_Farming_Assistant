"""
Try a Kannada-trained Whisper model on your own recording.

Run from the project root. It uses the recordings already in audio/ :

    python test_kannada_stt.py                      # small model, audio/input.wav and audio/clean.wav
    python test_kannada_stt.py medium               # bigger model (slower, usually more accurate)
    python test_kannada_stt.py small path/to/file.wav

The first run downloads the model (about 1 GB for small, 3 GB for medium), so it needs internet.
Needs:  pip install transformers torch faster-whisper
"""
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")   # hides a harmless Windows warning

import sys
import time

from faster_whisper import decode_audio      # reads wav/m4a/mp3 and returns 16 kHz mono
from transformers import pipeline, GenerationConfig

SIZE = sys.argv[1] if len(sys.argv) > 1 else "small"
MODEL = f"vasista22/whisper-kannada-{SIZE}"
BASE = f"openai/whisper-{SIZE}"
FILES = sys.argv[2:] or ["audio/input.wav", "audio/clean.wav"]

print("Loading", MODEL, "(first run downloads it)...")
t = time.time()
asr = pipeline(
    task="automatic-speech-recognition",
    model=MODEL,
    device=-1,                                # CPU
)

# This model was fine-tuned in 2022, so its generation settings are out of date.
# Borrow the standard settings from the matching OpenAI model so the language
# option works with current versions of transformers.
asr.model.generation_config = GenerationConfig.from_pretrained(BASE)
print(f"Loaded in {time.time() - t:.0f}s")


def transcribe(audio):
    last_error = None
    for language in ("kannada", "kn"):
        try:
            # Build a NEW dictionary on every call: the pipeline consumes it.
            result = asr(
                {"raw": audio.copy(), "sampling_rate": 16000},
                generate_kwargs={"language": language, "task": "transcribe"},
            )
            return result["text"]
        except Exception as e:
            last_error = e
    raise last_error


for path in FILES:
    if not os.path.exists(path):
        print("\nMissing:", path)
        continue
    audio = decode_audio(path, sampling_rate=16000)
    t = time.time()
    try:
        text = transcribe(audio)
    except Exception as e:
        print(f"\n{path}\n   ERROR: {type(e).__name__}: {e}")
        continue
    print(f"\n{path}  ({time.time() - t:.0f}s)")
    print("   heard:", text.strip())