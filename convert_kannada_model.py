"""
Convert the Kannada Whisper model to the fast format used by faster-whisper.
Run ONCE from the project root (needs internet):

    python convert_kannada_model.py            # small model  -> models/whisper-kn-small-ct2
    python convert_kannada_model.py medium     # medium model -> models/whisper-kn-medium-ct2

Afterwards app.py finds the folder and uses it for Kannada automatically.
"""
import os
import shutil
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import ctranslate2
from huggingface_hub import hf_hub_download

SIZE = sys.argv[1] if len(sys.argv) > 1 else "small"
MODEL = f"vasista22/whisper-kannada-{SIZE}"
BASE = f"openai/whisper-{SIZE}"
OUT = f"models/whisper-kn-{SIZE}-ct2"

print("Converting", MODEL, "->", OUT, "(this takes a few minutes)...")
converter = ctranslate2.converters.TransformersConverter(MODEL)
converter.convert(OUT, quantization="int8", force=True)

# faster-whisper needs these two files next to the converted model.
# The Kannada models use the same vocabulary as the OpenAI model they came from.
for name in ("tokenizer.json", "preprocessor_config.json"):
    shutil.copy(hf_hub_download(BASE, name), os.path.join(OUT, name))

print("Done. Files in", OUT, ":", sorted(os.listdir(OUT)))