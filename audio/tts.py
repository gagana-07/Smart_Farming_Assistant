import re
import numpy as np
import torch
import scipy.io.wavfile as wav
from transformers import VitsModel, AutoTokenizer

# Load once at startup (not on every request)
MODELS = {
    "kn": "models/mms-tts-kan",   # or "facebook/mms-tts-kan"
    "hi": "models/mms-tts-hin",   # or "facebook/mms-tts-hin"
}
_cache = {}

def _load(lang):
    if lang not in _cache:
        tok = AutoTokenizer.from_pretrained(MODELS[lang])
        mdl = VitsModel.from_pretrained(MODELS[lang]).eval()
        _cache[lang] = (tok, mdl)
    return _cache[lang]

def clean_text(text, lang):
    text = re.sub(r"[*#•_`\-–—]+", " ", text)          # markdown / bullets
    text = re.sub(r"^\s*\d+[.)]\s*", "", text, flags=re.M)  # "1. " list markers
    text = text.replace("\n", ". ")
    if lang == "kn":
        text = re.sub(r"[A-Za-z]+", " ", text)          # drop Latin words
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_sentences(text):
    parts = re.split(r"(?<=[.!?।])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 1]

def speak(text, lang, out_path="outputs/response.wav"):
    tok, mdl = _load(lang)
    sr = mdl.config.sampling_rate
    chunks = []
    for sent in split_sentences(clean_text(text, lang)):
        inputs = tok(sent, return_tensors="pt")
        if inputs["input_ids"].shape[1] == 0:
            continue
        with torch.no_grad():
            wave = mdl(**inputs).waveform.squeeze().cpu().numpy()
        chunks.append(wave)
        chunks.append(np.zeros(int(0.25 * sr), dtype=np.float32))  # pause
    if not chunks:
        raise ValueError("Nothing speakable after cleaning")
    audio = np.concatenate(chunks)
    audio = audio / max(1e-6, np.abs(audio).max()) * 0.95      # normalize
    wav.write(out_path, sr, (audio * 32767).astype(np.int16))
    return out_path