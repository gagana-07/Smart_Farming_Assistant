import os
import sys
import time
import subprocess

import ollama
import chromadb
import soundfile as sf
import noisereduce as nr
from faster_whisper import WhisperModel
from faster_whisper import WhisperModel, decode_audio

from translate import translate
from audio.tts import speak

AUDIO_FILE = "audio/input.wav"
CLEAN_FILE = "audio/clean.wav"

# ==========================
# LOAD MODELS ONCE
# ==========================
print("Loading Whisper...")
# "small" is fast on CPU. Try "medium" if Kannada accuracy is too low.
fw = WhisperModel("small", device="cpu", compute_type="int8")

client = chromadb.PersistentClient(path="data/vector_db")
collection = client.get_collection("agriculture")

SYSTEM_PROMPT = """
You are a Smart Agriculture Assistant for Indian farmers.
Reply ONLY in English.
Use simple words and short sentences.
Maximum 4 short sentences.
Give practical farming advice using the context.
Write numbers and units in words (for example: "fifty kilograms").
Do not use abbreviations, bullet points, symbols or markdown.
Answer only the question.
Do not use words like ring or cultural practices. Say "around the plant".
Answer only what was asked.
"""


def timed(label, fn, *args):
    t = time.time()
    out = fn(*args)
    print(f"   [{label}: {time.time() - t:.1f}s]")
    return out


# ==========================
# STEP 1: CLEAN AUDIO
# ==========================
def clean_audio(src, dst):
    data, sr = sf.read(src)
    if data.ndim > 1:
        data = data.mean(axis=1)
    sf.write(dst, nr.reduce_noise(y=data, sr=sr), sr)


# ==========================
# STEP 2: SPEECH TO TEXT
# ==========================


ALLOWED = ["en", "kn", "hi"]
RANGES = {"kn": ("\u0C80", "\u0CFF"), "hi": ("\u0900", "\u097F")}


def looks_valid(text, lang):
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 4:
        return False
    if lang == "en":
        return sum(c.isascii() for c in letters) / len(letters) > 0.9
    lo, hi = RANGES[lang]
    return sum(lo <= c <= hi for c in letters) / len(letters) > 0.8


def transcribe(path):
    audio = decode_audio(path, sampling_rate=16000)

    # Detect language, but only among English / Kannada / Hindi
    _, _, all_probs = fw.detect_language(audio)
    probs = {l: p for l, p in all_probs if l in ALLOWED}
    for l in ALLOWED:
        probs.setdefault(l, 0.0)
    ranked = sorted(probs, key=probs.get, reverse=True)
    print("Language probabilities:", {l: round(probs[l], 3) for l in ALLOWED})

    # Try the most likely language first. If the output is not valid text
    # in that language's script, try the next one.
    for lang in ranked:
        segments, _ = fw.transcribe(
            audio,
            language=lang,
            beam_size=1,
            condition_on_previous_text=False,
        )
        text = " ".join(s.text for s in segments).strip()
        if looks_valid(text, lang):
            return text, lang

    return "", ranked[0]


# ==========================
# STEP 3: RETRIEVAL
# ==========================
def retrieve(question_en):
    res = collection.query(
        query_texts=[question_en], n_results=5,
        include=["documents", "distances"],
    )
    docs, dists = res["documents"][0], res["distances"][0]
    print("Distances:", [round(d, 2) for d in dists])   
    kept = [d for d, dist in zip(docs, dists) if dist < 1.0]  
    return "\n".join(kept)[:1500]

# ==========================
# STEP 4: ANSWER (ENGLISH)
# ==========================
def generate_answer(question_en, context):
    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question:\n{question_en}\n\nContext:\n{context}",
            },
        ],
        options={"num_predict": 120},
        keep_alive="30m",
    )
    answer = response["message"]["content"].strip()
    for ch in ["*", "#", "•"]:
        answer = answer.replace(ch, "")
    return answer


# ==========================
# ONE QUESTION
# ==========================
def run_once():
    if not os.path.exists(AUDIO_FILE):
        print("Audio file not found:", AUDIO_FILE)
        return

    os.makedirs("outputs", exist_ok=True)

    print("\nReducing noise...")
    timed("noise", clean_audio, AUDIO_FILE, CLEAN_FILE)

    print("Transcribing...")
    question, lang = timed("whisper", transcribe, CLEAN_FILE)
    if len(question) < 2:
        print("No speech detected")
        return
    print(f"\nFarmer question ({lang}): {question}")

    print("\nTranslating question...")
    question_en = timed("translate", translate, question, lang, "en")
    print("Question in English:", question_en)

    context = timed("retrieve", retrieve, question_en)
    if not context:
        print("No relevant farming information found.")
        return

    print("\nGenerating answer...")
    answer_en = timed("llm", generate_answer, question_en, context)
    print("Answer (English):", answer_en)

    print("\nTranslating answer...")
    answer = timed("translate", translate, answer_en, "en", lang)
    print(f"\nFarmer Assistant ({lang}):\n{answer}")

    with open("answer.txt", "w", encoding="utf-8") as f:
        f.write(answer)

    if lang in ("kn", "hi"):
        print("\nGenerating speech...")
        path = timed("tts", speak, answer, lang, "outputs/response.wav")
        print("Audio saved:", path)
    else:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.say(answer)
        engine.runAndWait()


# ==========================
# LOOP: models stay loaded between questions
# ==========================
if __name__ == "__main__":
    while True:
        try:
            input("\nPress Enter, then speak for 10 seconds (Ctrl+C to quit)...")
        except KeyboardInterrupt:
            print("\nBye")
            break
        subprocess.run([sys.executable, "audio/record.py"])
        total = time.time()
        run_once()
        print(f"\nTotal: {time.time() - total:.1f}s")