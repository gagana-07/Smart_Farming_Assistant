import os
import whisper
import ollama
import chromadb
import soundfile as sf
import noisereduce as nr

from translate import translate
from audio.tts import speak

AUDIO_FILE = "audio/input.wav"
CLEAN_FILE = "audio/clean.wav"
SUPPORTED = ["en", "kn", "hi"]

# ==========================
# LOAD MODELS ONCE
# ==========================
print("Loading Whisper...")
whisper_model = whisper.load_model("medium")

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
"""


# ==========================
# STEP 1: CLEAN AUDIO
# ==========================
def clean_audio(src, dst):
    data, sr = sf.read(src)
    if data.ndim > 1:
        data = data.mean(axis=1)
    reduced = nr.reduce_noise(y=data, sr=sr)
    sf.write(dst, reduced, sr)


# ==========================
# STEP 2: SPEECH TO TEXT
# ==========================
def detect_language(path):
    audio = whisper.pad_or_trim(whisper.load_audio(path))
    mel = whisper.log_mel_spectrogram(
        audio, n_mels=whisper_model.dims.n_mels
    ).to(whisper_model.device)
    _, probs = whisper_model.detect_language(mel)
    p = {l: probs.get(l, 0) for l in SUPPORTED}
    print("Language probabilities:", {k: round(v, 3) for k, v in p.items()})

    best = max(p, key=p.get)
    # Whisper often confuses Kannada with English on accented speech
    if best == "en" and abs(p["en"] - p["kn"]) < 0.10:
        best = "kn"
    return best


def transcribe(path):
    lang = detect_language(path)
    text = whisper_model.transcribe(path, language=lang, fp16=False)["text"].strip()

    # Correct the language from the script actually produced
    if any("\u0C80" <= c <= "\u0CFF" for c in text):
        lang = "kn"
    elif any("\u0900" <= c <= "\u097F" for c in text):
        lang = "hi"
    return text, lang


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
    )
    answer = response["message"]["content"].strip()
    for ch in ["*", "#", "•"]:
        answer = answer.replace(ch, "")
    return answer


# ==========================
# MAIN
# ==========================
def main():
    if not os.path.exists(AUDIO_FILE):
        print("Audio file not found:", AUDIO_FILE)
        return

    os.makedirs("outputs", exist_ok=True)

    print("\nReducing noise...")
    clean_audio(AUDIO_FILE, CLEAN_FILE)

    print("\nTranscribing...")
    question, lang = transcribe(CLEAN_FILE)
    if len(question) < 2:
        print("No speech detected")
        return
    print(f"\nFarmer question ({lang}): {question}")

    # Translate question to English for retrieval + LLM
    question_en = translate(question, lang, "en")
    print("Question in English:", question_en)

    context = retrieve(question_en)
    if not context:
        print("No relevant farming information found.")
        return

    print("\nGenerating answer...")
    answer_en = generate_answer(question_en, context)
    print("Answer (English):", answer_en)

    # Translate answer back to the farmer's language
    answer = translate(answer_en, "en", lang)
    print(f"\nFarmer Assistant ({lang}):\n{answer}")

    with open("answer.txt", "w", encoding="utf-8") as f:
        f.write(answer)

    # Speak
    if lang in ("kn", "hi"):
        path = speak(answer, lang, "outputs/response.wav")
        print("\nAudio saved:", path)
    else:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.say(answer)
        engine.runAndWait()


if __name__ == "__main__":
    main()