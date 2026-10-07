import whisper
import ollama
import chromadb
import os
import torch
import scipy.io.wavfile as wav
import soundfile as sf
import noisereduce as nr

from transformers import VitsModel, AutoTokenizer

print("Loading Whisper...")
model = whisper.load_model("medium")

# ==========================
# CHROMADB
# ==========================

client = chromadb.PersistentClient(path="data/vector_db")
collection = client.get_collection("agriculture")

# ==========================
# AUDIO FILE
# ==========================

audio_file = "audio/input.wav"

if not os.path.exists(audio_file):
    print("Audio file not found:", audio_file)
    exit()
    
    # ==========================
# NOISE REDUCTION
# ==========================

print("\nReducing background noise...")

audio_data, sr = sf.read(audio_file)

reduced_noise = nr.reduce_noise(
    y=audio_data,
    sr=sr
)

clean_audio_file = "audio/clean.wav"

sf.write(
    clean_audio_file,
    reduced_noise,
    sr
)

print("Noise reduction completed.")

# ==========================
# LANGUAGE DETECTION
# ==========================

print("\nDetecting language...")

audio = whisper.load_audio(clean_audio_file)
audio = whisper.pad_or_trim(audio)

mel = whisper.log_mel_spectrogram(
    audio,
    n_mels=model.dims.n_mels
).to(model.device)

_, probs = model.detect_language(mel)

supported_languages = ["en", "kn", "hi"]

supported_languages = ["en", "kn", "hi"]

restricted_probs = {
    lang: probs.get(lang, 0)
    for lang in supported_languages
}

print("\nLanguage Probabilities:")
print("English :", round(restricted_probs["en"], 3))
print("Kannada :", round(restricted_probs["kn"], 3))
print("Hindi   :", round(restricted_probs["hi"], 3))

en_prob = restricted_probs["en"]
kn_prob = restricted_probs["kn"]
hi_prob = restricted_probs["hi"]

# Prefer Kannada if English and Kannada are very close
if abs(en_prob - kn_prob) < 0.10:
    lang = "kn"
else:
    lang = max(restricted_probs, key=restricted_probs.get)

print("\nSelected Language:", lang)
# ==========================
# SPEECH TO TEXT
# ==========================

print("\nTranscribing audio...")

result = model.transcribe(
    clean_audio_file,
    language=lang,
    fp16=False
)

question = result["text"].strip()
if any('\u0C80' <= ch <= '\u0CFF' for ch in question):
    lang = "kn"

elif any('\u0900' <= ch <= '\u097F' for ch in question):
    lang = "hi"

print("\nFarmer Question:")
print(question)

if len(question) < 2:
    print("No speech detected")
    exit()

# ==========================
# CHROMADB SEARCH
# ==========================

results = collection.query(
    query_texts=[question],
    n_results=5
)

if (
    "documents" not in results
    or not results["documents"]
    or not results["documents"][0]
):
    print("No relevant farming information found.")
    exit()

context = "\n".join(results["documents"][0])
context = context[:1500]

print("\n===================")
print("CONTEXT RETRIEVED")
print("===================")
print(context)
print("===================\n")

# ==========================
# PROMPT
# ==========================

if lang == "kn":

    system_prompt = """
You are a Smart Agriculture Assistant.

Reply ONLY in Kannada Unicode script.

Correct:
ಭತ್ತಕ್ಕೆ ನೀರು ನೀಡಿ.

Wrong:
Bhattakke neeru nidi.

Never use English letters.
Never use Roman Kannada.

Use simple Kannada.
Maximum 4 short lines.
Give practical farming advice.
Answer only the question.
"""

elif lang == "hi":

    system_prompt = """
You are a Smart Agriculture Assistant.

Reply ONLY in Hindi.

Use simple Hindi.
Maximum 4 short lines.
Give practical farming advice.
Answer only the question.
"""

else:

    system_prompt = """
You are a Smart Agriculture Assistant.

Reply ONLY in English.

Use simple English.
Maximum 4 short lines.
Give practical farming advice.
Answer only the question.
"""

# ==========================
# OLLAMA
# ==========================

print("\nGenerating answer...")

try:

    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"""
Question:
{question}

Context:
{context}
"""
            }
        ]
    )

    answer = response["message"]["content"].strip()

except Exception as e:
    print("Ollama Error:", e)
    exit()

# ==========================
# CLEAN OUTPUT
# ==========================

for ch in ["*", "#", "•"]:
    answer = answer.replace(ch, "")

print("\nFarmer Assistant:\n")
print(answer)

# Save answer

with open("answer.txt", "w", encoding="utf-8") as f:
    f.write(answer)

print("\nSaved to answer.txt")

# ==========================
# KANNADA AUDIO
# ==========================

if lang == "kn":

    try:

        print("\nGenerating Kannada audio...")

        print("\nAnswer sent to TTS:")
        print(answer)

        tokenizer = AutoTokenizer.from_pretrained(
            "models/mms-tts-kan"
        )

        tts_model = VitsModel.from_pretrained(
            "models/mms-tts-kan"
        )

        inputs = tokenizer(
            answer,
            return_tensors="pt",
            truncation=True,
            max_length=200
        )

        with torch.no_grad():
            output = tts_model(**inputs).waveform

        audio = output.squeeze().cpu().numpy()

        wav.write(
            "outputs/response.wav",
            rate=tts_model.config.sampling_rate,
            data=audio
        )

        print("\nKannada audio saved:")
        print("outputs/response.wav")

    except Exception as e:

        print("Kannada TTS Error:", e)

# ==========================
# ENGLISH / HINDI AUDIO
# ==========================

else:

    try:

        import pyttsx3

        engine = pyttsx3.init()

        engine.setProperty("rate", 150)

        print("\nSpeaking answer...")

        engine.say(answer)
        engine.runAndWait()

    except Exception as e:

        print("TTS Error:", e)