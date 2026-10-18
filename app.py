print("Starting... loading libraries (this can take a minute or two)")

import os
import re
import sys
import time
import subprocess

import ollama
import chromadb
import soundfile as sf
import noisereduce as nr
from faster_whisper import WhisperModel, decode_audio

from translate import translate
from audio.tts import speak

try:
    import winsound          # Windows only
except ImportError:
    winsound = None

AUDIO_FILE = "audio/input.wav"
CLEAN_FILE = "audio/clean.wav"

# Noise reduction removed the crop name from your Kannada test (ರಾಗಿ was lost),
# so it is switched off. Set to True only if a test shows it helps.
USE_NOISE_REDUCTION = False
ALLOWED = ["en", "kn", "hi"]
RANGES = {"kn": ("\u0C80", "\u0CFF"), "hi": ("\u0900", "\u097F")}

# Lower distance = more similar. In your tests, good farming matches were
# 0.43 to 0.65 and off-topic questions were 1.0 or more.
DIST_CUTOFF = 0.9

# ==========================
# LOAD MODELS ONCE
# ==========================
print("Loading Whisper...")
fw = WhisperModel("small", device="cpu", compute_type="int8")

# Kannada-trained model (create it once with: python convert_kannada_model.py)
KN_MODEL_DIR = "models/whisper-kn-small-ct2"
if os.path.isdir(KN_MODEL_DIR):
    print("Loading Kannada speech model...")
    fw_kn = WhisperModel(KN_MODEL_DIR, device="cpu", compute_type="int8")
else:
    print("Kannada model not converted yet. Using the general model for Kannada.")
    fw_kn = fw
SPEECH = {"en": fw, "hi": fw, "kn": fw_kn}

client = chromadb.PersistentClient(path="data/vector_db")
collection = client.get_collection("agriculture")

SYSTEM_PROMPT = """
You are a Smart Agriculture Assistant for Indian farmers.
Reply ONLY in English.
Use simple words and short sentences.
Maximum 4 short sentences.
Use ONLY the context to answer. Never summarize the context.
The answer must match what the question asks. For example, do not give
spray water volume as an irrigation answer.
If the context does not answer the question, say exactly:
I do not have information on this.
Write numbers and units in words (for example: "fifty kilograms").
Do not use abbreviations, bullet points, symbols or markdown.
"""

NOT_UNDERSTOOD = "I could not understand the audio. Please speak clearly and ask again."
NO_INFO = "I do not have information on this. Please ask another question."
CAUTION = "Please confirm with your local agriculture officer before using any chemical."

CHEMICAL_WORDS = re.compile(
    r"\b(spray|pesticide|insecticide|fungicide|herbicide|dose|dosage|"
    r"gram|grams|millilit\w+|kilogram\w*|litre\w*|liter\w*)\b",
    re.I,
)


def timed(label, fn, *args):
    t = time.time()
    out = fn(*args)
    print(f"   [{label}: {time.time() - t:.1f}s]")
    return out


def play(path):
    if winsound is None:
        print("(Playback is only built in for Windows. Open", path, "to listen.)")
        return
    try:
        winsound.PlaySound(path, winsound.SND_FILENAME)
    except Exception as e:
        print("Playback error:", e)


def speak_answer(text, lang):
    if lang in ("kn", "hi"):
        path = timed("tts", speak, text, lang, "outputs/response.wav")
        print("Audio saved:", path)
        play(path)
    else:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("TTS Error:", e)


def say(message_en, lang):
    """Print and speak a fixed message in the farmer's language."""
    msg = translate(message_en, "en", lang) if lang != "en" else message_en
    print("\nFarmer Assistant:", msg)
    speak_answer(msg, lang)


# ==========================
# NUMBERS -> WORDS (so doses are not lost when the answer is spoken)
# ==========================
try:
    from num2words import num2words
except ImportError:
    num2words = None
    print("Tip: run  pip install num2words  so quantities are spoken as words.")


def numbers_to_words(text, to_words=None):
    """'20 grams' -> 'twenty grams'; '50-100 kg' -> 'fifty to one hundred kg'."""
    to_words = to_words or num2words
    if to_words is None:
        return text

    def word(n):
        try:
            return to_words(float(n) if "." in n else int(n))
        except Exception:
            return n

    num = r"\d+(?:\.\d+)?"
    # fertilizer grades such as 17-17-17 or 10:26:26
    text = re.sub(
        r"\b(\d{1,2})[-:](\d{1,2})[-:](\d{1,2})\b",
        lambda m: " ".join(word(g) for g in m.groups()),
        text,
    )
    # ranges such as 50-100
    text = re.sub(
        rf"({num})\s*[-\u2013]\s*({num})",
        lambda m: f"{word(m.group(1))} to {word(m.group(2))}",
        text,
    )
    text = re.sub(rf"({num})\s*%", lambda m: f"{word(m.group(1))} percent", text)
    return re.sub(num, lambda m: word(m.group(0)), text)


# ==========================
# STEP 1: CLEAN AUDIO
# ==========================
def clean_audio(src, dst):
    data, sr = sf.read(src)
    if data.ndim > 1:
        data = data.mean(axis=1)
    sf.write(dst, nr.reduce_noise(y=data, sr=sr), sr)


# ==========================
# STEP 2: SPEECH TO TEXT (English / Kannada / Hindi only)
# ==========================
# English words that real English farming questions contain. Romanized Kannada
# such as "Batta vanu hege belia beko" contains none of them, so it is rejected.
EN_WORDS = set(
    """the a an is are was what how when where which why who to of in on for with do does
    did can could should would i my me we you it this that and or not at by from be have
    has had will about much many best time need needed
    ragi paddy rice maize corn wheat tomato onion potato cotton sugarcane groundnut jowar
    sorghum millet chilli coconut areca banana mango grape sow sowing plant planting grow
    fertilizer fertiliser urea manure compost pest pests disease water irrigation soil
    seed seeds crop crops spray weed weeds harvest yield leaf leaves yellow""".split()
)

# Words that nudge Whisper toward farming vocabulary.
PROMPTS = {
    "en": "Farming questions: ragi, paddy, maize, tomato, sowing, fertilizer, urea, pest, irrigation.",
    "kn": "ರಾಗಿ, ಭತ್ತ, ಮೆಕ್ಕೆಜೋಳ, ಟೊಮೆಟೊ, ಗೊಬ್ಬರ, ಕೀಟ, ನೀರು, ಬಿತ್ತನೆ",
    "hi": "धान, रागी, मक्का, टमाटर, खाद, कीट, सिंचाई, बुवाई",
}


def looks_valid(text, lang):
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 4:
        return False
    if lang == "en":
        if sum(c.isascii() for c in letters) / len(letters) < 0.9:
            return False
        words = [w.strip(".,?!;:").lower() for w in text.split()]
        hits = sum(w in EN_WORDS for w in words)
        return hits >= 1 and hits / max(1, len(words)) >= 0.25
    lo, hi = RANGES[lang]
    return sum(lo <= c <= hi for c in letters) / len(letters) > 0.8


def _run(audio, lang):
    """Transcribe in one language. Returns (text, confidence)."""
    model = SPEECH[lang]
    kwargs = dict(
        language=lang,
        beam_size=1,
        temperature=0.0,                 # no slow retry passes on unclear audio
        condition_on_previous_text=False,
        initial_prompt=PROMPTS[lang],
        max_new_tokens=200,              # Kannada script needs many tokens per word
    )
    if model is not fw:
        kwargs["initial_prompt"] = None  # the Kannada model does not need vocabulary hints
    try:
        segments, _ = model.transcribe(audio, **kwargs)
    except TypeError:                    # older faster-whisper without max_new_tokens
        kwargs.pop("max_new_tokens")
        segments, _ = model.transcribe(audio, **kwargs)
    segs = list(segments)
    text = " ".join(s.text for s in segs).strip()
    conf = sum(s.avg_logprob for s in segs) / len(segs) if segs else -9.0
    return text, conf


def transcribe(path):
    """Detect the language automatically (English / Kannada / Hindi only)."""
    audio = decode_audio(path, sampling_rate=16000)

    _, _, all_probs = fw.detect_language(audio)
    probs = {l: p for l, p in all_probs if l in ALLOWED}
    for l in ALLOWED:
        probs.setdefault(l, 0.0)
    ranked = sorted(probs, key=probs.get, reverse=True)
    print("Language probabilities:", {l: round(probs[l], 3) for l in ALLOWED})

    # Whisper's language guess is often wrong on short or accented speech, so
    # we do not trust it alone. Try the likely languages, keep only transcripts
    # that are real text in that language, and pick the most confident one.
    candidates = []
    for lang in ranked:
        text, conf = _run(audio, lang)
        print(f"   [{lang} attempt] {text}   (confidence {conf:.2f})")
        if looks_valid(text, lang):
            # A transcript in the language Whisper thought most likely gets a bonus.
            # (Writing Kannada sounds in Devanagari can look "confident" without
            # being the right language, so confidence alone is not enough.)
            prior = probs[lang] / max(sum(probs.values()), 1e-6)
            candidates.append((conf + 1.5 * prior, lang, text))
            # Sure about the language and the transcript: no need to try the others
            if probs[lang] >= 0.6 or conf > -0.5:
                break

    if not candidates:
        return "", ranked[0]
    _, lang, text = max(candidates)
    return text, lang


# ==========================
# STEP 3: RETRIEVAL
# ==========================
def clean_context(text):
    # KCC records contain Kannada text next to the English. The English-only
    # model does not need it, and it wastes space.
    text = re.sub(r"[\u0C80-\u0CFF\u0900-\u097F]+", " ", text)
    return re.sub(r"[ \t]+", " ", text)


def retrieve(question_en):
    res = collection.query(
        query_texts=[question_en],
        n_results=5,
        include=["documents", "distances", "metadatas"],
    )
    docs = res["documents"][0] if res.get("documents") else []
    dists = res["distances"][0] if res.get("distances") else []
    metas = res["metadatas"][0] if res.get("metadatas") else []
    print("Distances:", [round(d, 2) for d in dists])
    kept = []
    for doc, dist, meta in zip(docs, dists, metas):
        if dist < DIST_CUTOFF:
            kept.append(clean_context(doc))
            print(f"   used [{(meta or {}).get('source', '?')}] {dist:.2f}")
    return "\n\n".join(kept)[:1500]


# ==========================
# STEP 4: ANSWER (ENGLISH)
# ==========================
def generate_answer(question_en, context):
    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": EXAMPLE_Q},
            {"role": "assistant", "content": EXAMPLE_A},
            {
                "role": "user",
                "content": f"Question:\n{question_en}\n\nContext:\n{context}",
            },
        ],
        options={"num_predict": 80, "temperature": 0},
        keep_alive="30m",
    )
    answer = response["message"]["content"].strip()
    for ch in ["*", "#", "•"]:
        answer = answer.replace(ch, "")
    # turn "June-July" into "June to July" and colons into full stops
    answer = re.sub(r"\b([A-Z][a-z]+)\s*-\s*([A-Z][a-z]+)\b", r"\1 to \2", answer)
    answer = answer.replace(":", ",")
    return answer


# ==========================
# ONE QUESTION
# ==========================
def run_once():
    if not os.path.exists(AUDIO_FILE):
        print("Audio file not found:", AUDIO_FILE)
        return

    os.makedirs("outputs", exist_ok=True)

    speech_file = AUDIO_FILE
    if USE_NOISE_REDUCTION:
        print("\nReducing noise...")
        timed("noise", clean_audio, AUDIO_FILE, CLEAN_FILE)
        speech_file = CLEAN_FILE

    print("Transcribing...")
    question, lang = timed("whisper", transcribe, speech_file)
    if not question:
        print("Could not get valid text from the audio.")
        say(NOT_UNDERSTOOD, lang)
        return
    print(f"\nFarmer question ({lang}): {question}")

    print("\nTranslating question...")
    question_en = timed("translate", translate, question, lang, "en")
    print("Question in English:", question_en)

    context = timed("retrieve", retrieve, question_en)
    if not context:
        print("No relevant farming information found.")
        say(NO_INFO, lang)
        return

    print("\nGenerating answer...")
    answer_en = timed("llm", generate_answer, question_en, context)
    print("Answer (English):", answer_en)

    if "do not have information" in answer_en.lower():
        say(NO_INFO, lang)
        return

    # Add a safety line when the answer talks about chemicals or doses
    if CHEMICAL_WORDS.search(answer_en):
        answer_en = answer_en.rstrip() + " " + CAUTION

    # Spell out quantities so they are translated and spoken, not dropped
    answer_en = numbers_to_words(answer_en)

    print("\nTranslating answer...")
    answer = timed("translate", translate, answer_en, "en", lang)
    print(f"\nFarmer Assistant ({lang}):\n{answer}")

    with open("answer.txt", "w", encoding="utf-8") as f:
        f.write(answer)

    print("\nGenerating speech...")
    speak_answer(answer, lang)


# ==========================
# WARM-UP: load the slow models once at startup
# ==========================
def ensure_ollama():
    """Make sure the Ollama server is running; start it if it is not."""
    try:
        ollama.list()
        return True
    except Exception:
        pass
    print("   Ollama is not running. Trying to start it...")
    try:
        flags = 0x08000000 if os.name == "nt" else 0      # no extra console window
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
    except FileNotFoundError:
        print("   Could not find the 'ollama' command. Install Ollama from ollama.com/download.")
        return False
    for _ in range(20):
        time.sleep(1)
        try:
            ollama.list()
            print("   Ollama started.")
            return True
        except Exception:
            continue
    print("   Ollama did not start. Open the Ollama app from the Start menu and try again.")
    return False


def warm_up():
    print("\nWarming up the translator, answer model and voices (one time)...")
    t = time.time()
    ensure_ollama()
    try:
        translate("ನಮಸ್ಕಾರ", "kn", "en")          # loads the translation model
        print(f"   translator ready ({time.time() - t:.0f}s)")
    except Exception as e:
        print("   translator warm-up failed:", e)
    try:
        ollama.chat(
            model="qwen2.5:3b",
            messages=[{"role": "user", "content": "Hi"}],
            options={"num_predict": 1},
            keep_alive="30m",
        )
        print(f"   answer model ready ({time.time() - t:.0f}s)")
    except Exception as e:
        print("   answer model warm-up failed (is Ollama running?):", e)
    try:
        from audio import tts as _tts
        _tts._load("kn")
        _tts._load("hi")
        print(f"   voices ready ({time.time() - t:.0f}s)")
    except Exception as e:
        print("   voice warm-up skipped:", e)


# ==========================
# LOOP: models stay loaded between questions
# ==========================
if __name__ == "__main__":
    warm_up()
    while True:
        try:
            input("\nPress Enter, then speak (the language is detected automatically, Ctrl+C to quit)...")
        except KeyboardInterrupt:
            print("\nBye")
            break
        subprocess.run([sys.executable, "audio/record.py"])
        total = time.time()
        run_once()
        print(f"\nTotal: {time.time() - total:.1f}s")