"""Smart Farming Assistant: offline ASR + translation + ChromaDB RAG + Ollama + TTS.

Run from the project root: python app_corrected.py
After testing, you may rename it to app.py (keep a backup of your original).
"""

import os
import re
import sys
import time
import subprocess
from pathlib import Path

import chromadb
import ollama
from faster_whisper import WhisperModel, decode_audio
from translate import translate
from audio.tts import speak

try:
    import winsound  # Windows-only playback
except ImportError:
    winsound = None

# -------------------- Configuration --------------------
AUDIO_FILE = Path("audio/input.wav")
CLEAN_FILE = Path("audio/clean.wav")
OUTPUT_DIR = Path("outputs")
ANSWER_FILE = Path("answer.txt")
DB_PATH = "data/vector_db"
COLLECTION_NAME = "agriculture"
OLLAMA_MODEL = "qwen2.5:3b"
WHISPER_MODEL_SIZE = "small"
KN_MODEL_DIR = Path("models/whisper-kn-small-ct2")
USE_NOISE_REDUCTION = False
DIST_CUTOFF = float(os.getenv("RAG_DISTANCE_CUTOFF", "0.9"))  # Tune for your embedding metric.
ALLOWED = ["en", "kn", "hi"]
RANGES = {"kn": ("\u0C80", "\u0CFF"), "hi": ("\u0900", "\u097F")}

SYSTEM_PROMPT = """You are a Smart Agriculture Assistant for Indian farmers.
Reply only in English. Use simple words and at most four short sentences.
Use only the supplied context to answer the question. Do not invent facts.
Answer the exact question asked. If the context does not contain the answer,
reply exactly: I do not have information on this.
Spell out numbers and units in words when practical. Do not use bullet points,
markdown, or unnecessary references to documents or authors."""

NOT_UNDERSTOOD = "I could not understand the audio. Please speak clearly and ask again."
NO_INFO = "I do not have information on this. Please ask another question."
CAUTION = "Please confirm with your local agriculture officer before using any chemical."

CHEMICAL_WORDS = re.compile(
    r"\b(spray|pesticide|insecticide|fungicide|herbicide|chemical|dose|dosage|"
    r"gram|grams|millilit(?:er|re)s?|kilograms?|kg|lit(?:er|re)s?|urea)\b",
    re.IGNORECASE,
)

EN_WORDS = set("""the a an is are was what how when where which why who to of in on for with do does
 did can could should would i my me we you it this that and not at by from be have has had
 will about much many best time need ragi paddy rice maize corn wheat tomato onion potato cotton
 sugarcane groundnut jowar sorghum millet chilli coconut areca banana mango grape sow sowing plant
 planting grow fertilizer fertiliser urea manure compost pest pests disease water irrigation soil
 seed seeds crop crops spray weed weeds harvest yield leaf leaves yellow""".split())

PROMPTS = {
    "en": "Farming questions: ragi, paddy, maize, tomato, sowing, fertilizer, urea, pest, irrigation.",
    "kn": "ರಾಗಿ, ಭತ್ತ, ಮೆಕ್ಕೆಜೋಳ, ಟೊಮೆಟೊ, ಗೊಬ್ಬರ, ಕೀಟ, ನೀರು, ಬಿತ್ತನೆ",
    "hi": "धान, रागी, मक्का, टमाटर, खाद, कीट, सिंचाई, बुवाई",
}


def timed(label, fn, *args, **kwargs):
    start = time.time()
    result = fn(*args, **kwargs)
    print(f"   [{label}: {time.time() - start:.1f}s]")
    return result


def play(path):
    if winsound is None:
        print(f"Playback is not built in on this OS. Audio saved to: {path}")
        return
    try:
        winsound.PlaySound(str(path), winsound.SND_FILENAME)
    except Exception as exc:
        print("Playback error:", exc)


def speak_answer(text, lang):
    if lang in ("kn", "hi"):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = timed("tts", speak, text, lang, str(OUTPUT_DIR / "response.wav"))
        print("Audio saved:", path)
        if path and os.path.exists(path):
            play(path)
    else:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as exc:
            print("English TTS error:", exc)


def say(message_en, lang):
    try:
        message = translate(message_en, "en", lang) if lang != "en" else message_en
    except Exception as exc:
        print("Message translation failed; using English:", exc)
        message = message_en
    print("\nFarmer Assistant:", message)
    speak_answer(message, lang)


# -------------------- Number conversion --------------------
try:
    from num2words import num2words
except ImportError:
    num2words = None
    print("Tip: install num2words with: pip install num2words")


def numbers_to_words(text):
    """Convert common numeric values to words, if num2words is installed."""
    if num2words is None:
        return text

    def word(value):
        try:
            if "." in value:
                return num2words(float(value))
            return num2words(int(value))
        except (ValueError, TypeError, OverflowError):
            return value

    # Fertilizer grades such as 17-17-17 and 10:26:26.
    grade_pattern = re.compile(r"\b(\d{1,2})[-:](\d{1,2})[-:](\d{1,2})\b")
    text = grade_pattern.sub(lambda m: " ".join(word(part) for part in m.groups()), text)
    number = r"\d+(?:\.\d+)?"
    # Ranges such as 50-100 or 2.5–5.
    text = re.sub(
        rf"({number})\s*[-–]\s*({number})",
        lambda m: f"{word(m.group(1))} to {word(m.group(2))}",
        text,
    )
    text = re.sub(rf"({number})\s*%", lambda m: f"{word(m.group(1))} percent", text)
    return re.sub(number, lambda m: word(m.group(0)), text)


# -------------------- Optional noise reduction --------------------
def clean_audio(src, dst):
    import soundfile as sf
    import noisereduce as nr
    import numpy as np

    data, sample_rate = sf.read(str(src))
    if data.ndim > 1:
        data = data.mean(axis=1)
    cleaned = nr.reduce_noise(y=data.astype(np.float32), sr=sample_rate)
    sf.write(str(dst), cleaned, sample_rate)
    return str(dst)


# -------------------- Load speech models and database --------------------
print("Loading Whisper model...")
fw = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
if KN_MODEL_DIR.is_dir():
    print("Loading Kannada speech model...")
    fw_kn = WhisperModel(str(KN_MODEL_DIR), device="cpu", compute_type="int8")
else:
    print("Kannada-specific model not found; using the general model for Kannada.")
    fw_kn = fw
SPEECH = {"en": fw, "hi": fw, "kn": fw_kn}

client = chromadb.PersistentClient(path=DB_PATH)
try:
    collection = client.get_collection(COLLECTION_NAME)
except Exception as exc:
    raise RuntimeError(
        f"ChromaDB collection '{COLLECTION_NAME}' was not found at '{DB_PATH}'. "
        "Run your ingestion script only after backing up your existing database."
    ) from exc


def looks_valid(text, lang):
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 4:
        return False
    if lang == "en":
        ascii_ratio = sum(char.isascii() for char in letters) / len(letters)
        if ascii_ratio < 0.9:
            return False
        words = [re.sub(r"^[^\w]+|[^\w]+$", "", w.lower()) for w in text.split()]
        words = [w for w in words if w]
        # Avoid rejecting every valid sentence just because it uses unfamiliar crop names.
        return len(words) >= 2 and any(w in EN_WORDS for w in words)
    low, high = RANGES[lang]
    ratio = sum(low <= char <= high for char in letters) / len(letters)
    return ratio > 0.65


def run_transcription(audio, lang):
    model = SPEECH[lang]
    kwargs = {
        "language": lang,
        "beam_size": 1,
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "initial_prompt": PROMPTS[lang] if model is fw else None,
        "vad_filter": True,
    }
    if kwargs["initial_prompt"] is None:
        kwargs.pop("initial_prompt")
    try:
        segments, _info = model.transcribe(audio, **kwargs)
    except TypeError:
        # Compatibility with older faster-whisper versions.
        kwargs.pop("vad_filter", None)
        segments, _info = model.transcribe(audio, **kwargs)
    segments = list(segments)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    confidence = sum(segment.avg_logprob for segment in segments) / len(segments) if segments else -9.0
    return text, confidence


def transcribe(path):
    audio = decode_audio(str(path), sampling_rate=16000)
    _language, _probability, all_probs = fw.detect_language(audio)
    probs = {lang: prob for lang, prob in all_probs if lang in ALLOWED}
    for lang in ALLOWED:
        probs.setdefault(lang, 0.0)
    ranked = sorted(ALLOWED, key=lambda lang: probs[lang], reverse=True)
    print("Language probabilities:", {lang: round(probs[lang], 3) for lang in ALLOWED})

    candidates = []
    for lang in ranked:
        try:
            text, confidence = run_transcription(audio, lang)
        except Exception as exc:
            print(f"   [{lang} transcription error] {exc}")
            continue
        print(f"   [{lang} attempt] {text} (confidence {confidence:.2f})")
        if looks_valid(text, lang):
            prior = probs[lang] / max(sum(probs.values()), 1e-6)
            candidates.append((confidence + 1.5 * prior, lang, text))
        # Try at least the top two likely languages; stop early only for a strong prior.
        if len(candidates) and probs[lang] >= 0.80:
            break

    if not candidates:
        return "", ranked[0]
    _score, language, text = max(candidates, key=lambda item: item[0])
    return text, language


# -------------------- Retrieval and answer generation --------------------
def clean_context(text):
    # Keep the original regional-language text; it may contain information absent in English.
    return re.sub(r"[ \t]+", " ", text).strip()


def retrieve(question_en):
    if not question_en.strip() or collection.count() == 0:
        return ""
    result = collection.query(
        query_texts=[question_en],
        n_results=min(5, collection.count()),
        include=["documents", "distances", "metadatas"],
    )
    documents = (result.get("documents") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    numeric_distances = [float(distance) for distance in distances]
    print("Distances:", [round(distance, 3) for distance in numeric_distances])
    kept = []
    for document, distance, metadata in zip(documents, numeric_distances, metadatas):
        if document and distance < DIST_CUTOFF:
            kept.append(clean_context(document))
            source = (metadata or {}).get("source", "unknown")
            print(f"   used [{source}] distance={distance:.3f}")
    if not kept:
        print(
            f"No retrieved document passed RAG_DISTANCE_CUTOFF={DIST_CUTOFF}. "
            "Check the embedding model/metric and calibrate this threshold with test questions."
        )
    return "\n\n".join(kept)[:2500]


def generate_answer(question_en, context):
    if not context.strip():
        return NO_INFO
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question_en}\n\nContext:\n{context}"},
        ],
        options={"num_predict": 60, "temperature": 0},
        keep_alive="30m",
    )
    answer = response["message"]["content"].strip()
    answer = answer.replace("*", "").replace("#", "").replace("•", "")
    answer = re.sub(r"\b([A-Z][a-z]+)\s*-\s*([A-Z][a-z]+)\b", r"\1 to \2", answer)
    answer = answer.replace(":", ",")
    return answer


# -------------------- Ollama startup --------------------
def ensure_ollama():
    try:
        ollama.list()
        return True
    except Exception:
        pass
    print("Ollama is not responding. Trying to start it...")
    try:
        flags = 0x08000000 if os.name == "nt" else 0
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
        )
    except FileNotFoundError:
        print("Could not find Ollama. Install it from https://ollama.com/download")
        return False
    for _ in range(20):
        time.sleep(1)
        try:
            ollama.list()
            print("Ollama started.")
            return True
        except Exception:
            continue
    print("Ollama did not start. Open the Ollama app and try again.")
    return False


def warm_up():
    print("\nWarming up translation, answer model, and voices...")
    if not ensure_ollama():
        print("Warning: answer generation may fail until Ollama is running.")
    try:
        translate("ನಮಸ್ಕಾರ", "kn", "en")
        print("Translator ready.")
    except Exception as exc:
        print("Translator warm-up failed:", exc)
    try:
        ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "Reply with one word: Ready"}],
            options={"num_predict": 2, "temperature": 0},
            keep_alive="30m",
        )
        print("Answer model ready.")
    except Exception as exc:
        print("Answer model warm-up failed:", exc)
    try:
        from audio import tts as tts_module
        tts_module._load("kn")
        tts_module._load("hi")
        print("Kannada and Hindi voices ready.")
    except Exception as exc:
        print("Voice warm-up skipped:", exc)

GIVE_CONTEXT = ("ಗೊಬ್ಬರ", "ನೀರು", "ಔಷಧ", "ಕೀಟನಾಶಕ", "ಪೋಷಕ")


def fix_kannada(text):
    if "ಗಿಡ" in text and not any(w in text for w in GIVE_CONTEXT):
        text = text.replace("ನೀಡಬೇಕು", "ನೆಡಬೇಕು")
    return text


def run_once():
    if not AUDIO_FILE.exists():
        print("Audio file not found:", AUDIO_FILE)
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    speech_file = AUDIO_FILE

    if USE_NOISE_REDUCTION:
        print("\nReducing noise...")
        try:
            timed("noise reduction", clean_audio, AUDIO_FILE, CLEAN_FILE)
            speech_file = CLEAN_FILE
        except Exception as exc:
            print("Noise reduction failed; using original audio:", exc)

    print("Transcribing...")
    try:
        question, lang = timed("speech recognition", transcribe, speech_file)
    except Exception as exc:
        print("Speech recognition failed:", exc)
        return
    if not question:
        print("Could not get valid text from the audio.")
        say(NOT_UNDERSTOOD, lang)
        return

    if lang == "kn":
        question = fix_kannada(question)

    print(f"\nFarmer question ({lang}): {question}")
    try:
        question_en = timed("translate question", translate, question, lang, "en")
    except Exception as exc:
        print("Question translation failed:", exc)
        say(NOT_UNDERSTOOD, lang)
        return
    print("Question in English:", question_en)

    try:
        context = timed("retrieval", retrieve, question_en)
    except Exception as exc:
        print("ChromaDB retrieval failed:", exc)
        say(NO_INFO, lang)
        return
    if not context:
        print("No relevant farming information found.")
        say(NO_INFO, lang)
        return

    print("\nGenerating answer...")
    try:
        answer_en = timed("answer generation", generate_answer, question_en, context)
    except Exception as exc:
        print("Ollama answer generation failed:", exc)
        say(NO_INFO, lang)
        return

    print("Answer (English):", answer_en)
    if "i do not have information" in answer_en.lower():
        say(NO_INFO, lang)
        return

    caution_needed = bool(CHEMICAL_WORDS.search(answer_en))

    for pat, word in [
        (r"\bcm\b", "centimetres"),
        (r"\bkg\b", "kilograms"),
        (r"\bml\b", "millilitres"),
        (r"\bg\b", "grams"),
    ]:
        answer_en = re.sub(pat, word, answer_en, flags=re.IGNORECASE)

    answer_en = numbers_to_words(answer_en)

    if caution_needed:
        answer_en = answer_en.rstrip(" .") + ". " + CAUTION

    try:
        answer = timed("translate answer", translate, answer_en, "en", lang)
    except Exception as exc:
        print("Answer translation failed; using English:", exc)
        answer = answer_en
    print(f"\nFarmer Assistant ({lang}):\n{answer}")
    ANSWER_FILE.write_text(answer, encoding="utf-8")
    print("Answer saved to:", ANSWER_FILE)
    print("\nGenerating speech...")
    speak_answer(answer, lang)


if __name__ == "__main__":
    try:
        warm_up()
        while True:
            try:
                input("\nPress Enter, then speak (Ctrl+C to quit)...")
            except KeyboardInterrupt:
                print("\nBye.")
                break
            try:
                subprocess.run([sys.executable, "audio/record.py"], check=False)
                total_start = time.time()
                run_once()
                print(f"\nTotal: {time.time() - total_start:.1f}s")
            except KeyboardInterrupt:
                print("\nStopped.")
            except Exception as exc:
                print("Unexpected error; the next question can still be tried:", exc)
    except KeyboardInterrupt:
        print("\nBye.")
