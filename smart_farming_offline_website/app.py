"""Smart Farming Assistant: offline ASR + translation + ChromaDB RAG + Ollama + TTS.
Run from the project root:  python app.py
"""
import os
import re
import sys
import time
import subprocess
from pathlib import Path
try:
    import winsound  # Windows-only playback
except ImportError:
    winsound = None
import chromadb
import ollama
from faster_whisper import WhisperModel, decode_audio
from translate import translate
from audio.tts import speak
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
USE_GENERAL_FALLBACK = True      # answer from general knowledge when the database has nothing
KN_USE_PROMPT = True             # give Kannada ASR farming vocabulary hints
FORCE_LANGUAGE = os.getenv("SFA_FORCE_LANGUAGE", "auto").lower()  # auto, kn, hi, en
DIST_CUTOFF = float(os.getenv("RAG_DISTANCE_CUTOFF", "0.9"))
CONTEXT_CHARS = 800              # shorter context = faster answers
ALLOWED = ["en", "kn", "hi"]
RANGES = {"kn": ("\u0C80", "\u0CFF"), "hi": ("\u0900", "\u097F")}
SYSTEM_PROMPT = """You are a Smart Agriculture Assistant for Indian farmers.
Reply ONLY in English. Use simple words and at most two short sentences.
Use ONLY the supplied context to answer. Never copy the context as a list.
Answer only what was asked. Do not list variety names unless the question asks for them.
Do not use words like ring or cultural practices. Say "around the plant".
The answer must match the exact crop and intent in the question. Never answer about a different crop. Ignore any context that does not directly answer the question. If the context does not answer the question, reply exactly: I do not have information on this.
Write numbers and units in words. Do not use abbreviations, bullet points, symbols or markdown."""
EXAMPLE_Q = "Question: When to sow ragi?\n\nContext:\nSowing time: July to August. Varieties: GPU 28, MR 1."
EXAMPLE_A = "Sow ragi in July or August."
GENERAL_PROMPT = """You are a Smart Agriculture Assistant for Indian farmers.
Reply ONLY in English. Use simple words and at most three short sentences.
Answer only farming questions: crops, soil, water, pests, fertilizer, livestock.
If the question is not about farming, reply exactly: I can only help with farming questions.
Give general guidance only. Never give chemical names or doses.
Write numbers in words. Do not use bullet points, symbols or markdown."""
NOT_UNDERSTOOD = "I could not understand the audio. Please speak clearly and ask again."
NO_INFO = "I do not have information on this. Please ask another question."
ASK_CROP = "Which crop are you asking about? Please say the crop name and ask again."
CAUTION = "Please confirm with your local agriculture officer before using any chemical."
GENERAL_NOTE = "This is general advice. Please confirm with your local agriculture officer."  # retained for optional use; not appended automatically
CHEMICAL_WORDS = re.compile(
    r"\b(spray|pesticide|insecticide|fungicide|herbicide|chemical|dose|dosage|"
    r"gram|grams|millilit(?:er|re)s?|kilograms?|kg|lit(?:er|re)s?|urea)\b",
    re.IGNORECASE,
)
UNIT_REPLACEMENTS = [
    (r"\bcm\b", "centimetres"),
    (r"\bkg\b", "kilograms"),
    (r"\bml\b", "millilitres"),
    (r"\bg\b", "grams"),
]
CROPS = set("""ragi paddy rice maize corn wheat tomato onion potato cotton sugarcane
groundnut peanut jowar sorghum millet chilli coconut areca banana mango grapes sunflower
papaya coffee cardamom okra brinjal cabbage mustard sesame chickpea gram""".split())
FARM_WORDS = set("""soil fertilizer fertiliser manure compost pest pests disease irrigation
cow buffalo goat sheep poultry hen seed seeds crop crops weed weeds harvest""".split())
EN_WORDS = set("""the a an is are was what how when where which why who to of in on for with do does
did can could should would i my me we you it this that and not at by from be have has had
will about much many best time need ragi paddy rice maize corn wheat tomato onion potato cotton
sugarcane groundnut jowar sorghum millet chilli coconut areca banana mango grape sow sowing plant
planting grow fertilizer fertiliser urea manure compost pest pests disease water irrigation soil
seed seeds crop crops spray weed weeds harvest yield leaf leaves yellow""".split())
PROMPTS = {
    "en": "Farming questions: ragi, paddy, maize, tomato, sowing, fertilizer, urea, pest, irrigation.",
    "kn": "ರಾಗಿ, ಭತ್ತ, ಮೆಕ್ಕೆಜೋಳ, ಟೊಮೆಟೊ, ಪಪ್ಪಾಯಿ, ಈರುಳ್ಳಿ, ಬಾಳೆ, ಅಡಿಕೆ, ಗೊಬ್ಬರ, ಕೀಟ, ನೀರು, ಬಿತ್ತನೆ",
    "hi": "धान, रागी, मक्का, टमाटर, खाद, कीट, सिंचाई, बुवाई",
}
GIVE_CONTEXT = ("ಗೊಬ್ಬರ", "ನೀರು", "ಔಷಧ", "ಕೀಟನಾಶಕ", "ಪೋಷಕ")
# -------------------- Small helpers --------------------
_cache = {}
def cached_translate(text, src, tgt):
    """Translate fixed sentences only once; reuse the result afterwards."""
    key = (text, src, tgt)
    if key not in _cache:
        _cache[key] = translate(text, src, tgt)
    return _cache[key]
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
def prepare_tts_text(text, lang):
    """Prepare Kannada/Hindi text so numerals and common English units are spoken.

    Keep this limited to the spoken copy; the displayed/saved answer remains unchanged.
    """
    if lang not in ("kn", "hi"):
        return text

    # Replace symbols and English unit names with words in the selected language.
    if lang == "kn":
        replacements = [
            (r"\bdegrees?\s+Fahrenheit\b|\bFahrenheit\b|°\s*F\b|ಫ್ಯಾರನ್ಹೈಟ್ಗಳ?|ಫ್ಯಾರನ್‌ಹೀಟ್ಗಳ?", "ಡಿಗ್ರಿ ಫ್ಯಾರನ್‌ಹೀಟ್"),
            (r"\bdegrees?\s+Celsius\b|\bCelsius\b|°\s*C\b|ಸೆಲ್ಸಿಯಸ್", "ಡಿಗ್ರಿ ಸೆಲ್ಸಿಯಸ್"),
            (r"\bpercent(age)?\b|%", "ಶೇಕಡಾ"),
            (r"\bkilograms?\b|\bkg\b", "ಕಿಲೋಗ್ರಾಂ"),
            (r"\bgrams?\b|\bg\b", "ಗ್ರಾಂ"),
            (r"\bmillilit(?:er|re)s?\b|\bml\b", "ಮಿಲಿಲೀಟರ್"),
            (r"\blit(?:er|re)s?\b|\bl\b", "ಲೀಟರ್"),
            (r"\bdegrees?\b|°", "ಡಿಗ್ರಿ"),
            (r"\bFahrenheit\b", "ಫ್ಯಾರನ್‌ಹೀಟ್"),
            (r"\bCelsius\b", "ಸೆಲ್ಸಿಯಸ್"),
        ]
    else:
        replacements = [
            (r"\bdegrees?\s+Fahrenheit\b|\bFahrenheit\b|°\s*F\b", "डिग्री फ़ारेनहाइट"),
            (r"\bdegrees?\s+Celsius\b|\bCelsius\b|°\s*C\b", "डिग्री सेल्सियस"),
            (r"\bpercent(age)?\b|%", "प्रतिशत"),
            (r"\bkilograms?\b|\bkg\b", "किलोग्राम"),
            (r"\bgrams?\b|\bg\b", "ग्राम"),
            (r"\bmillilit(?:er|re)s?\b|\bml\b", "मिलीलीटर"),
            (r"\blit(?:er|re)s?\b|\bl\b", "लीटर"),
            (r"\bdegrees?\b|°", "डिग्री"),
        ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # TTS often reads ASCII digits poorly in Kannada/Hindi. Convert short numbers
    # into words in the target language. Never convert long phone-number-like strings.
    def spoken_number(match):
        token = match.group(0)
        if len(token.replace(".", "")) >= 7:
            return token
        try:
            if num2words is None:
                english_words = token
            else:
                value = float(token) if "." in token else int(token)
                if lang == "hi":
                    # num2words has Hindi support; use it directly where available.
                    try:
                        return num2words(value, lang="hi")
                    except (NotImplementedError, TypeError, ValueError):
                        pass
                english_words = num2words(value)
            localized = cached_translate(english_words, "en", lang).strip()
            return localized or token
        except Exception:
            return token

    # Match digits even when surrounded by non-Latin scripts or punctuation.
    text = re.sub(r"(?<!\d)\d+(?:\.\d+)?(?!\d)", spoken_number, text)
    # Clean up spacing around units for more natural TTS pauses.
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text


def speak_answer(text, lang):
    if lang in ("kn", "hi"):
        text_to_speak = prepare_tts_text(text, lang)
        if text_to_speak != text:
            print("   TTS-friendly text:", text_to_speak)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = timed("tts", speak, text_to_speak, lang, str(OUTPUT_DIR / "response.wav"))
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
    """Print and speak a fixed message in the farmer's language."""
    try:
        message = cached_translate(message_en, "en", lang) if lang != "en" else message_en
    except Exception as exc:
        print("Message translation failed; using English:", exc)
        message = message_en
    print("\nFarmer Assistant:", message)
    speak_answer(message, lang)
def fix_kannada(text):
    """Whisper often hears 'ನೆಡಬೇಕು' (plant) as 'ನೀಡಬೇಕು' (give)."""
    if "ಗಿಡ" in text and not any(w in text for w in GIVE_CONTEXT):
        text = text.replace("ನೀಡಬೇಕು", "ನೆಡಬೇಕು")
    return text


def fix_hindi(text):
    """Normalize common Hindi Whisper errors for the six demo questions."""
    text = text.strip()
    text = re.sub(r"कोंची|कोनची|कौनसी|कोनसी", "कौन सी", text)
    text = re.sub(r"काद", "खाद", text)
    # Correct only a whole mistaken word; do not damage the correct word पपीता.
    text = re.sub(r"(?<!\S)(?:पीता|पपीटा)(?!\S)", "पपीता", text)
    text = re.sub(r"गुब्बर|गुबर", "खाद", text)
    # In the known ragi-fertilizer demo phrase, Whisper may hear खाद as काट.
    if "रागी" in text and re.search(r"काट|काद", text):
        text = re.sub(r"काट|काद", "खाद", text)
    # Correct common noisy transcripts from the demo recordings.
    if re.search(r"दान में", text) and re.search(r"कौन|कुँ|कों|काद|खाद|डाल", text):
        text = "धान में कौन सी खाद डालनी है?"
    if "पपी" in text and re.search(r"भगानी|तख", text):
        text = "पपीता कब लगानी है?"
    # Hindi wheat-fertilizer demo: Whisper can hear गेहूँ as "के हु" and
    # कौन सी खाद अच्छी as "काँची काद अची". Correct only when the surrounding
    # words strongly indicate a fertilizer question, so unrelated speech is untouched.
    wheat_noisy = re.search(r"गेहू|गेहूँ|गेंहू|गेहु|गहु|गिहु|के\s*हु|किहु", text)
    fertilizer_intent = re.search(r"खाद|काद|काँची|कांची|फसल|अची|अच्छी|चाहिए", text)
    if wheat_noisy and fertilizer_intent:
        return "गेहूँ की फसल के लिए कौन सी खाद अच्छी है?"
    return text


def known_question_to_english(text, lang):
    """Normalize known demo questions before machine translation.

    Returns a clear English question for known Kannada, Hindi, or noisy
    romanized transcripts; returns None for other questions.
    """
    raw = re.sub(r"\s+", " ", text.strip())
    normalized = re.sub(r"[?!।.]+$", "", raw).strip()
    low = normalized.lower()

    # Kannada script: match distinctive crop + intent words from the demo set.
    if lang == "kn":
        if re.search(r"ಮೆಕ್ಕೆಜೋಳ|ಮೆಕ್ಕೆ ಜೋಳ|ಮಕ್ಕಜೋಳ", normalized) and re.search(r"ಯಾವಾಗ|ಬಿತ್ತನೆ|ಬಿತ್ತಬೇಕು", normalized):
            return "When should I sow maize?"
        if re.search(r"ಭತ್ತ|ಭತ್ತಕ್ಕೆ", normalized) and re.search(r"ಗೊಬ್ಬರ|ಯಾವ ಗೊಬ್ಬರ", normalized):
            return "Which fertilizer should I apply to paddy?"
        if re.search(r"ಅಡಿಕೆ|ಅಡಕ್ಕೆ|ಅಡಕೆ", normalized) and re.search(r"ಗೊಬ್ಬರ|ಯಾವ ಗೊಬ್ಬರ", normalized):
            return "Which fertilizer should I apply to arecanut?"
        if re.search(r"ರಾಗಿ|ರಾಗಿಗೆ", normalized) and re.search(r"ಗೊಬ್ಬರ", normalized):
            return "Which fertilizer should I apply to ragi?"
        if re.search(r"ರಾಗಿಯಲ್ಲಿ|ರಾಗಿ", normalized) and re.search(r"ಕಳೆ|ನಿಯಂತ್ರಣ", normalized):
            return "How can I control weeds in ragi?"
        if re.search(r"ಟೊಮೆಟೊ|ಟೊಮಾಟೊ", normalized) and re.search(r"ಕೀಟ|ಹುಳು|ಏನು ಮಾಡ", normalized):
            return "How can I control pests in tomato?"
        if re.search(r"ಪಪ್ಪಾಯಿ", normalized) and re.search(r"ಯಾವಾಗ|ನೆಡ|ನಾಟಿ", normalized):
            return "When is the best time to plant papaya?"

    if lang == "hi":
        # Exact template for wheat fertilizer questions, before NLLB can confuse wheat with cotton.
        if re.search(r"गेहू|गेहूँ|गेंहू|गेहु|गहु|गिहु|के\s*हु|किहु", normalized) and re.search(r"खाद|काद|काँची|कांची|कौन|अची|अच्छी|फसल", normalized):
            return "Which fertilizer is suitable for wheat?"
        if "रागी" in normalized and re.search(r"खाद|काद|गुब्बर|गुबर|गुबारा", normalized) and re.search(r"कौन|कोंची|कोनची|डाल|चाहिए|है", normalized):
            return "Which fertilizer should I apply to ragi?"
        if "रागी" in normalized and re.search(r"कब|समय|लग|बो", normalized):
            return "When should I sow ragi?"
        if re.search(r"पपीता|पपीटा|(?<!\S)पीता(?!\S)", normalized) and re.search(r"कब|समय|लग|रोप", normalized):
            return "When is the best time to plant papaya?"
        if re.search(r"मक्का|मकका", normalized) and re.search(r"कब|समय|बो", normalized):
            return "When should I sow maize?"
        if re.search(r"धान|भत्त|भात", normalized) and re.search(r"खाद|काद|गुब्बर|गुबर", normalized):
            return "Which fertilizer should I apply to paddy?"
        if re.search(r"टमाटर", normalized) and re.search(r"कीड़े|कीट|कीड़ा|किरा|क्या करें", normalized):
            return "How can I control pests in tomato?"
        if "रागी" in normalized and re.search(r"खरपतवार|खरपत|कले|खर", normalized):
            return "How can I control weeds in ragi?"

    # Noisy Kannada transcription written in Latin letters.
    if lang == "en" and re.search(r"\b(ragi|raagi)\b", low):
        if re.search(r"gub+a?r|gubbar|gubara|gub(b|b)ara", low) and re.search(r"yawa|yaava|beku|ha", low):
            return "Which fertilizer should I apply to ragi?"
        if re.search(r"yavaaga|yavaga|bitt|sow", low):
            return "When should I sow ragi?"
    if lang == "en" and re.search(r"adakke|adike|areca", low) and re.search(r"gub+a?r|gubbar|gubara|fertil", low):
        return "Which fertilizer should I apply to arecanut?"
    if lang == "en" and re.search(r"\b(papaya|papita|pappayi)\b", low) and re.search(r"when|yavaga|yaavaga|ne", low):
        return "When is the best time to plant papaya?"
    return None


def mentions_crop(question_en):
    low = question_en.lower()
    return any(re.search(rf"\b{w}s?\b", low) for w in CROPS | FARM_WORDS)
# -------------------- Number conversion --------------------
try:
    from num2words import num2words
except ImportError:
    num2words = None
    print("Tip: install num2words with: pip install num2words")
def numbers_to_words(text):
    """'20 grams' -> 'twenty grams'; '50-100' -> 'fifty to one hundred'."""
    if num2words is None:
        return text
    def word(value):
        try:
            if "." in value:
                return num2words(float(value))
            return num2words(int(value))
        except (ValueError, TypeError, OverflowError):
            return value
    number = r"\d+(?:\.\d+)?"
    text = re.sub(
        r"\b(\d{1,2})[-:](\d{1,2})[-:](\d{1,2})\b",
        lambda m: " ".join(word(part) for part in m.groups()),
        text,
    )
    text = re.sub(
        rf"({number})\s*[-–]\s*({number})",
        lambda m: f"{word(m.group(1))} to {word(m.group(2))}",
        text,
    )
    text = re.sub(rf"({number})\s*%", lambda m: f"{word(m.group(1))} percent", text)
    # Leave long digit strings (such as phone numbers) unchanged.
    return re.sub(number, lambda m: m.group(0) if len(m.group(0).replace(".", "")) >= 7 else word(m.group(0)), text)
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
        f"ChromaDB collection '{COLLECTION_NAME}' was not found at '{DB_PATH}'."
    ) from exc
# -------------------- Speech to text --------------------
def looks_valid(text, lang):
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 4:
        return False
    if lang == "en":
        if sum(char.isascii() for char in letters) / len(letters) < 0.9:
            return False
        words = [re.sub(r"^[^\w]+|[^\w]+$", "", w.lower()) for w in text.split()]
        words = [w for w in words if w]
        return len(words) >= 2 and any(w in EN_WORDS for w in words)
    low, high = RANGES[lang]
    return sum(low <= char <= high for char in letters) / len(letters) > 0.65
def run_transcription(audio, lang):
    """Transcribe in one language. Returns (text, confidence)."""
    model = SPEECH[lang]
    kwargs = {
        "language": lang,
        "beam_size": 1,
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "max_new_tokens": 200,   # Kannada script needs many tokens per word
        "vad_filter": True,
    }
    if model is fw or KN_USE_PROMPT:
        kwargs["initial_prompt"] = PROMPTS[lang]
    try:
        segments, _info = model.transcribe(audio, **kwargs)
    except TypeError:
        # Older faster-whisper versions do not know these options.
        kwargs.pop("vad_filter", None)
        kwargs.pop("max_new_tokens", None)
        segments, _info = model.transcribe(audio, **kwargs)
    segments = list(segments)
    text = " ".join(s.text.strip() for s in segments).strip()
    confidence = sum(s.avg_logprob for s in segments) / len(segments) if segments else -9.0
    return text, confidence
def transcribe(path):
    """Transcribe audio. SFA_FORCE_LANGUAGE=kn/hi/en bypasses unreliable auto-detection."""
    audio = decode_audio(str(path), sampling_rate=16000)

    # Fast, reliable demo workaround: explicitly force the spoken language.
    # In PowerShell, set $env:SFA_FORCE_LANGUAGE="kn" before running app.py.
    forced = FORCE_LANGUAGE if FORCE_LANGUAGE in ALLOWED else "auto"
    if forced != "auto":
        print(f"Forced speech language: {forced}")
        text, confidence = run_transcription(audio, forced)
        print(f"   [{forced} forced attempt] {text} (confidence {confidence:.2f})")
        if not text.strip():
            return "", forced
        return text.strip(), forced

    _language, _probability, all_probs = fw.detect_language(audio)
    probs = {lang: prob for lang, prob in all_probs if lang in ALLOWED}
    for lang in ALLOWED:
        probs.setdefault(lang, 0.0)
    ranked = sorted(ALLOWED, key=lambda lang: probs[lang], reverse=True)
    print("Language probabilities:", {lang: round(probs[lang], 3) for lang in ALLOWED})

    candidates = []
    # Do not stop early: short Kannada speech is often misclassified as English/Hindi.
    for lang in ranked:
        try:
            text, confidence = run_transcription(audio, lang)
        except Exception as exc:
            print(f"   [{lang} transcription error] {exc}")
            continue
        print(f"   [{lang} attempt] {text} (confidence {confidence:.2f})")
        if looks_valid(text, lang):
            prior = probs[lang] / max(sum(probs.values()), 1e-6)
            score = confidence + 1.5 * prior
            # A valid Kannada-script result should not lose only because the
            # language detector assigned Kannada a near-zero probability.
            if lang == "kn":
                score += 0.35
            candidates.append((score, lang, text))

    if not candidates:
        return "", ranked[0]
    _score, language, text = max(candidates, key=lambda item: item[0])
    return text, language

# -------------------- Retrieval and answer generation --------------------
def clean_context(text):
    # KCC records contain Kannada/Hindi text next to the English. The
    # English-only model does not need it, and it wastes space.
    text = re.sub(r"[\u0C80-\u0CFF\u0900-\u097F]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
def retrieve(question_en):
    """Retrieve context, rejecting documents about a different crop when crop is explicit."""
    if not question_en.strip() or collection.count() == 0:
        return ""
    result = collection.query(
        query_texts=[question_en],
        n_results=min(8, collection.count()),
        include=["documents", "distances", "metadatas"],
    )
    documents = (result.get("documents") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = [float(d) for d in distances]
    print("Distances:", [round(d, 3) for d in distances])
    for i, document in enumerate(documents):
        print(f"\nTop document {i + 1}:")
        print((document or "")[:300])

    q = question_en.lower()
    crop_aliases = {
        "arecanut": ("arecanut", "areca nut", "betel nut", "areca"),
        "paddy": ("paddy", "rice crop", "rice plant", "paddy rice"),
        "ragi": ("ragi", "finger millet", "eleusine coracana"),
        "maize": ("maize", "corn"),
        "papaya": ("papaya", "papita"),
        "tomato": ("tomato",),
        "wheat": ("wheat", "wheat crop", "triticum aestivum"),
    }
    target_crop = next((crop for crop, aliases in crop_aliases.items()
                        if any(alias in q for alias in aliases)), None)
    aliases = crop_aliases.get(target_crop, ()) if target_crop else ()
    kept = []
    # Use a stricter cutoff for retrieval. Crop-specific questions must not use
    # a document about another crop just because its embedding distance is close.
    cutoff = min(DIST_CUTOFF, 0.90)
    for document, distance, metadata in zip(documents, distances, metadatas):
        if not document or distance >= cutoff:
            continue
        doc_low = document.lower()
        if target_crop:
            if not any(alias in doc_low for alias in aliases):
                print(f"   skipped unrelated crop document (distance={distance:.3f})")
                continue
        kept.append(clean_context(document))
        print(f"   used [{(metadata or {}).get('source', 'unknown')}] distance={distance:.3f}")
        if len(kept) >= 3:
            break
    if not kept:
        print(f"No relevant document passed crop matching and distance cutoff {cutoff}.")
    return "\n\n".join(kept)[:CONTEXT_CHARS]
def _tidy(answer, question_en=""):
    answer = answer.strip()
    for ch in ("*", "#", "•"):
        answer = answer.replace(ch, "")
    # "June-July" -> "June to July"
    answer = re.sub(r"\b([A-Z][a-z]+)\s*-\s*([A-Z][a-z]+)\b", r"\1 to \2", answer)
    # "in June to July" -> "from June to July"
    answer = re.sub(r"\bin ([A-Z][a-z]+) to ([A-Z][a-z]+)\b", r"from \1 to \2", answer)
    # drop variety lists unless the farmer asked about varieties
    if "variet" not in question_en.lower():
        sentences = re.split(r"(?<=[.!?])\s+", answer)
        kept = [s for s in sentences if "variet" not in s.lower()]
        answer = " ".join(kept or sentences[:1])
    # colons confuse the translator
    return answer.replace(":", ",")
def generate_answer(question_en, context):
    if not context.strip():
        return NO_INFO
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": EXAMPLE_Q},
            {"role": "assistant", "content": EXAMPLE_A},
            {"role": "user", "content": f"Question: {question_en}\n\nContext:\n{context}"},
        ],
        options={"num_predict": 60, "temperature": 0},
        keep_alive="30m",
    )
    return _tidy(response["message"]["content"], question_en)
def generate_general_answer(question_en):
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": GENERAL_PROMPT},
            {"role": "user", "content": question_en},
        ],
        options={"num_predict": 80, "temperature": 0},
        keep_alive="30m",
    )
    return _tidy(response["message"]["content"])
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
# -------------------- One question --------------------
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
    elif lang == "hi":
        question = fix_hindi(question)
    print(f"\nFarmer question ({lang}): {question}")
    known_question = known_question_to_english(question, lang)
    if known_question:
        question_en = known_question
        print("Using matched farming-question template.")
    else:
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
        context = ""
    answer_en = ""
    general = False
    try:
        if context:
            print("\nGenerating answer...")
            answer_en = timed("answer generation", generate_answer, question_en, context)
            print("Answer (English):", answer_en)
    except Exception as exc:
        print("Ollama answer generation failed:", exc)
        answer_en = ""
    # Nothing useful from the database: try general farming knowledge.
    if not answer_en or "do not have information" in answer_en.lower():
        if not USE_GENERAL_FALLBACK:
            say(NO_INFO, lang)
            return
        if not mentions_crop(question_en):
            say(ASK_CROP, lang)
            return
        print("\nNo database answer. Using general knowledge...")
        try:
            answer_en = timed("general answer", generate_general_answer, question_en)
        except Exception as exc:
            print("General answer failed:", exc)
            say(NO_INFO, lang)
            return
        print("Answer (English):", answer_en)
        if "only help with farming" in answer_en.lower():
            say(answer_en, lang)
            return
        general = True
    caution_needed = bool(CHEMICAL_WORDS.search(answer_en))
    for pattern, replacement in UNIT_REPLACEMENTS:
        answer_en = re.sub(pattern, replacement, answer_en)
    answer_en = numbers_to_words(answer_en)
    try:
        answer = timed("translate answer", translate, answer_en, "en", lang)
        if caution_needed:
            answer = answer.rstrip() + " " + cached_translate(CAUTION, "en", lang)
    except Exception as exc:
        print("Answer translation failed; using English:", exc)
        answer = answer_en
        if caution_needed:
            answer += " " + CAUTION
    print(f"\nFarmer Assistant ({lang}):\n{answer}")
    ANSWER_FILE.write_text(answer, encoding="utf-8")
    print("Answer saved to:", ANSWER_FILE)
    print("\nGenerating speech...")
    speak_answer(answer, lang)
# -------------------- Main loop --------------------
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
