import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "models/nllb-200-distilled-600M"

# Dynamic int8 quantization: less memory and faster on CPU, small quality cost.
# Set to False if crop names or numbers start coming out wrong.
USE_QUANTIZATION = True

CODES = {
    "en": "eng_Latn",
    "kn": "kan_Knda",
    "hi": "hin_Deva",
}

# ==========================
# FARMING GLOSSARY
# NLLB gets some crop names wrong. Each entry is:
#   (word that must appear in the SOURCE text, regex of the WRONG output, correct text)
# Add a line whenever you find a new mistake (use test_translation.py to find them).
# ==========================

# English answer -> Kannada
EN_TO_KN = [
    ("paddy", "ಧಾನ್ಯ", "ಭತ್ತ"),
    ("field", "ಕ್ಷೇತ್ರ", "ಹೊಲ"),
    ("ragi", r"(?:ರೇಜಿ|ರಾಜಿ|ರಾಜೀ|ರಾಗಿ)", "ರಾಗಿ"),
    ("maize", "ಮಧುಮೇಹ", "ಮೆಕ್ಕೆಜೋಳ"),
    ("tomato", "ಹಣ್ಣಿನ ಸಸ್ಯ", "ಟೊಮೆಟೊ ಗಿಡ"),
    ("sugarcane", "ಸಕ್ಕರೆ ಗಲ್ಲದ", "ಕಬ್ಬಿನ"),
    ("areca", "ಅರೆಕಾ ಅಡಿಕೆ", "ಅಡಿಕೆ"),
    ("groundnut", "ನೆಲಗೊಬ್ಬರ", "ಕಡಲೆಕಾಯಿ"),
    ("jowar", "ಜವಾರ್", "ಜೋಳ"),
    ("coconut", "ಕೋಕೋಸ್", "ತೆಂಗಿನ"),
    ("chilli", "ಚಿಲ್ಲಿ", "ಮೆಣಸಿನಕಾಯಿ"),
    ("papaya", r"(?:ಪಪಾಯಾ|ಪಪ್ಪಾಯ|ಪಪ್ಪಾಯಿ)", "ಪಪ್ಪಾಯಿ"),
    ("plant", "ನೆಟ್ಟಿರಿ", "ನೆಡಿ"),
    ("apply", "ಅನ್ವಯಿಸಿ", "ಹಾಕಿ"),
    ("ask another question", "ಪ್ರಶ್ನೆ ಕೇಳು", "ಪ್ರಶ್ನೆ ಕೇಳಿ"),
]

# English answer -> Hindi
EN_TO_HI = [
    ("ragi", r"(?:रगी|राजी|रागी)", "रागी"),
    ("tomato", "टोमेटो", "टमाटर"),
    ("groundnut", "अखरोट", "मूंगफली"),
    ("areca", "अरेका नट", "सुपारी"),
    ("cotton", "सूती", "कपास"),
    ("chilli", "चिली", "मिर्च"),
    ("field", "क्षेत्र", "खेत"),
    ("treat", "इलाज करें", "उपचार करें"),
]

# Kannada question -> English (used for database search)
# The right-hand side lists both names so retrieval matches either spelling.
KN_TO_EN = [
    ("ಭತ್ತ", r"\b[Gg]rains?\b", "paddy"),
    ("ಹೊಲ", r"\b[Cc]onstituency\b", "field"),
    ("ರಾಗಿ", r"\b[Rr]ye\b|\b[Rr]agis\b", "ragi finger millet"),
    ("ಮೆಕ್ಕೆಜೋಳ", r"\b[Cc]orn\b", "maize"),
    ("ಜೋಳ", r"\b[Hh]aystacks?\b|\b[Jj]ell(?:y|ies)\b", "jowar sorghum"),
    ("ಗಿಡ", r"\b[Tt]rees?\b", "plant"),
    ("ಕಬ್ಬ", r"\b[Pp]eppers?\b", "sugarcane"),
    ("ಅಡಿಕೆ", r"(?<!areca )\b[Nn]ut\b", "areca nut"),
    ("ಮೆಣಸಿನಕಾಯಿ", r"\b[Cc]ucumbers?\b", "chilli"),
]

# Hindi question -> English (used for database search)
HI_TO_EN = [
    ("", r"\b[Tt]amat\w*\b", "tomato"),      # always on: catches tamatir, tamater...
    ("रागी", r"\b[Rr]ye\b|\b[Rr]agis\b", "ragi finger millet"),
    ("मक्का", r"\b[Cc]orn\b", "maize"),
    ("ज्वार", r"\b[Tt]idal waves?\b|\b[Hh]aystacks?\b", "jowar sorghum"),
    ("गन्न", r"\b[Cc]ane\b", "sugarcane"),
    ("मूंगफली", r"\b[Pp]eanuts?\b", "groundnut"),
]

FIXES = {
    ("en", "kn"): EN_TO_KN,
    ("en", "hi"): EN_TO_HI,
    ("kn", "en"): KN_TO_EN,
    ("hi", "en"): HI_TO_EN,
}

_tok = None
_model = None


def _load():
    global _tok, _model
    if _model is None:
        print("Loading NLLB translation model...")
        _tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).eval()
        if USE_QUANTIZATION:
            try:
                _model = torch.quantization.quantize_dynamic(
                    _model, {torch.nn.Linear}, dtype=torch.qint8
                )
                print("   NLLB quantized to int8.")
            except Exception as exc:
                print("   Quantization skipped:", exc)
    return _tok, _model


def _split_sentences(text):
    parts = re.split(r"(?<=[.!?।])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _translate_raw(text, src, tgt):
    tok, model = _load()
    tok.src_lang = CODES[src]
    target_id = tok.convert_tokens_to_ids(CODES[tgt])

    outputs = []
    for sent in _split_sentences(text):
        inputs = tok(sent, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                max_new_tokens=128,
                max_length=None,
                num_beams=1,
                no_repeat_ngram_size=3,
            )
        outputs.append(tok.batch_decode(generated, skip_special_tokens=True)[0])

    return " ".join(outputs)


def _apply_fixes(text, out, fixes):
    low = text.lower()
    for trigger, wrong, right in fixes:
        if trigger in low:
            out = re.sub(wrong, right, out, flags=re.IGNORECASE)
    return out


# ==========================
# CROP DICTIONARY
# If a Kannada or Hindi question mentions a known crop, the English question
# always carries the right English name, even if the translator got it wrong.
# Each entry: (words to look for, English name, words that cancel the match)
# ==========================
CROP_WORDS = {
    "kn": [
        (("ರಾಗಿ",), "ragi finger millet", ()),
        (("ಭತ್ತ",), "paddy rice", ()),
        (("ಮೆಕ್ಕೆ",), "maize corn", ()),
        (("ಜೋಳ",), "jowar sorghum", ("ಮೆಕ್ಕೆ",)),
        (("ಟೊಮೆಟೊ", "ಟೊಮ್ಯಾಟೊ", "ಟೊಮೇಟೊ"), "tomato", ()),
        (("ಈರುಳ್ಳಿ",), "onion", ()),
        (("ಆಲೂಗಡ್ಡೆ",), "potato", ()),
        (("ಕಬ್ಬ",), "sugarcane", ()),
        (("ಹತ್ತಿ",), "cotton", ()),
        (("ತೆಂಗು", "ತೆಂಗಿನ"), "coconut", ()),
        (("ಅಡಿಕೆ",), "areca nut", ()),
        (("ಕಡಲೆಕಾಯಿ", "ಶೇಂಗಾ", "ನೆಲಗಡಲೆ"), "groundnut peanut", ()),
        (("ಕಡಲೆ",), "chickpea bengal gram", ("ಕಡಲೆಕಾಯಿ",)),
        (("ಮೆಣಸಿನಕಾಯಿ",), "chilli", ()),
        (("ಬಾಳೆ",), "banana", ()),
        (("ಮಾವು",), "mango", ()),
        (("ದ್ರಾಕ್ಷಿ",), "grapes", ()),
        (("ಸೂರ್ಯಕಾಂತಿ",), "sunflower", ()),
        (("ತೊಗರಿ",), "red gram pigeon pea", ()),
        (("ಹೆಸರುಕಾಳು",), "green gram moong", ()),
        (("ಗೋಧಿ",), "wheat", ()),
        (("ಸಾಸಿವೆ",), "mustard", ()),
        (("ಎಳ್ಳು",), "sesame", ()),
        (("ಪಪ್ಪಾಯಿ",), "papaya", ()),
        (("ಕಾಫಿ",), "coffee", ()),
        (("ಏಲಕ್ಕಿ",), "cardamom", ()),
        (("ಬೆಂಡೆ",), "okra", ()),
        (("ಬದನೆ",), "brinjal eggplant", ()),
        (("ಎಲೆಕೋಸು",), "cabbage", ()),
    ],
    "hi": [
        (("रागी",), "ragi finger millet", ()),
        (("धान",), "paddy rice", ()),
        (("मक्का",), "maize corn", ()),
        (("ज्वार",), "jowar sorghum", ()),
        (("टमाटर",), "tomato", ()),
        (("प्याज",), "onion", ()),
        (("आलू",), "potato", ()),
        (("गन्न",), "sugarcane", ()),
        (("कपास",), "cotton", ()),
        (("नारियल",), "coconut", ()),
        (("सुपारी",), "areca nut", ()),
        (("मूंगफली",), "groundnut peanut", ()),
        (("चना",), "chickpea bengal gram", ()),
        (("मिर्च",), "chilli", ()),
        (("केला",), "banana", ()),
        (("अंगूर",), "grapes", ()),
        (("सूरजमुखी",), "sunflower", ()),
        (("अरहर", "तुअर"), "red gram pigeon pea", ()),
        (("उड़द", "उड़द"), "black gram urad", ()),
        (("मूंग",), "green gram moong", ("मूंगफली",)),
        (("गेहूं", "गेहूँ"), "wheat", ()),
        (("सरसों",), "mustard", ()),
        (("पपीता",), "papaya", ()),
        (("कॉफी",), "coffee", ()),
        (("इलायची",), "cardamom", ()),
        (("भिंडी",), "okra", ()),
        (("बैंगन",), "brinjal eggplant", ()),
        (("पत्तागोभी",), "cabbage", ()),
    ],
}


def _add_crop_hints(text, out, src):
    """Append the English name of any crop in the question that the translation lost."""
    missing = []
    for words, english, cancel in CROP_WORDS[src]:
        if any(c in text for c in cancel):
            continue
        if not any(w in text for w in words):
            continue
        first = english.split()[0]
        if first not in out.lower() and english not in missing:
            missing.append(english)
    if missing:
        out = out.rstrip() + " [crop: " + ", ".join(missing) + "]"
    return out


def translate(text, src, tgt):
    """Translate text between 'en', 'kn' and 'hi', then apply the farming glossary."""
    if src == tgt or not text.strip():
        return text
    out = _translate_raw(text, src, tgt)
    out = _apply_fixes(text, out, FIXES.get((src, tgt), []))
    # Robust crop-name normalization for common NLLB variants.
    if src == "en" and tgt == "kn":
        source = text.lower()
        crop_rules = [
            (r"\\bragi\\b", r"(?:ರೇಜಿ|ರಾಜಿ|ರಾಜೀ|ರಾಗಿ)", "ರಾಗಿ"),
            (r"\\bpapaya\\b", r"(?:ಪಪಾಯಾ|ಪಪ್ಪಾಯ|ಪಪ್ಪಾಯಿ)", "ಪಪ್ಪಾಯಿ"),
            (r"\\bpaddy|\\brice\\b", r"(?:ಧಾನ್ಯ|ಭತ್ತ)", "ಭತ್ತ"),
        ]
        for trigger, wrong, right in crop_rules:
            if re.search(trigger, source):
                out = re.sub(wrong, right, out)
    elif src == "en" and tgt == "hi":
        source = text.lower()
        for trigger, wrong, right in [
            (r"\\bragi\\b", r"(?:रगी|राजी|रागी)", "रागी"),
            (r"\\bpapaya\\b", r"(?:पपाया|पपीता|पपिता)", "पपीता"),
        ]:
            if re.search(trigger, source):
                out = re.sub(wrong, right, out)
    if tgt == "en" and src in CROP_WORDS:
        out = _add_crop_hints(text, out, src)
    return out


if __name__ == "__main__":
    en = "Prepare the field well before planting paddy. Give water regularly."
    print("KN:", translate(en, "en", "kn"))
    print("HI:", translate(en, "en", "hi"))
    print("KN->EN:", translate("ಭತ್ತಕ್ಕೆ ಯಾವಾಗ ನೀರು ಕೊಡಬೇಕು?", "kn", "en"))