import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Local folder (from download_models.py) or the Hugging Face repo name
MODEL_PATH = "models/nllb-200-distilled-600M"

CODES = {
    "en": "eng_Latn",
    "kn": "kan_Knda",
    "hi": "hin_Deva",
}

_tok = None
_model = None


def _load():
    global _tok, _model
    if _model is None:
        print("Loading NLLB translation model...")
        _tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).eval()
    return _tok, _model


def _split_sentences(text):
    parts = re.split(r"(?<=[.!?।])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def translate(text, src, tgt):
    """Translate text between 'en', 'kn' and 'hi'."""
    if src == tgt or not text.strip():
        return text

    tok, model = _load()
    tok.src_lang = CODES[src]
    target_id = tok.convert_tokens_to_ids(CODES[tgt])

    # Sentence by sentence = fewer dropped or repeated words
    sentences = _split_sentences(text)
    outputs = []
    for sent in sentences:
        inputs = tok(sent, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                max_new_tokens=128,
                num_beams=1,
            )
        outputs.append(tok.batch_decode(generated, skip_special_tokens=True)[0])

    return " ".join(outputs)


if __name__ == "__main__":
    en = "Prepare the field well before planting paddy. Give water regularly."
    print("KN:", translate(en, "en", "kn"))
    print("HI:", translate(en, "en", "hi"))
    print("KN->EN:", translate("ಭತ್ತಕ್ಕೆ ಯಾವಾಗ ನೀರು ಕೊಡಬೇಕು?", "kn", "en"))