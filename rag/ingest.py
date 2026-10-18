"""
Clean ingestion for the Smart Farming Assistant.

Put this file at rag/ingest.py (back up your old one) and run from the project root:
    python rag/ingest.py

Needs:  pip install pypdf pandas pyarrow
Optional (multilingual search):  pip install sentence-transformers

Tips:
- Any folder or file whose name starts with "_" is skipped. Move bad sources into
  datasets/farming_pdfs/_rejected/ instead of deleting them.
- Every chunk is stored with its source file name, so you can trace a bad answer
  back to the document it came from.
"""
import os
import re
import json
import shutil
import hashlib

import chromadb

# ==========================
# SETTINGS
# ==========================
DATASET_FOLDER = "datasets/farming_pdfs"
DB_PATH = "data/vector_db"
COLLECTION = "agriculture"

# None = Chroma's default English embedder (fine, because the app translates
# questions to English first).
# Use "paraphrase-multilingual-MiniLM-L12-v2" if your Q&A data is in Kannada/Hindi.
# If you set this, app.py must open the collection with the SAME embedding function.
EMBED_MODEL = None

MAX_ROWS_PER_FILE = 40000   # cap for huge parquet/csv files
CHUNK_CHARS = 800           # target size of a PDF chunk
BATCH = 256

QUESTION_COLS = ["question", "questions", "query", "querytext", "instruction", "input", "prompt"]
ANSWER_COLS = ["answer", "answers", "response", "output", "kccans", "reply"]

STOP = set(
    "the of and in to is for are with a an on as by be or from that this it at "
    "can should plant crop soil water seed seeds field farmers yield land".split()
)


# ==========================
# QUALITY CHECKS
# ==========================
def script_ratio(t, lo, hi):
    letters = [c for c in t if c.isalpha()]
    return sum(lo <= c <= hi for c in letters) / max(1, len(letters))


def is_readable(t):
    """Lenient check used for short Q&A records."""
    t = t.strip()
    if len(t) < 20:
        return False
    if re.search(r"PASSAGE:|INSTRUCTION:|Fill in the blank", t, re.I):
        return False   # exam-style records, not farming advice
    if script_ratio(t, "\u0C80", "\u0CFF") > 0.5 or script_ratio(t, "\u0900", "\u097F") > 0.5:
        return True
    if script_ratio(t, "\u0B80", "\u0BFF") > 0.3:        # Tamil, Unicode
        return False
    words = re.findall(r"[A-Za-z]+", t)
    if len(words) < 3:
        return False
    junk = len(re.findall(r"[A-Za-z][;\[\]\^][A-Za-z]", t)) / max(1, len(words))
    digits = sum(c.isdigit() for c in t) / len(t)
    return junk < 0.02 and digits < 0.4


def is_good_prose(t):
    """Stricter check used for PDF chunks."""
    if len(t.strip()) < 80 or not is_readable(t):
        return False
    if script_ratio(t, "\u0C80", "\u0CFF") > 0.5 or script_ratio(t, "\u0900", "\u097F") > 0.5:
        return True
    words = re.findall(r"[A-Za-z]+", t.lower())
    stop_ratio = sum(w in STOP for w in words) / max(1, len(words))
    digits = sum(c.isdigit() for c in t) / len(t)
    return stop_ratio >= 0.15 and digits < 0.25


# ==========================
# CHUNKING
# ==========================
def clean_pdf_text(text):
    text = re.sub(r"-\n(?=[a-z])", "", text)        # join hyphenated line breaks
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_text(text):
    sentences = re.split(r"(?<=[.!?\u0964])\s+|\n\n", text)
    chunks, current = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(current) + len(s) + 1 > CHUNK_CHARS and current:
            chunks.append(current)
            current = current[-120:] + " " + s     # small overlap
        else:
            current = (current + " " + s).strip()
    if current:
        chunks.append(current)
    return chunks


# ==========================
# LOADERS
# ==========================
def fingerprint(text):
    """Same hash for records that differ only in spacing or letter case."""
    norm = re.sub(r"\s+", "", text.lower())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def pdf_chunks(path):
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages, bad_pages = [], 0
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            bad_pages += 1          # some PDFs have damaged fonts on a few pages
    if bad_pages:
        print(f"   skipped {bad_pages} unreadable pages")
    return chunk_text(clean_pdf_text("\n".join(pages)))


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue            # skip a damaged line
            if len(rows) >= MAX_ROWS_PER_FILE:
                break
    return rows


def load_records(path):
    """Return a list of dicts from json / jsonl / csv / parquet."""
    low = path.lower()
    if low.endswith(".jsonl"):
        return read_jsonl(path)
    if low.endswith(".json"):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            return read_jsonl(path)     # the file is really JSON Lines
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
        return data[:MAX_ROWS_PER_FILE] if isinstance(data, list) else []
    import pandas as pd
    if low.endswith(".csv"):
        df = pd.read_csv(path, nrows=MAX_ROWS_PER_FILE)
    else:
        df = pd.read_parquet(path).head(MAX_ROWS_PER_FILE)
    return df.to_dict("records")


def find_col(keys, wanted):
    lowered = {str(k).lower().replace("_", "").replace(" ", ""): k for k in keys}
    for w in wanted:
        if w in lowered:
            return lowered[w]
    return None


def qa_documents(path):
    rows = load_records(path)
    if not rows or not isinstance(rows[0], dict):
        print("   could not read records")
        return []
    q_col = find_col(rows[0].keys(), QUESTION_COLS)
    a_col = find_col(rows[0].keys(), ANSWER_COLS)
    if not a_col:
        print("   no answer column found. Columns:", list(rows[0].keys()))
        return []
    docs = []
    for r in rows:
        q = str(r.get(q_col, "") or "").strip() if q_col else ""
        a = str(r.get(a_col, "") or "").strip()
        text = f"Question: {q}\nAnswer: {a}" if q else a
        docs.append(text)
    return docs


# ==========================
# MAIN
# ==========================
def main():
    if os.path.isdir(DB_PATH):
        shutil.copytree(DB_PATH, DB_PATH + "_backup", dirs_exist_ok=True)
        print("Backed up old database to", DB_PATH + "_backup")

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    kwargs = {}
    if EMBED_MODEL:
        from chromadb.utils import embedding_functions
        kwargs["embedding_function"] = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
    collection = client.get_or_create_collection(COLLECTION, **kwargs)

    seen = set()
    summary = []

    for root, dirs, files in os.walk(DATASET_FOLDER):
        dirs[:] = [d for d in dirs if not d.startswith("_")]
        for name in sorted(files):
            if name.startswith("_"):
                continue
            path = os.path.join(root, name)
            low = name.lower()
            print(f"\nReading {name}")
            try:
                if low.endswith(".pdf"):
                    docs, kind, check = pdf_chunks(path), "pdf", is_good_prose
                elif low.endswith((".json", ".jsonl", ".csv", ".parquet")):
                    docs, kind, check = qa_documents(path), "qa", is_readable
                else:
                    continue
            except Exception as e:
                print("   error:", e)
                continue

            if docs:
                print("   example:", docs[0][:150].replace("\n", " "))
            ids, texts, metas = [], [], []
            unreadable = duplicates = 0
            for d in docs:
                if not check(d):
                    unreadable += 1
                    continue
                h = fingerprint(d)
                if h in seen:
                    duplicates += 1
                    continue
                seen.add(h)
                ids.append(h)
                texts.append(d)
                metas.append({"source": name, "type": kind})

            for i in range(0, len(ids), BATCH):
                collection.add(
                    ids=ids[i:i + BATCH],
                    documents=texts[i:i + BATCH],
                    metadatas=metas[i:i + BATCH],
                )
                print(f"   added {min(i + BATCH, len(ids))}/{len(ids)}", end="\r")

            print(
                f"   kept {len(texts)} of {len(docs)} "
                f"(unreadable: {unreadable}, duplicates: {duplicates})          "
            )
            summary.append((name, len(texts), len(docs)))

    print("\n=========================")
    print("INGESTION COMPLETE")
    print("=========================")
    for name, kept, total in summary:
        print(f"{name:45s} kept {kept:6d} / {total}")
    print("Total documents:", collection.count())


if __name__ == "__main__":
    main()