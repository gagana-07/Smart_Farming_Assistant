"""Local offline website wrapper for the existing Smart Farming Assistant.
Run from the project root with: python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations
import os, re, time, threading, sys, sqlite3, secrets, hashlib, hmac
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent  # website folder (templates/static live here)
PROJECT_ROOT = ROOT.parent               # C:\Smart_Farming_Assistant
AUDIO_DIR = PROJECT_ROOT / "audio"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
AUDIO_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# The website lives in a subfolder, but all AI assets and the real app.py live
# in the project root. Make imports and relative paths resolve from that root.
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Importing the project module loads the existing local Whisper models and ChromaDB.
# It does not run the console loop because app.py is imported, not executed directly.
import app as core

api_lock = threading.Lock()
app = FastAPI(title="Smart Farming Assistant", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
templates = Jinja2Templates(directory=str(ROOT / "templates"))

LANGS = {"kn", "hi", "en"}
AUTH_DB = PROJECT_ROOT / "data" / "website_accounts.sqlite3"
AUTH_DB.parent.mkdir(parents=True, exist_ok=True)
SESSION_COOKIE = "sfa_session"


def _db():
    conn = sqlite3.connect(AUTH_DB, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _init_auth_db():
    with _db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            password_salt TEXT NOT NULL,
            preferred_language TEXT NOT NULL DEFAULT 'kn',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            expires_at INTEGER NOT NULL
        )""")
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (int(time.time()),))


_init_auth_db()


def _password_hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310000).hex()


def _create_session(response, user_id: int):
    token = secrets.token_urlsafe(32)
    expires = int(time.time()) + 14 * 24 * 60 * 60
    with _db() as conn:
        conn.execute("INSERT INTO sessions(token_hash, user_id, expires_at) VALUES (?, ?, ?)",
                     (hashlib.sha256(token.encode()).hexdigest(), user_id, expires))
    response.set_cookie(SESSION_COOKIE, token, max_age=14 * 24 * 60 * 60,
                        httponly=True, samesite="strict", secure=False, path="/")


def _current_user(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    with _db() as conn:
        row = conn.execute("""SELECT users.id, users.username, users.full_name, users.preferred_language
            FROM sessions JOIN users ON users.id=sessions.user_id
            WHERE sessions.token_hash=? AND sessions.expires_at>?""", (token_hash, int(time.time()))).fetchone()
    return dict(row) if row else None


def _require_user(request: Request):
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in to continue.")
    return user

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})

def _fix_question(text: str, lang: str) -> str:
    text = (text or "").strip()
    if lang == "kn":
        fixer = getattr(core, "fix_kannada", None)
        return fixer(text) if callable(fixer) else text
    if lang == "hi":
        fixer = getattr(core, "fix_hindi", None)
        return fixer(text) if callable(fixer) else text
    return text

def _make_audio(text: str, lang: str) -> Optional[str]:
    """Create an audio file for browser playback without playing it on the server."""
    out = OUTPUT_DIR / "website_response.wav"
    try:
        if lang in ("kn", "hi"):
            spoken = core.prepare_tts_text(text, lang)
            from audio.tts import speak
            result = speak(spoken, lang, str(out))
            if result and Path(str(result)).exists():
                return "/outputs/" + Path(str(result)).name
            if out.exists():
                return "/outputs/website_response.wav"
        else:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.save_to_file(text, str(out))
            engine.runAndWait()
            engine.stop()
            if out.exists():
                return "/outputs/website_response.wav"
    except Exception as exc:
        print("Website TTS failed:", exc)
    return None

def _answer(question: str, lang: str) -> dict:
    question = _fix_question(question, lang)
    if not question:
        raise HTTPException(status_code=400, detail="I couldn't read a question. Please try again.")
    print(f"\n[Website] Question ({lang}): {question}")
    known = core.known_question_to_english(question, lang)
    if known:
        question_en = known
        print("[Website] Using matched farming-question template:", question_en)
    elif lang == "en":
        question_en = question
    else:
        question_en = core.translate(question, lang, "en")
    print("[Website] English question:", question_en)
    context = core.retrieve(question_en)
    answer_en = ""
    if context:
        answer_en = core.generate_answer(question_en, context)
    if not answer_en or "do not have information" in answer_en.lower():
        if not core.USE_GENERAL_FALLBACK:
            answer_en = core.NO_INFO
        elif not core.mentions_crop(question_en):
            answer_en = core.ASK_CROP
        else:
            answer_en = core.generate_general_answer(question_en)
    chemical_pattern = getattr(core, "CHEMICAL_WORDS", None)
    caution_needed = bool(chemical_pattern.search(answer_en)) if chemical_pattern else False
    for pattern, replacement in core.UNIT_REPLACEMENTS:
        answer_en = re.sub(pattern, replacement, answer_en)
    answer_en = core.numbers_to_words(answer_en)
    if lang == "en":
        answer = answer_en
    else:
        answer = core.translate(answer_en, "en", lang)
    if caution_needed:
        try:
            cached_fn = getattr(core, "cached_translate", None)
            caution = cached_fn(core.CAUTION, "en", lang) if callable(cached_fn) else core.translate(core.CAUTION, "en", lang)
            answer = answer.rstrip() + " " + caution
        except Exception:
            pass
    audio_url = _make_audio(answer, lang)
    return {
        "ok": True,
        "language": lang,
        "question": question,
        "question_en": question_en,
        "answer": answer,
        "answer_en": answer_en,
        "audio_url": audio_url,
        "source": "local models and database",
    }

@app.get("/api/me")
def me(request: Request):
    user = _current_user(request)
    return {"authenticated": bool(user), "user": user}


@app.post("/api/register")
def register(payload: dict, request: Request):
    full_name = str(payload.get("full_name", "")).strip()
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    language = str(payload.get("language", "kn")).lower()
    if len(full_name) < 2 or len(full_name) > 80:
        raise HTTPException(status_code=400, detail="Enter a name between 2 and 80 characters.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,30}", username):
        raise HTTPException(status_code=400, detail="Username must be 3–30 characters using letters, numbers, dot, underscore, or hyphen.")
    if len(password) < 8 or len(password) > 128:
        raise HTTPException(status_code=400, detail="Password must be between 8 and 128 characters.")
    if language not in LANGS:
        raise HTTPException(status_code=400, detail="Choose Kannada, Hindi, or English.")
    salt = secrets.token_bytes(16)
    try:
        with _db() as conn:
            cur = conn.execute("INSERT INTO users(full_name, username, password_hash, password_salt, preferred_language) VALUES (?, ?, ?, ?, ?)",
                (full_name, username, _password_hash(password, salt), salt.hex(), language))
            user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="That username is already registered. Please choose another.")
    response = JSONResponse({"ok": True, "user": {"username": username, "full_name": full_name, "preferred_language": language}})
    _create_session(response, user_id)
    return response


@app.post("/api/login")
def login(payload: dict):
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    with _db() as conn:
        user = conn.execute("SELECT * FROM users WHERE username=? COLLATE NOCASE", (username,)).fetchone()
    valid = False
    if user:
        try:
            salt = bytes.fromhex(user["password_salt"])
            valid = hmac.compare_digest(_password_hash(password, salt), user["password_hash"])
        except (ValueError, TypeError):
            valid = False
    if not valid:
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    response = JSONResponse({"ok": True, "user": {"username": user["username"], "full_name": user["full_name"], "preferred_language": user["preferred_language"]}})
    _create_session(response, user["id"])
    return response


@app.post("/api/logout")
def logout(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        with _db() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),))
    response = JSONResponse({"ok": True})
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@app.post("/api/ask-text")
def ask_text(payload: dict, request: Request):
    _require_user(request)
    lang = str(payload.get("language", "kn")).lower()
    question = str(payload.get("question", "")).strip()
    if lang not in LANGS:
        raise HTTPException(status_code=400, detail="Choose Kannada, Hindi, or English.")
    with api_lock:
        try:
            return _answer(question, lang)
        except HTTPException:
            raise
        except Exception as exc:
            print("Website question error:", repr(exc))
            raise HTTPException(status_code=500, detail=f"The local assistant encountered an error: {exc}")

@app.post("/api/ask-audio")
def ask_audio(request: Request, file: UploadFile = File(...), language: str = Form("auto")):
    _require_user(request)
    lang = language.lower()
    if lang not in ("auto", *LANGS):
        raise HTTPException(status_code=400, detail="Choose auto, Kannada, Hindi, or English.")
    suffix = Path(file.filename or "recording.webm").suffix or ".webm"
    # only allow common audio suffixes; the content is decoded locally by faster-whisper.
    if suffix.lower() not in {".webm", ".wav", ".mp3", ".m4a", ".ogg", ".mp4", ".mpeg"}:
        suffix = ".webm"
    temp_path = AUDIO_DIR / ("website_input" + suffix.lower())
    with api_lock:
        try:
            content = file.file.read()
            if not content:
                raise HTTPException(status_code=400, detail="The recording was empty. Please record again.")
            if len(content) > 25 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Recording is too large. Keep it under 25 MB.")
            temp_path.write_bytes(content)
            # Honor explicit language selection even if SFA_FORCE_LANGUAGE is auto.
            if lang == "auto":
                question, detected = core.transcribe(temp_path)
            else:
                old = core.FORCE_LANGUAGE
                try:
                    core.FORCE_LANGUAGE = lang
                    question, detected = core.transcribe(temp_path)
                finally:
                    core.FORCE_LANGUAGE = old
            if not question:
                raise HTTPException(status_code=400, detail="I couldn't understand that recording. Try speaking a little slower.")
            result = _answer(question, detected)
            result["detected_language"] = detected
            return result
        except HTTPException:
            raise
        except Exception as exc:
            print("Website audio error:", repr(exc))
            raise HTTPException(status_code=500, detail=f"Audio processing failed: {exc}. Check that FFmpeg/audio decoding is available.")
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

@app.get("/api/health")
def health():
    return {"ok": True, "app": "Smart Farming Assistant", "offline_mode": True}
