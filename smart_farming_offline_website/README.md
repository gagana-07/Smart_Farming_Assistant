# FarmWise — Smart Farming Assistant Offline Website

This website is a local browser interface for the existing Smart Farming Assistant project. It reuses your local `app.py`, Whisper models, NLLB translator, ChromaDB collection, Ollama model, and Kannada/Hindi TTS. It does not use cloud APIs or external image/font/CDN requests.

## Important: existing local assets are not included
Model weights, your ChromaDB at `data/vector_db`, the Kannada Whisper model, and your existing `audio/` package are not copied into this small website bundle. Keep these folders from your working project. The `app.py` inside the website folder is not used as the backend; `web_app.py` imports your existing `C:\Smart_Farming_Assistant\app.py`. Keep your real backend in place.

## Install / copy
1. Back up your current `C:\Smart_Farming_Assistant` folder or at least `app.py`.
2. Extract the ZIP into `C:\Smart_Farming_Assistant`. Keep the website files together inside `C:\Smart_Farming_Assistant\smart_farming_offline_website`. The launcher is configured to use the parent project’s existing `venv` and root `app.py`.
3. Keep your existing `venv`, `models`, `data`, `audio`, and `translate.py` files/folders. Do not delete them.
4. Make sure Ollama is installed and the model used in `app.py` is available (`qwen2.5:3b`). Start Ollama once if needed.
5. Open the `smart_farming_offline_website` folder and double-click `run_website.bat`, or in PowerShell run:

   ```powershell
   .\venv\Scripts\python.exe -m pip install -r requirements-web.txt
   .\venv\Scripts\python.exe -m uvicorn web_app:app --host 127.0.0.1 --port 8000
   ```

6. Open **http://127.0.0.1:8000** in Chrome or Edge. Keep the terminal window open while using the site. Stop it with `Ctrl+C`.

## Features
- Nature-inspired responsive farming dashboard with locally embedded illustrations.
- Text questions in Kannada, Hindi, or English.
- Voice recording uploaded only to the local server and transcribed with your local faster-whisper models.
- Existing translation, ChromaDB retrieval, Ollama answer generation, and localized TTS reused.
- Play or copy the generated answer.

## Offline and microphone notes
- The interface itself makes no requests to external sites. All inference is local after model files are present.
- Microphone access normally works on `http://127.0.0.1:8000` because browsers treat localhost as a trusted origin. Allow microphone access when prompted.
- Browser recording is usually WebM/Opus. The audio decoder used by faster-whisper must support that format. If audio upload reports a decoding error, install FFmpeg or use the text input while checking your current audio dependencies.
- First launch can take a few minutes because local models initialize. Later requests still take time because inference runs on the same computer.
- One question is processed at a time to avoid running multiple requests through the local models simultaneously.

## Troubleshooting
- **`No module named fastapi`**: run the pip install command above inside your project `venv`.
- **ChromaDB collection missing**: confirm `data/vector_db` exists and contains the `agriculture` collection.
- **Ollama/model error**: open Ollama and check that `qwen2.5:3b` is installed locally.
- **TTS audio not available**: text answers can still appear. Check the console for TTS errors and confirm the existing `audio/tts.py` voice dependencies are installed.
- **Port 8000 in use**: run uvicorn on another port, e.g. `--port 8001`, and open `http://127.0.0.1:8001`.

## Security / access
The server binds to `127.0.0.1`, so it is intended for use on this computer, not other devices on the network. Uploaded audio is stored temporarily under `audio/` and deleted after transcription. Generated response audio is written to `outputs/website_response.wav` and can be overwritten by the next answer.


## Local accounts (Login/Register)

The website includes local account registration and login. User records are stored in `data/website_accounts.sqlite3` under the project root. Read `LOGIN_AND_LANGUAGE_README.txt` for setup, backup, and security limitations. This is intended for a localhost project demo, not public internet deployment.
