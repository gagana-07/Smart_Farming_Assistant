SMART FARMING ASSISTANT — CORRECTED CODE PACKAGE

Included:
- app.py: main assistant
- translate.py: crop-name glossary fixes (including ragi -> ರಾಗಿ / रागी)
- audio/record.py, audio/stt.py, audio/tts.py
- rag/ingest.py and rag/query.py
- smart_farming_offline_website/web_app.py and website assets

Important:
1. This package contains code and website assets only. It does not include your local models,
   ChromaDB database, venv, or recorded audio.
2. Back up your existing project before copying files.
3. Copy the root app.py and translate.py to C:\Smart_Farming_Assistant\.
4. Copy the audio and rag folders to the same project root.
5. Copy the website folder into C:\Smart_Farming_Assistant\smart_farming_offline_website\.
6. Run the website from that folder with:
   ..\venv\Scripts\python.exe -m uvicorn web_app:app --host 127.0.0.1 --port 8000
   (Adjust the venv path if your virtual environment is located elsewhere.)
7. The local models and database must already exist in the expected paths. The package cannot
   make a missing model or database available offline.
8. Python syntax was checked for the included Python files. Full inference was not run here
   because the local models, Ollama runtime and ChromaDB database are on your computer.
9. The file smart_farming_offline_website/app.py from the supplied ZIP was not copied over
   the root app.py: web_app.py imports the project-root app.py. Keep the root app.py as the
   single source of truth.
