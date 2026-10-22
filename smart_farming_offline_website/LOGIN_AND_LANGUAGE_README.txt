SMART FARMING ASSISTANT — LOGIN, REGISTER AND LANGUAGE UI

This package includes a local registration/login system for the offline website.

FEATURES
- Separate Login and Register tabs.
- Registration fields: full name, username, password, confirm password, preferred interface language.
- Supported interface languages: Kannada, Hindi, and English.
- Accounts are stored locally in data/website_accounts.sqlite3.
- Passwords are stored as PBKDF2-HMAC-SHA256 hashes with individual random salts; plaintext passwords are not stored.
- Login uses an HttpOnly session cookie; sessions expire after 14 days. Logout invalidates the session.
- Asking text/audio questions requires a logged-in session.
- No cloud service is used for registration or login.

INSTALL / UPDATE
1. Stop the running server with Ctrl+C.
2. Back up your existing project folder.
3. Extract this ZIP into your project folder, preserving the folders.
4. Ensure the website files are in smart_farming_offline_website/ and the corrected project Python files are in the project root as packaged.
5. Start from the project root with your existing virtual environment activated:
   python -m uvicorn smart_farming_offline_website.web_app:app --host 127.0.0.1 --port 8000
   If your README/run script uses a different working-directory setup, use that existing method.
6. Open http://127.0.0.1:8000 and press Ctrl+F5.
7. Register an account, then the website should sign you in automatically. Next time use Login.

DATA AND LIMITATIONS
- User accounts are saved in data/website_accounts.sqlite3 on this computer. Back up this file if you need to preserve registered users.
- This is intended for a local student project demo on 127.0.0.1, not deployment to the public internet. It does not include email verification, password reset, administrator tools, or production-grade rate limiting/CSRF defenses. Do not reuse a password from another service.
- The AI models and agriculture database must already be installed locally for the farming assistant itself to answer questions offline. First-time model downloads/installation may require internet.
- Interface translations cover the designed static UI. AI answers are generated/translated by the project's local AI/translation modules, so machine translation quality may vary.
