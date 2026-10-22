const $ = (id) => document.getElementById(id);
const language = $('language');
const question = $('question');
const askButton = $('ask-button');
const micButton = $('mic-button');
const micLabel = $('mic-label');
const micStatus = $('mic-status');
const answerStatus = $('answer-status');
const emptyState = $('empty-state');
const answerContent = $('answer-content');
const loadingState = $('loading-state');
const errorState = $('error-state');
const answerAudio = $('answer-audio');
let recorder = null;
let recordStream = null;
let audioChunks = [];
let recording = false;
let lastRequest = null;
let toastTimer = null;

const placeholders = {
  kn: 'ಉದಾ: ಭತ್ತಕ್ಕೆ ಯಾವ ಗೊಬ್ಬರ ಹಾಕಬೇಕು?',
  hi: 'उदाहरण: गेहूँ की फसल के लिए कौन सी खाद अच्छी है?',
  en: 'e.g. Which fertilizer is suitable for wheat?'
};
language.addEventListener('change', () => { question.placeholder = placeholders[language.value]; });
question.addEventListener('input', () => { $('char-count').textContent = `${question.value.length} / 500`; });

function toast(message) {
  const el = $('toast'); el.textContent = message; el.classList.add('show');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.remove('show'), 2800);
}
function setBusy(busy, title='Thinking through your question…', detail='Searching your local farming knowledge.') {
  askButton.disabled = busy; askButton.style.opacity = busy ? '.65' : '1';
  micButton.disabled = busy;
  emptyState.classList.toggle('hidden', busy || !!lastRequest);
  answerContent.classList.add('hidden'); errorState.classList.add('hidden');
  loadingState.classList.toggle('hidden', !busy);
  answerStatus.classList.toggle('working', busy); answerStatus.classList.toggle('done', false);
  answerStatus.innerHTML = busy ? '<i></i> Working' : '<i></i> Ready';
  $('loading-title').textContent = title; $('loading-detail').textContent = detail;
}
function showError(message) {
  loadingState.classList.add('hidden'); emptyState.classList.add('hidden'); answerContent.classList.add('hidden'); errorState.classList.remove('hidden');
  $('error-text').textContent = message || 'Please try again.';
  answerStatus.classList.remove('working','done'); answerStatus.innerHTML = '<i></i> Needs attention';
  askButton.disabled = false; micButton.disabled = false;
}
function showAnswer(data) {
  lastRequest = {type: 'text', question: data.question, language: data.language};
  loadingState.classList.add('hidden'); emptyState.classList.add('hidden'); errorState.classList.add('hidden'); answerContent.classList.remove('hidden');
  $('recognized-question').textContent = data.question || data.question_en || '';
  $('answer-text').textContent = data.answer || 'No answer was returned.';
  answerStatus.classList.remove('working'); answerStatus.classList.add('done'); answerStatus.innerHTML = '<i></i> Answer ready';
  answerAudio.pause(); answerAudio.removeAttribute('src'); answerAudio.load();
  $('play-answer').disabled = !data.audio_url;
  if (data.audio_url) { answerAudio.src = data.audio_url + '?t=' + Date.now(); answerAudio.load(); }
  askButton.disabled = false; micButton.disabled = false;
}
async function parseResponse(response) {
  let data;
  try { data = await response.json(); } catch { throw new Error('The local server returned an unreadable response.'); }
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status}).`);
  return data;
}
async function askText() {
  const q = question.value.trim();
  if (!q) { toast('Type a farming question first.'); question.focus(); return; }
  if (q.length > 500) { toast('Please keep your question under 500 characters.'); return; }
  lastRequest = {type:'text', question:q, language:language.value};
  setBusy(true, 'Thinking through your question…', 'Using your local models and farming knowledge.');
  try {
    const res = await fetch('/api/ask-text', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question:q, language:language.value})});
    showAnswer(await parseResponse(res));
  } catch (err) { showError(err.message || 'Could not connect to the local assistant.'); }
}
askButton.addEventListener('click', askText);
question.addEventListener('keydown', (event) => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') askText(); });

// Demo prompt chips use the language currently selected.
document.querySelectorAll('.chip').forEach(chip => chip.addEventListener('click', () => {
  const q = chip.getAttribute(`data-q-${language.value}`) || chip.getAttribute('data-q-en');
  question.value = q; $('char-count').textContent = `${q.length} / 500`; question.focus();
}));

micButton.addEventListener('click', async () => {
  if (recording) { recorder.stop(); return; }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
    micStatus.textContent = 'This browser does not support local recording. Please type your question instead.'; return;
  }
  try {
    recordStream = await navigator.mediaDevices.getUserMedia({audio:true});
    audioChunks = [];
    let options = {};
    if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) options.mimeType = 'audio/webm;codecs=opus';
    recorder = new MediaRecorder(recordStream, options);
    recorder.ondataavailable = event => { if (event.data && event.data.size) audioChunks.push(event.data); };
    recorder.onstop = async () => {
      const blob = new Blob(audioChunks, {type: recorder.mimeType || 'audio/webm'});
      recordStream.getTracks().forEach(track => track.stop()); recordStream = null;
      recording = false; micButton.classList.remove('recording'); micLabel.textContent = 'Ask by voice';
      if (!blob.size) { micStatus.textContent = 'No audio captured. Please try again.'; return; }
      await sendAudio(blob);
    };
    recorder.start(); recording = true; micButton.classList.add('recording'); micLabel.textContent = 'Stop recording';
    micStatus.textContent = 'Listening… Speak clearly, then press “Stop recording”.'; toast('Recording started.');
  } catch (err) {
    micStatus.textContent = 'Microphone access was blocked. Allow microphone access for localhost or type your question.';
    toast('Could not access the microphone.');
  }
});
async function sendAudio(blob) {
  const lang = language.value;
  lastRequest = {type:'audio', language:lang};
  micStatus.textContent = 'Processing your recording with the local speech model…';
  setBusy(true, 'Listening to your question…', 'Transcribing audio with your local speech model.');
  document.querySelectorAll('.loading-steps span').forEach((s,i)=>s.classList.toggle('active',i===0));
  const form = new FormData();
  const ext = blob.type.includes('wav') ? 'wav' : blob.type.includes('ogg') ? 'ogg' : 'webm';
  form.append('file', blob, `question.${ext}`); form.append('language', lang);
  try {
    $('loading-title').textContent = 'Understanding your voice question…';
    const res = await fetch('/api/ask-audio', {method:'POST', body:form});
    const data = await parseResponse(res); showAnswer(data);
    micStatus.textContent = `Voice question processed${data.detected_language ? ` (${data.detected_language.toUpperCase()})` : ''}.`;
  } catch (err) { showError(err.message || 'Audio processing failed.'); micStatus.textContent = 'Try recording again or type your question.'; }
}
$('play-answer').addEventListener('click', async () => {
  if (!answerAudio.src) { toast('No generated audio is available for this answer.'); return; }
  try { await answerAudio.play(); } catch { answerAudio.classList.remove('hidden'); toast('Use the audio controls to play the answer.'); }
});
$('copy-answer').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText($('answer-text').textContent); toast('Answer copied.'); }
  catch { const range = document.createRange(); range.selectNodeContents($('answer-text')); const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range); toast('Select and copy the highlighted answer.'); }
});
$('retry-button').addEventListener('click', () => {
  if (!lastRequest) { emptyState.classList.remove('hidden'); errorState.classList.add('hidden'); return; }
  if (lastRequest.type === 'text') { question.value = lastRequest.question; askText(); }
  else { toast('Please record the voice question again.'); errorState.classList.add('hidden'); emptyState.classList.remove('hidden'); }
});

// Confirm the interface is talking to the local server; no remote services are called.
fetch('/api/health').then(r => r.json()).then(data => { if (data.ok) console.info('FarmWise local server ready.'); }).catch(() => { micStatus.textContent = 'Local server connection not detected. Start run_website.bat from the project folder.'; });


// Offline-capable interface localization and account login/registration.
const uiText = {
  en: {
    loginKicker:"SMART FARMING ASSISTANT", loginTitle:"Welcome back",
    loginSubtitle:"Sign in to open your offline farming assistant.", usernameLabel:"Username",
    passwordLabel:"Password", loginNote:"Demo login: enter any username and password. This is a local demo gate, not secure account authentication.",
    loginButton:"Sign in", languageLabel:"Interface language", usernamePlaceholder:"Enter username", passwordPlaceholder:"Enter password", loginTab:"Login",registerTab:"Register",registerTitle:"Create your account",registerSubtitle:"Register once to use your local farming assistant.",fullNameLabel:"Full name",confirmPasswordLabel:"Confirm password",registerButton:"Create account",accountNote:"Accounts are stored locally on this computer. Do not use a password you use elsewhere.",logoutButton:"Log out",
    offline:"Runs locally", tagline:"Your farm. Your language.", eyebrow:"✳ LOCAL KNOWLEDGE · REAL FARMING",
    heroTitle:"Better decisions", heroEm:"with a question.", heroDesc:"Ask about crops, soil, sowing, and plant care in Kannada, Hindi, or English. Your assistant works with locally stored AI models and farming knowledge.",
    point1:"Three languages",point2:"Voice or text",point3:"Offline-ready",artTitle:"Rooted in local knowledge",artSub:"Made for everyday farming",
    helper:"YOUR FARMING HELPER", askHeading:"What would you like to know?", choose:"Choose your language", typeQuestion:"Type your question", questionHint:"e.g. Which fertilizer is suitable for wheat?", oneQuestion:"Ask one question at a time",
    ask:"Get farming advice", voice:"Ask by voice", stop:"Stop recording", voiceStatus:"Voice recording is processed by your local Python models.",
    tryAsking:"TRY ASKING", chip1:"🌾 Fertilizer advice",chip2:"🌱 Sowing time",chip3:"🍅 Plant pests",
    advisory:"YOUR ADVISORY", fieldAdvice:"Advice for your field", ready:"Ready", emptyTitle:"Your answer will grow here",
    emptyDesc:"Ask a question about your crop. Your assistant will find relevant local farming information and respond in your selected language.",
    tag1:"Crop care",tag2:"Soil health",tag3:"Sowing advice",recognized:"QUESTION UNDERSTOOD AS",answerLabel:"FARMING ADVICE",
    listen:"Listen to answer",copy:"Copy answer",source:"Local AI + knowledge base",thinking:"Thinking through your question…",
    searching:"Searching your local farming knowledge.", understand:"Understand",findAdvice:"Find advice",prepareAnswer:"Prepare answer",
    errorTitle:"We couldn't finish that request",retry:"Try again ↗",knowledge:"BUILT AROUND YOUR NEEDS",knowledgeTitle:"Practical help, rooted in the field.",
    cropGuidance:"Crop guidance",cropDesc:"Sowing windows, crop growth, and seasonal practices.",soilNutrition:"Soil & nutrition",soilDesc:"Understand fertilizer and soil-care questions.",
    plantCare:"Plant care",plantDesc:"Get help with common pest and plant-health concerns.",footer:"Designed for farmers. Runs on your local computer.",footerStatus:"Offline-first experience",
    askFirst:"Type a farming question first.",tooLong:"Please keep your question under 500 characters.",working:"Working",answerReady:"Answer ready",
    voiceUnsupported:"This browser does not support local recording. Please type your question instead.",listening:"Listening… Speak clearly, then press “Stop recording”.",
    processing:"Processing your recording with the local speech model…",understanding:"Understanding your voice question…",transcribing:"Transcribing audio with your local speech model.",
    noAudio:"No audio captured. Please try again.",microphoneBlocked:"Microphone access was blocked. Allow microphone access for localhost or type your question.",
    recordingStarted:"Recording started.",noAudioAnswer:"No generated audio is available for this answer.",copied:"Answer copied.",
    noAnswer:"No answer was returned.",tryAgain:"Please try again.",needsAttention:"Needs attention"
  },
  kn: {
    loginKicker:"ಸ್ಮಾರ್ಟ್ ಕೃಷಿ ಸಹಾಯಕ",loginTitle:"ಮತ್ತೆ ಸ್ವಾಗತ",loginSubtitle:"ಆಫ್‌ಲೈನ್ ಕೃಷಿ ಸಹಾಯಕವನ್ನು ತೆರೆಯಲು ಸೈನ್ ಇನ್ ಮಾಡಿ.",
    usernameLabel:"ಬಳಕೆದಾರ ಹೆಸರು",passwordLabel:"ಪಾಸ್‌ವರ್ಡ್",loginNote:"ಡೆಮೊ ಲಾಗಿನ್: ಯಾವುದೇ ಬಳಕೆದಾರ ಹೆಸರು ಮತ್ತು ಪಾಸ್‌ವರ್ಡ್ ನಮೂದಿಸಿ. ಇದು ಪ್ರದರ್ಶನಕ್ಕಾಗಿ ಮಾತ್ರ; ಸುರಕ್ಷಿತ ಖಾತೆ ದೃಢೀಕರಣವಲ್ಲ.",
    loginButton:"ಸೈನ್ ಇನ್",languageLabel:"ಇಂಟರ್ಫೇಸ್ ಭಾಷೆ",usernamePlaceholder:"ಬಳಕೆದಾರ ಹೆಸರನ್ನು ನಮೂದಿಸಿ",passwordPlaceholder:"ಪಾಸ್‌ವರ್ಡ್ ನಮೂದಿಸಿ",loginTab:"ಲಾಗಿನ್",registerTab:"ನೋಂದಣಿ",registerTitle:"ನಿಮ್ಮ ಖಾತೆ ರಚಿಸಿ",registerSubtitle:"ಸ್ಥಳೀಯ ಕೃಷಿ ಸಹಾಯಕವನ್ನು ಬಳಸಲು ನೋಂದಾಯಿಸಿ.",fullNameLabel:"ಪೂರ್ಣ ಹೆಸರು",confirmPasswordLabel:"ಪಾಸ್‌ವರ್ಡ್ ದೃಢೀಕರಿಸಿ",registerButton:"ಖಾತೆ ರಚಿಸಿ",accountNote:"ಖಾತೆಗಳು ಈ ಕಂಪ್ಯೂಟರ್‌ನಲ್ಲಿ ಸ್ಥಳೀಯವಾಗಿ ಸಂಗ್ರಹವಾಗುತ್ತವೆ. ಬೇರೆಡೆ ಬಳಸುವ ಪಾಸ್‌ವರ್ಡ್ ಬಳಸಬೇಡಿ.",logoutButton:"ಲಾಗ್ ಔಟ್",
    offline:"ಸ್ಥಳೀಯವಾಗಿ ಕಾರ್ಯನಿರ್ವಹಿಸುತ್ತದೆ",tagline:"ನಿಮ್ಮ ಹೊಲ. ನಿಮ್ಮ ಭಾಷೆ.",eyebrow:"✳ ಸ್ಥಳೀಯ ಜ್ಞಾನ · ನೈಜ ಕೃಷಿ",
    heroTitle:"ಉತ್ತಮ ನಿರ್ಧಾರಗಳು",heroEm:"ಒಂದು ಪ್ರಶ್ನೆಯಿಂದ ಆರಂಭವಾಗುತ್ತವೆ.",heroDesc:"ಬೆಳೆ, ಮಣ್ಣು, ಬಿತ್ತನೆ ಮತ್ತು ಸಸ್ಯ ಆರೈಕೆಯ ಬಗ್ಗೆ ಕನ್ನಡ, ಹಿಂದಿ ಅಥವಾ ಇಂಗ್ಲಿಷ್‌ನಲ್ಲಿ ಕೇಳಿ. ನಿಮ್ಮ ಸಹಾಯಕ ಸ್ಥಳೀಯ AI ಮಾದರಿಗಳು ಮತ್ತು ಕೃಷಿ ಮಾಹಿತಿಯನ್ನು ಬಳಸುತ್ತದೆ.",
    point1:"ಮೂರು ಭಾಷೆಗಳು",point2:"ಧ್ವನಿ ಅಥವಾ ಪಠ್ಯ",point3:"ಆಫ್‌ಲೈನ್ ಸಿದ್ಧ",artTitle:"ಸ್ಥಳೀಯ ಜ್ಞಾನದಿಂದ ರೂಪಿತ",artSub:"ದೈನಂದಿನ ಕೃಷಿಗಾಗಿ",
    helper:"ನಿಮ್ಮ ಕೃಷಿ ಸಹಾಯಕ",askHeading:"ನೀವು ಏನು ತಿಳಿದುಕೊಳ್ಳಲು ಬಯಸುತ್ತೀರಿ?",choose:"ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ",typeQuestion:"ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಬರೆಯಿರಿ",questionHint:"ಉದಾ: ಭತ್ತಕ್ಕೆ ಯಾವ ಗೊಬ್ಬರ ಸೂಕ್ತ?",oneQuestion:"ಒಂದು ಸಮಯದಲ್ಲಿ ಒಂದು ಪ್ರಶ್ನೆ ಕೇಳಿ",
    ask:"ಕೃಷಿ ಸಲಹೆ ಪಡೆಯಿರಿ",voice:"ಧ್ವನಿಯಲ್ಲಿ ಕೇಳಿ",stop:"ರೆಕಾರ್ಡಿಂಗ್ ನಿಲ್ಲಿಸಿ",voiceStatus:"ಧ್ವನಿಯನ್ನು ನಿಮ್ಮ ಸ್ಥಳೀಯ Python ಮಾದರಿಗಳು ಸಂಸ್ಕರಿಸುತ್ತವೆ.",
    tryAsking:"ಇವುಗಳನ್ನು ಕೇಳಿ",chip1:"🌾 ಗೊಬ್ಬರ ಸಲಹೆ",chip2:"🌱 ಬಿತ್ತನೆ ಸಮಯ",chip3:"🍅 ಸಸ್ಯ ಕೀಟಗಳು",
    advisory:"ನಿಮ್ಮ ಕೃಷಿ ಸಲಹೆ",fieldAdvice:"ನಿಮ್ಮ ಹೊಲಕ್ಕೆ ಸಲಹೆ",ready:"ಸಿದ್ಧ",emptyTitle:"ನಿಮ್ಮ ಉತ್ತರ ಇಲ್ಲಿ ಕಾಣಿಸುತ್ತದೆ",
    emptyDesc:"ನಿಮ್ಮ ಬೆಳೆಯ ಬಗ್ಗೆ ಪ್ರಶ್ನೆ ಕೇಳಿ. ಸಹಾಯಕ ಸ್ಥಳೀಯ ಕೃಷಿ ಮಾಹಿತಿಯನ್ನು ಹುಡುಕಿ ಆಯ್ಕೆಮಾಡಿದ ಭಾಷೆಯಲ್ಲಿ ಉತ್ತರಿಸುತ್ತದೆ.",
    tag1:"ಬೆಳೆ ಆರೈಕೆ",tag2:"ಮಣ್ಣಿನ ಆರೋಗ್ಯ",tag3:"ಬಿತ್ತನೆ ಸಲಹೆ",recognized:"ಅರ್ಥಮಾಡಿಕೊಂಡ ಪ್ರಶ್ನೆ",answerLabel:"ಕೃಷಿ ಸಲಹೆ",
    listen:"ಉತ್ತರವನ್ನು ಕೇಳಿ",copy:"ಉತ್ತರವನ್ನು ನಕಲಿಸಿ",source:"ಸ್ಥಳೀಯ AI + ಜ್ಞಾನ ಸಂಗ್ರಹ",thinking:"ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಪರಿಶೀಲಿಸಲಾಗುತ್ತಿದೆ…",
    searching:"ಸ್ಥಳೀಯ ಕೃಷಿ ಮಾಹಿತಿಯನ್ನು ಹುಡುಕಲಾಗುತ್ತಿದೆ.",understand:"ಅರ್ಥಮಾಡಿಕೊಳ್ಳಿ",findAdvice:"ಸಲಹೆ ಹುಡುಕಿ",prepareAnswer:"ಉತ್ತರ ಸಿದ್ಧಪಡಿಸಿ",
    errorTitle:"ಈ ವಿನಂತಿಯನ್ನು ಪೂರ್ಣಗೊಳಿಸಲಾಗಲಿಲ್ಲ",retry:"ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ ↗",knowledge:"ನಿಮ್ಮ ಅಗತ್ಯಗಳಿಗೆ ಅನುಗುಣವಾಗಿ",knowledgeTitle:"ಹೊಲಕ್ಕೆ ಉಪಯುಕ್ತವಾದ ಪ್ರಾಯೋಗಿಕ ಸಹಾಯ",
    cropGuidance:"ಬೆಳೆ ಮಾರ್ಗದರ್ಶನ",cropDesc:"ಬಿತ್ತನೆ ಅವಧಿ, ಬೆಳೆ ಬೆಳವಣಿಗೆ ಮತ್ತು ಋತುಮಾನ ಪದ್ಧತಿಗಳು.",soilNutrition:"ಮಣ್ಣು ಮತ್ತು ಪೋಷಕಾಂಶ",soilDesc:"ಗೊಬ್ಬರ ಮತ್ತು ಮಣ್ಣಿನ ಆರೈಕೆ ಕುರಿತು ಮಾಹಿತಿ.",
    plantCare:"ಸಸ್ಯ ಆರೈಕೆ",plantDesc:"ಸಾಮಾನ್ಯ ಕೀಟ ಮತ್ತು ಸಸ್ಯ ಆರೋಗ್ಯ ಸಮಸ್ಯೆಗಳಿಗೆ ಸಹಾಯ.",footer:"ರೈತರಿಗಾಗಿ ವಿನ್ಯಾಸಗೊಳಿಸಲಾಗಿದೆ. ನಿಮ್ಮ ಕಂಪ್ಯೂಟರ್‌ನಲ್ಲೇ ಕಾರ್ಯನಿರ್ವಹಿಸುತ್ತದೆ.",footerStatus:"ಆಫ್‌ಲೈನ್‌ಗೆ ಆದ್ಯತೆ",
    askFirst:"ಮೊದಲು ಕೃಷಿ ಪ್ರಶ್ನೆಯನ್ನು ಬರೆಯಿರಿ.",tooLong:"ಪ್ರಶ್ನೆಯನ್ನು 500 ಅಕ್ಷರಗಳೊಳಗೆ ಇರಿಸಿ.",working:"ಕಾರ್ಯನಿರ್ವಹಿಸುತ್ತಿದೆ",answerReady:"ಉತ್ತರ ಸಿದ್ಧ",
    voiceUnsupported:"ಈ ಬ್ರೌಸರ್‌ನಲ್ಲಿ ಧ್ವನಿ ರೆಕಾರ್ಡಿಂಗ್ ಸಾಧ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ಪ್ರಶ್ನೆಯನ್ನು ಬರೆಯಿರಿ.",listening:"ಕೇಳಲಾಗುತ್ತಿದೆ… ಸ್ಪಷ್ಟವಾಗಿ ಮಾತನಾಡಿ, ನಂತರ ರೆಕಾರ್ಡಿಂಗ್ ನಿಲ್ಲಿಸಿ.",
    processing:"ಸ್ಥಳೀಯ ಧ್ವನಿ ಮಾದರಿಯಿಂದ ರೆಕಾರ್ಡಿಂಗ್ ಸಂಸ್ಕರಿಸಲಾಗುತ್ತಿದೆ…",understanding:"ಧ್ವನಿ ಪ್ರಶ್ನೆಯನ್ನು ಅರ್ಥಮಾಡಿಕೊಳ್ಳಲಾಗುತ್ತಿದೆ…",transcribing:"ಸ್ಥಳೀಯ ಮಾದರಿಯಿಂದ ಧ್ವನಿಯನ್ನು ಪಠ್ಯಗೊಳಿಸಲಾಗುತ್ತಿದೆ.",
    noAudio:"ಧ್ವನಿ ದಾಖಲಾಗಿಲ್ಲ. ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",microphoneBlocked:"ಮೈಕ್ರೋಫೋನ್ ಅನುಮತಿ ಇಲ್ಲ. localhost ಗೆ ಅನುಮತಿ ನೀಡಿ ಅಥವಾ ಪ್ರಶ್ನೆಯನ್ನು ಬರೆಯಿರಿ.",
    recordingStarted:"ರೆಕಾರ್ಡಿಂಗ್ ಪ್ರಾರಂಭವಾಗಿದೆ.",noAudioAnswer:"ಈ ಉತ್ತರಕ್ಕೆ ಧ್ವನಿ ಲಭ್ಯವಿಲ್ಲ.",copied:"ಉತ್ತರ ನಕಲಿಸಲಾಗಿದೆ.",
    noAnswer:"ಉತ್ತರ ಲಭ್ಯವಿಲ್ಲ.",tryAgain:"ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",needsAttention:"ಗಮನ ಅಗತ್ಯ"
  },
  hi: {
    loginKicker:"स्मार्ट कृषि सहायक",loginTitle:"वापसी पर स्वागत है",loginSubtitle:"ऑफ़लाइन कृषि सहायक खोलने के लिए साइन इन करें।",
    usernameLabel:"उपयोगकर्ता नाम",passwordLabel:"पासवर्ड",loginNote:"डेमो लॉगिन: कोई भी उपयोगकर्ता नाम और पासवर्ड दर्ज करें। यह केवल स्थानीय डेमो है, सुरक्षित खाता प्रमाणीकरण नहीं।",
    loginButton:"साइन इन",languageLabel:"इंटरफ़ेस भाषा",usernamePlaceholder:"उपयोगकर्ता नाम दर्ज करें",passwordPlaceholder:"पासवर्ड दर्ज करें",loginTab:"लॉगिन",registerTab:"रजिस्टर",registerTitle:"अपना खाता बनाएँ",registerSubtitle:"स्थानीय कृषि सहायक का उपयोग करने के लिए पंजीकरण करें।",fullNameLabel:"पूरा नाम",confirmPasswordLabel:"पासवर्ड की पुष्टि करें",registerButton:"खाता बनाएँ",accountNote:"खाते इसी कंप्यूटर पर स्थानीय रूप से संग्रहीत होते हैं। कहीं और इस्तेमाल किया गया पासवर्ड न रखें।",logoutButton:"लॉग आउट",
    offline:"स्थानीय रूप से चलता है",tagline:"आपका खेत। आपकी भाषा।",eyebrow:"✳ स्थानीय ज्ञान · वास्तविक खेती",
    heroTitle:"बेहतर निर्णय",heroEm:"एक सवाल से शुरू होते हैं।",heroDesc:"फसल, मिट्टी, बुवाई और पौधों की देखभाल के बारे में हिंदी, कन्नड़ या अंग्रेज़ी में पूछें। सहायक स्थानीय AI मॉडल और खेती की जानकारी का उपयोग करता है।",
    point1:"तीन भाषाएँ",point2:"आवाज़ या टेक्स्ट",point3:"ऑफ़लाइन तैयार",artTitle:"स्थानीय ज्ञान पर आधारित",artSub:"रोज़मर्रा की खेती के लिए",
    helper:"आपका कृषि सहायक",askHeading:"आप क्या जानना चाहते हैं?",choose:"अपनी भाषा चुनें",typeQuestion:"अपना सवाल लिखें",questionHint:"उदाहरण: धान के लिए कौन सी खाद उपयुक्त है?",oneQuestion:"एक बार में एक सवाल पूछें",
    ask:"खेती की सलाह लें",voice:"आवाज़ से पूछें",stop:"रिकॉर्डिंग रोकें",voiceStatus:"आवाज़ को आपके स्थानीय Python मॉडल संसाधित करते हैं।",
    tryAsking:"ये सवाल पूछें",chip1:"🌾 खाद की सलाह",chip2:"🌱 बुवाई का समय",chip3:"🍅 पौधों के कीट",
    advisory:"आपकी सलाह",fieldAdvice:"आपके खेत के लिए सलाह",ready:"तैयार",emptyTitle:"आपका जवाब यहाँ दिखाई देगा",
    emptyDesc:"अपनी फसल के बारे में पूछें। सहायक स्थानीय कृषि जानकारी खोजकर चुनी गई भाषा में जवाब देगा।",
    tag1:"फसल की देखभाल",tag2:"मिट्टी का स्वास्थ्य",tag3:"बुवाई की सलाह",recognized:"समझा गया सवाल",answerLabel:"खेती की सलाह",
    listen:"जवाब सुनें",copy:"जवाब कॉपी करें",source:"स्थानीय AI + ज्ञान आधार",thinking:"आपके सवाल पर विचार हो रहा है…",
    searching:"स्थानीय कृषि जानकारी खोजी जा रही है।",understand:"समझें",findAdvice:"सलाह खोजें",prepareAnswer:"जवाब तैयार करें",
    errorTitle:"यह अनुरोध पूरा नहीं हो सका",retry:"फिर कोशिश करें ↗",knowledge:"आपकी ज़रूरतों के अनुसार",knowledgeTitle:"खेत के लिए व्यावहारिक मदद",
    cropGuidance:"फसल मार्गदर्शन",cropDesc:"बुवाई का समय, फसल वृद्धि और मौसमी तरीके।",soilNutrition:"मिट्टी और पोषण",soilDesc:"खाद और मिट्टी की देखभाल से जुड़े सवालों के जवाब।",
    plantCare:"पौधों की देखभाल",plantDesc:"कीट और पौधों के स्वास्थ्य की सामान्य समस्याओं में मदद।",footer:"किसानों के लिए बनाया गया। आपके कंप्यूटर पर चलता है।",footerStatus:"ऑफ़लाइन-प्रथम अनुभव",
    askFirst:"पहले खेती का सवाल लिखें।",tooLong:"सवाल 500 अक्षरों से कम रखें।",working:"काम जारी है",answerReady:"जवाब तैयार",
    voiceUnsupported:"यह ब्राउज़र स्थानीय रिकॉर्डिंग का समर्थन नहीं करता। कृपया सवाल लिखें।",listening:"सुन रहा है… स्पष्ट बोलें, फिर रिकॉर्डिंग रोकें।",
    processing:"स्थानीय स्पीच मॉडल से रिकॉर्डिंग संसाधित हो रही है…",understanding:"आपकी आवाज़ वाला सवाल समझा जा रहा है…",transcribing:"स्थानीय मॉडल आवाज़ को टेक्स्ट में बदल रहा है।",
    noAudio:"ऑडियो रिकॉर्ड नहीं हुआ। फिर कोशिश करें।",microphoneBlocked:"माइक्रोफ़ोन अनुमति नहीं मिली। localhost के लिए अनुमति दें या सवाल लिखें।",
    recordingStarted:"रिकॉर्डिंग शुरू हुई।",noAudioAnswer:"इस जवाब के लिए ऑडियो उपलब्ध नहीं है।",copied:"जवाब कॉपी किया गया।",
    noAnswer:"कोई जवाब नहीं मिला।",tryAgain:"फिर कोशिश करें।",needsAttention:"ध्यान दें"
  }
};
function applyInterfaceLanguage(code) {
  const d = uiText[code] || uiText.en;
  document.documentElement.lang = code;
  const set = (selector, value) => { const el = document.querySelector(selector); if (el && value) el.textContent = value; };
  document.querySelectorAll('[data-i18n]').forEach(el => { const k = el.dataset.i18n; if (d[k]) el.textContent = d[k]; });
  set('.offline-pill', d.offline); set('.topbar-note', d.tagline); set('.eyebrow', d.eyebrow);
  set('.hero h1', d.heroTitle);
  const em = document.querySelector('.hero h1 em'); if (em) em.textContent = d.heroEm;
  set('.hero-copy>p', d.heroDesc);
  const points = document.querySelectorAll('.hero-points span'); [d.point1,d.point2,d.point3].forEach((v,i)=>{if(points[i]) points[i].innerHTML='<b>✓</b> '+v;});
  set('.art-label strong',d.artTitle); set('.art-label small',d.artSub);
  set('.ask-panel .section-kicker',d.helper); set('.ask-panel h2',d.askHeading);
  const labels = document.querySelectorAll('.ask-panel .field-label'); if(labels[0]) labels[0].textContent=d.choose; if(labels[1]) labels[1].textContent=d.typeQuestion;
  question.placeholder = d.questionHint;
  const qf=document.querySelector('.question-footer span:last-child'); if(qf) qf.textContent=d.oneQuestion;
  const askSpan=document.querySelector('#ask-button span'); if(askSpan) askSpan.textContent=d.ask;
  micLabel.textContent=d.voice;
  set('.mic-status',d.voiceStatus); set('.suggestion-title',d.tryAsking);
  document.querySelectorAll('.chip').forEach((el,i)=>el.textContent=[d.chip1,d.chip2,d.chip3][i]||el.textContent);
  const answerKickers=document.querySelectorAll('#answer-panel .section-kicker'); if(answerKickers[0]) answerKickers[0].textContent=d.advisory;
  set('#answer-panel .answer-top h2',d.fieldAdvice);
  if (!answerStatus.classList.contains('working') && !answerStatus.classList.contains('done')) answerStatus.innerHTML='<i></i> '+d.ready;
  set('#empty-state h3',d.emptyTitle); set('#empty-state>p',d.emptyDesc);
  document.querySelectorAll('.empty-tags span').forEach((el,i)=>el.textContent=[d.tag1,d.tag2,d.tag3][i]||el.textContent);
  set('.recognized-question small',d.recognized); set('.answer-label',d.answerLabel);
  const play=$('play-answer'); if(play) play.innerHTML='<span>▶</span> '+d.listen;
  const copy=$('copy-answer'); if(copy) copy.innerHTML='<span>▣</span> '+d.copy;
  set('.local-source',d.source); set('#loading-title',d.thinking); set('#loading-detail',d.searching);
  document.querySelectorAll('.loading-steps span').forEach((el,i)=>el.textContent=[d.understand,d.findAdvice,d.prepareAnswer][i]);
  set('#error-state h3',d.errorTitle); set('#retry-button',d.retry);
  set('.strip-heading .section-kicker',d.knowledge); set('.strip-heading h2',d.knowledgeTitle);
  const cards=document.querySelectorAll('.knowledge-card');
  [[d.cropGuidance,d.cropDesc],[d.soilNutrition,d.soilDesc],[d.plantCare,d.plantDesc]].forEach((pair,i)=>{if(cards[i]){const h=cards[i].querySelector('h3'),p=cards[i].querySelector('p');if(h)h.textContent=pair[0];if(p)p.textContent=pair[1];}});
  set('footer p',d.footer); set('.footer-status',d.footerStatus);
  const loginLang=$('login-language'); if(loginLang && loginLang.value!==code) loginLang.value=code;
}
const loginForm = $('login-form');
const registerForm = $('register-form');
function showAuthForm(which) {
  const registering = which === 'register';
  loginForm.classList.toggle('hidden', registering);
  registerForm.classList.toggle('hidden', !registering);
  $('show-login').classList.toggle('active', !registering);
  $('show-register').classList.toggle('active', registering);
  $('login-error').textContent = '';
}
$('show-login').addEventListener('click', () => showAuthForm('login'));
$('show-register').addEventListener('click', () => showAuthForm('register'));
function authError(message) { $('login-error').textContent = message || (uiText[language.value]?.tryAgain || uiText.en.tryAgain); }
function enterApp(user) {
  $('login-screen').classList.add('hidden');
  document.body.classList.add('logged-in');
  const shell = $('app-shell'); if (shell) shell.style.visibility = 'visible';
  const chosen = user?.preferred_language || language.value || 'kn';
  if (['kn','hi','en'].includes(chosen)) language.value = chosen;
  applyInterfaceLanguage(language.value);
}
async function sendAuth(endpoint, payload) {
  const response = await fetch(endpoint, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || (uiText[language.value]?.tryAgain || uiText.en.tryAgain));
  return data;
}
loginForm.addEventListener('submit', async event => {
  event.preventDefault();
  $('login-error').textContent = '';
  try {
    const data = await sendAuth('/api/login', {username:$('login-username').value.trim(), password:$('login-password').value});
    enterApp(data.user);
  } catch (err) { authError(err.message); }
});
registerForm.addEventListener('submit', async event => {
  event.preventDefault();
  $('login-error').textContent = '';
  const password = $('register-password').value;
  if (password !== $('register-confirm').value) { authError(({en:'Passwords do not match.',kn:'ಪಾಸ್‌ವರ್ಡ್‌ಗಳು ಹೊಂದಿಕೆಯಾಗುತ್ತಿಲ್ಲ.',hi:'दोनों पासवर्ड मेल नहीं खाते।'})[language.value]); return; }
  try {
    const data = await sendAuth('/api/register', {full_name:$('register-name').value.trim(), username:$('register-username').value.trim(), password, language:$('register-language').value});
    enterApp(data.user);
  } catch (err) { authError(err.message); }
});
const loginLanguage = $('login-language');
if (loginLanguage) loginLanguage.addEventListener('change', () => {
  language.value = loginLanguage.value;
  $('register-language').value = loginLanguage.value;
  applyInterfaceLanguage(loginLanguage.value);
});
$('register-language').addEventListener('change', () => {
  language.value = $('register-language').value;
  if (loginLanguage) loginLanguage.value = language.value;
  applyInterfaceLanguage(language.value);
});
$('logout-button').addEventListener('click', async () => {
  try { await fetch('/api/logout', {method:'POST'}); } finally {
    document.body.classList.remove('logged-in');
    $('login-screen').classList.remove('hidden');
    $('login-password').value = '';
    showAuthForm('login');
  }
});
(async function restoreSession() {
  try {
    const response = await fetch('/api/me');
    const data = await response.json();
    if (data.authenticated && data.user) enterApp(data.user);
  } catch (_) { /* server may still be starting; login remains available */ }
})();
language.addEventListener('change', () => { if (loginLanguage) loginLanguage.value=language.value; $('register-language').value=language.value; applyInterfaceLanguage(language.value); });
const shellForLogin=document.querySelector('.app-shell'); if(shellForLogin) shellForLogin.id='app-shell';
applyInterfaceLanguage(language.value);

