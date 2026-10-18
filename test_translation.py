"""
Run farming sentences through the translator and read the output for mistakes.
Run from the project root:   python test_translation.py
Add a line to the glossary in translate.py for every wrong word you find.
"""
from translate import translate

EN_SENTENCES = [
    "Apply urea to the ragi crop.",
    "Sow maize at the start of the rainy season.",
    "Spray the pesticide on the tomato plants.",
    "Irrigate the sugarcane field every week.",
    "Apply manure to the areca nut trees.",
    "Water the groundnut crop after two weeks.",
    "Remove weeds from the jowar field.",
    "Treat the cotton seeds before sowing.",
    "Yellow leaves on the coconut tree show nutrient deficiency.",
    "Apply fertilizer to the onion and chilli crops.",
]

KN_QUESTIONS = [
    "ರಾಗಿಗೆ ಯಾವ ಗೊಬ್ಬರ ಹಾಕಬೇಕು?",
    "ಮೆಕ್ಕೆಜೋಳ ಯಾವಾಗ ಬಿತ್ತಬೇಕು?",
    "ಟೊಮೆಟೊ ಗಿಡಕ್ಕೆ ಕೀಟ ಬಂದರೆ ಏನು ಮಾಡಬೇಕು?",
    "ಜೋಳಕ್ಕೆ ಎಷ್ಟು ನೀರು ಬೇಕು?",
    "ಕಬ್ಬಿಗೆ ಯಾವ ರೋಗ ಬರುತ್ತದೆ?",
    "ಅಡಿಕೆ ಮರಕ್ಕೆ ಯಾವ ಗೊಬ್ಬರ ಹಾಕಬೇಕು?",
    "ತೆಂಗಿನ ಮರದ ಎಲೆ ಹಳದಿಯಾದರೆ ಏನು ಮಾಡಬೇಕು?",
    "ಈರುಳ್ಳಿ ಬೆಳೆಗೆ ಎಷ್ಟು ಬಾರಿ ನೀರು ಕೊಡಬೇಕು?",
    "ಮೆಣಸಿನಕಾಯಿ ಗಿಡಕ್ಕೆ ರೋಗ ಬಂದರೆ ಏನು ಮಾಡಬೇಕು?",
    "ಹತ್ತಿ ಯಾವಾಗ ಬಿತ್ತಬೇಕು?",
]

HI_QUESTIONS = [
    "रागी में कौन सी खाद डालें?",
    "मक्का की बुवाई कब करें?",
    "टमाटर के पौधे में कीड़े लग जाएं तो क्या करें?",
    "ज्वार को कितना पानी चाहिए?",
    "गन्ने में कौन सी बीमारी लगती है?",
    "प्याज की फसल में कितनी बार सिंचाई करें?",
    "कपास की बुवाई कब करें?",
    "मूंगफली में कौन सी खाद डालें?",
]

print("\n===== ENGLISH -> KANNADA (answers) =====")
for s in EN_SENTENCES:
    print(f"{s}\n   {translate(s, 'en', 'kn')}")

print("\n===== ENGLISH -> HINDI (answers) =====")
for s in EN_SENTENCES:
    print(f"{s}\n   {translate(s, 'en', 'hi')}")

print("\n===== KANNADA -> ENGLISH (questions) =====")
for s in KN_QUESTIONS:
    print(f"{s}\n   {translate(s, 'kn', 'en')}")

print("\n===== HINDI -> ENGLISH (questions) =====")
for s in HI_QUESTIONS:
    print(f"{s}\n   {translate(s, 'hi', 'en')}")