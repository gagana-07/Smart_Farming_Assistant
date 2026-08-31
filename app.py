import whisper
import ollama
import pyttsx3
import record_audio

print("Loading Whisper...")
model = whisper.load_model("base")

print("Transcribing audio...")
result = model.transcribe("audio/input.wav")

query = result["text"].strip()

print("\nFarmer:", query)

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "system",
            "content": """
You are an agriculture expert.

Rules:
- Give short and practical answers.
- Use simple language.
- Maximum 3 sentences.
- Do not use bullet points.
- Do not use symbols like *, -, or •.
- Focus only on farming advice.
"""
        },
        {
            "role": "user",
            "content": query
        }
    ]
)

answer = response["message"]["content"]

# Clean text for speech
answer = answer.replace("*", "")
answer = answer.replace("•", "")
answer = answer.replace("-", "")

print("\nAssistant:")
print(answer)

# Text-to-Speech
engine = pyttsx3.init()

# Optional: slow down speech slightly
engine.setProperty("rate", 170)

engine.say(answer)
engine.runAndWait()