import whisper

print("Loading Whisper model...")

model = whisper.load_model("base")

result = model.transcribe("audio/input.wav")

text = result["text"].strip()

print(result)

if len(text) < 2:
    print("No speech detected")
else:
    print("\nTranscription:")
    print(text)