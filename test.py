import whisper

print("Loading Whisper...")

model = whisper.load_model("medium")

result = model.transcribe(
    "audio/clean.wav",
    language="kn",
    fp16=False
)

print("\nDetected Text:")
print(result["text"])