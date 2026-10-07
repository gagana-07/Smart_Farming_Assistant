import whisper

print("Loading Whisper model...")

# Load model
model = whisper.load_model("medium")

# Transcribe Kannada audio
result = model.transcribe(
    "audio/recordings/kannada1.wav"
)

print("Detected Language:", result["language"])
print("Text:", result["text"])
# Extract text
text = result["text"].strip()

# Display result
if len(text) < 2:
    print("No speech detected")
else:
    print("\nTranscription:")
    print(text)