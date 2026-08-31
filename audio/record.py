import sounddevice as sd
from scipy.io.wavfile import write

fs = 16000
duration = 5

print("Speak now...")

audio = sd.rec(
    int(duration * fs),
    samplerate=fs,
    channels=1
)

sd.wait()

write("audio/input.wav", fs, audio)

print("Recording saved successfully!")