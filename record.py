import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100
seconds = 5

print("Recording... Speak now")

recording = sd.rec(
    int(seconds * fs),
    samplerate=fs,
    channels=1
)

sd.wait()

write("audio/input.wav", fs, recording)

print("Recording saved as audio/input.wav")