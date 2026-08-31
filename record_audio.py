import sounddevice as sd
from scipy.io.wavfile import write

duration = 5
sample_rate = 44100

print("Speak now...")

recording = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1
)

sd.wait()

write("audio/input.wav", sample_rate, recording)

print("Recording saved")