"""Record a ten-second microphone sample to audio/input.wav."""
from pathlib import Path

import sounddevice as sd
from scipy.io.wavfile import write

SAMPLE_RATE = 16000
DURATION_SECONDS = 10
OUTPUT = Path("audio/input.wav")


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Speak now for {DURATION_SECONDS} seconds...")
    audio = sd.rec(
        int(DURATION_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    write(str(OUTPUT), SAMPLE_RATE, audio)
    print(f"Recording saved to {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
