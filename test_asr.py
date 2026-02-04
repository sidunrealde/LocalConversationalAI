"""Quick ASR test - records 3 seconds and transcribes. SPEAK NOW after running!"""
import sounddevice as sd
import numpy as np
import time
import sys

# Add modules to path
sys.path.insert(0, '.')

print("=" * 50)
print("ASR Debug Test (using app's ASREngine)")
print("=" * 50)

# Use the app's ASR engine with updated settings
from modules import ASREngine

print("\n🔄 Initializing ASR Engine...")
asr = ASREngine(model_size="medium")
asr.load_model()
print("✅ Model loaded!")

# Record
print("\n🎙️ Recording 3 seconds... SPEAK NOW!")
sample_rate = 16000
duration = 3

recording = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype="float32",
    device=1,  # Razer Barracuda
)
sd.wait()

audio_data = recording.squeeze()
energy = np.sqrt(np.mean(audio_data ** 2))
peak = np.max(np.abs(audio_data))

print(f"\n📊 Audio Stats:")
print(f"  - Energy (RMS): {energy:.4f}")
print(f"  - Peak: {peak:.4f}")

# Transcribe
print("\n📝 Transcribing...")
start = time.time()
result = asr.transcribe(audio_data, sample_rate=sample_rate, language="en")
elapsed = time.time() - start

print(f"\n✅ Done in {elapsed:.2f}s")
print(f"📝 Result: '{result}'")
print("=" * 50)
