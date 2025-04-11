import os
import random
from pydub import AudioSegment
import torchaudio
from datasets import Dataset, Audio

AUDIO_DIR = "kb_data_clean_m4a/malayalam/valid/audio"          # Folder with .m4a files
WAV_OUTPUT_DIR = "kb_data_clean_m4a/malayalam/valid/wav_audio"        # Folder to store .wav files
os.makedirs(WAV_OUTPUT_DIR, exist_ok=True)

# Step 1: Convert .m4a to .wav
converted_files = []
for filename in os.listdir(AUDIO_DIR):
    if filename.endswith(".m4a"):
        m4a_path = os.path.join(AUDIO_DIR, filename)
        wav_filename = os.path.splitext(filename)[0] + ".wav"
        wav_path = os.path.join(WAV_OUTPUT_DIR, wav_filename)

        if not os.path.exists(wav_path):  # Avoid reconverting
            audio = AudioSegment.from_file(m4a_path, format="m4a")
            audio.export(wav_path, format="wav")

        converted_files.append(wav_path)

# Step 2: Compute durations
random.shuffle(converted_files)
durations = []

for file in converted_files:
    info = torchaudio.info(file)
    duration = info.num_frames / info.sample_rate
    durations.append((file, duration))

# Step 3: Select up to 15 hours of data
max_seconds = 15 * 60 * 60
selected_files = []
total = 0

for file, dur in durations:
    if total + dur <= max_seconds:
        selected_files.append((file, dur))
        total += dur
    else:
        break

print(f"Selected {len(selected_files)} files, total duration: {total / 3600:.2f} hours")

# Step 4: Create HF dataset
data = {
    "audio": [f for f, _ in selected_files]
}

dataset = Dataset.from_dict(data)
dataset = dataset.cast_column("audio", Audio())

dataset.save_to_disk("cptmal_audio_dataset")
