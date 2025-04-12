import os
import random
import torchaudio
from datasets import Dataset, Audio

AUDIO_DIR = "archive"  # Root folder containing speaker subfolders

# Step 1: Collect all .wav files from speaker subfolders
all_files = []
for root, dirs, files in os.walk(AUDIO_DIR):
    for file in files:
        if file.endswith(".wav"):
            full_path = os.path.join(root, file)
            all_files.append(full_path)

print(f"Found {len(all_files)} wav files")

# Step 2: Compute durations
random.shuffle(all_files)
durations = []

for file in all_files:
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

# Step 5: Save to disk
dataset.save_to_disk("cptmal_audio_dataset")
