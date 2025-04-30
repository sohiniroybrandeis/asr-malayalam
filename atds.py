import torch
import numpy as np
from datasets import load_dataset, load_from_disk, Dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from tqdm import tqdm

# Load the Malayalam data
malayalam_dataset = load_from_disk("cptmal_audio_trans_dataset")

# Function to compute duration of each audio sample
def compute_durations(batch):
    batch["duration"] = [len(a["array"]) / a["sampling_rate"] for a in batch["audio"]]
    return batch

# Compute durations
malayalam_dataset = malayalam_dataset.map(compute_durations, batched=True)

selected_samples_m = []
total_duration_m = 0.0

for sample in malayalam_dataset:
    if total_duration_m + sample["duration"] > (3600 * 3.75): #3.75 hours
        break
    selected_samples_m.append(sample)
    total_duration_m += sample["duration"]
    
print("Total duration: ", total_duration_m)

malayalam_dataset = Dataset.from_list(selected_samples_m)

# Load the Kannada data
kannada_dataset = load_from_disk("kannada_IS_audio_dataset")

# Compute durations
kannada_dataset = kannada_dataset.map(compute_durations, batched=True)

selected_samples_k = []
total_duration_k = 0.0

for sample in kannada_dataset:
    if total_duration_k + sample["duration"] > (3600 * 3.75): #3.75 hours
        break
    selected_samples_k.append(sample)
    total_duration_k += sample["duration"]
    
print("Total duration: ", total_duration_k)

kannada_dataset = Dataset.from_list(selected_samples_k)

# # Load the Tamil data
# tamil_dataset = load_from_disk("tammal_IS_audio_dataset")

# # Compute durations
# tamil_dataset = tamil_dataset.map(compute_durations, batched=True)

# selected_samples_t = []
# total_duration_t = 0.0

# for sample in tamil_dataset:
#     if total_duration_t + sample["duration"] > (3600 * 3.75): #3.75 hours
#         break
#     selected_samples_t.append(sample)
#     total_duration_t += sample["duration"]
    
# print("Total duration: ", total_duration_t)

# tamil_dataset = Dataset.from_list(selected_samples_t)

# 2. Load pretrained Wav2Vec2 model
model_name = "facebook/wav2vec2-xls-r-300m"  # or a model trained on Malayalam
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)

model.eval()
model.cuda()

# 3. Function to extract tokens (we'll use quantized features, pretending each output is a 'token')
def extract_tokens(batch):
    inputs = feature_extractor(batch["audio"]["array"], sampling_rate=16_000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(inputs.input_values.cuda())
    # outputs.last_hidden_state shape: (batch_size, time_steps, hidden_dim)
    # For ATDS, we could:
    # - Cluster these hidden states into discrete tokens (e.g., KMeans)
    # - OR simulate token ids by argmax over dimension (simple version)

    hidden_states = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # (time_steps, hidden_dim)
    
    # Simple 'tokenization': cluster by taking argmax dimension
    token_ids = np.argmax(hidden_states, axis=-1)  # Shape: (time_steps,)
    
    return token_ids

# 4. Build token frequency distribution
def build_token_frequency(dataset, sample_size=500):
    token_counter = Counter()
    for example in tqdm(dataset.select(range(min(sample_size, len(dataset))))):
        tokens = extract_tokens(example)
        token_counter.update(tokens.tolist())
    
    # Convert to a full vector
    max_token_id = max(token_counter.keys())
    freq_vector = np.zeros(max_token_id + 1)
    for token_id, count in token_counter.items():
        freq_vector[token_id] = count
    
    # Normalize the vector
    freq_vector = freq_vector / freq_vector.sum()
    
    return freq_vector

# 5. Compute frequency vectors
malayalam_freq = build_token_frequency(malayalam_dataset)
kann_freq = build_token_frequency(kannada_dataset)
# tamil_freq = build_token_frequency(tamil_dataset)

# Pad the shorter vector (make same length)
max_len = max(len(malayalam_freq), len(kann_freq))
malayalam_freq = np.pad(malayalam_freq, (0, max_len - len(malayalam_freq)))
tamil_freq = np.pad(kann_freq, (0, max_len - len(kann_freq)))
# max_len = max(len(malayalam_freq), len(tamil_freq))
# malayalam_freq = np.pad(malayalam_freq, (0, max_len - len(malayalam_freq)))
# tamil_freq = np.pad(tamil_freq, (0, max_len - len(tamil_freq)))

# 6. Cosine similarity
similarity = cosine_similarity([malayalam_freq], [kann_freq])[0][0]
# similarity = cosine_similarity([malayalam_freq], [tamil_freq])[0][0]

print(f"ATDS (Acoustic Token Distribution Similarity) between Malayalam and Kannada: {similarity:.4f}")
# print(f"ATDS (Acoustic Token Distribution Similarity) between Malayalam and Tamil: {similarity:.4f}")
