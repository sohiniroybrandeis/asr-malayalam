import torch
import numpy as np
from datasets import load_from_disk, Dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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

# Load the Telugu data
telugu_dataset = load_from_disk("telugu_IS_audio_dataset")

# Compute durations
telugu_dataset = telugu_dataset.map(compute_durations, batched=True)

selected_samples_k = []
total_duration_k = 0.0

for sample in telugu_dataset:
    if total_duration_k + sample["duration"] > (3600 * 3.75): #3.75 hours
        break
    selected_samples_k.append(sample)
    total_duration_k += sample["duration"]
    
print("Total duration: ", total_duration_k)

telugu_dataset = Dataset.from_list(selected_samples_k)

# 2. Load pretrained Wav2Vec2 model
model_name = "facebook/wav2vec2-xls-r-300m"  # or a model trained on Malayalam
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)

model.eval()
model.cuda()

# 3. Function to extract frame-level embeddings (hidden states)
def extract_representations(dataset, max_samples=500, max_frames=150):
    all_reps = []

    for sample in tqdm(dataset.select(range(min(max_samples, len(dataset))))):
        inputs = feature_extractor(sample["audio"]["array"], sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = model(inputs.input_values.cuda())
        reps = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # (T, D)
        reps = reps[::2]  # downsample time dimension
        if len(reps) > max_frames:
            reps = reps[:max_frames]
        all_reps.extend(reps)

    return np.array(all_reps)

# Extract all frame-level embeddings
mal_reps = extract_representations(malayalam_dataset)
tel_reps = extract_representations(telugu_dataset)

# 4. PCA Dimensionality Reduction
pca = PCA(n_components=50)  # Reduce to 50 components
mal_reps_pca = pca.fit_transform(mal_reps)
tel_reps_pca = pca.transform(tel_reps)  # Use same PCA transformation for Telugu

# 5. Clustering with KMeans
kmeans = KMeans(n_clusters=50, random_state=42)  # KMeans without mini-batching
kmeans.fit(np.vstack([mal_reps_pca, tel_reps_pca]))  # Fit on both languages

# Assign cluster labels
mal_labels = kmeans.predict(mal_reps_pca)
tel_labels = kmeans.predict(tel_reps_pca)

# 6. Frequency vectors
mal_freq = np.bincount(mal_labels, minlength=50)
tel_freq = np.bincount(tel_labels, minlength=50)

# Normalize
mal_freq = mal_freq / mal_freq.sum()
tel_freq = tel_freq / tel_freq.sum()

# 7. Cosine similarity (ATDS)
atds = cosine_similarity([mal_freq], [tel_freq])[0][0]
print(f"ATDS (PCA + KMeans-based) between Malayalam and Telugu: {atds:.4f}")
