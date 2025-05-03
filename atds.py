import torch
import numpy as np
from datasets import load_from_disk, Dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from collections import Counter

# Load Malayalam and Telugu datasets
mal_dataset = load_from_disk("cptmal_audio_trans_dataset")
tel_dataset = load_from_disk("telugu_IS_audio_dataset")

def compute_durations(batch):
    batch["duration"] = [len(a["array"]) / a["sampling_rate"] for a in batch["audio"]]
    return batch

def select_subset(dataset, hours=3.75):
    dataset = dataset.map(compute_durations, batched=True)
    selected, total_duration = [], 0.0
    for sample in dataset:
        if total_duration + sample["duration"] > hours * 3600:
            break
        selected.append(sample)
        total_duration += sample["duration"]
    print(f"Selected {len(selected)} samples, total duration: {total_duration/3600:.2f} hrs")
    return Dataset.from_list(selected)

# Subset each dataset
mal_dataset = select_subset(mal_dataset)
tel_dataset = select_subset(tel_dataset)

# Load Wav2Vec2
model_name = "facebook/wav2vec2-xls-r-300m"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name).cuda().eval()

# Extract hidden states for a sample of audio
def extract_embeddings(dataset, max_samples=500):
    all_embeddings = []
    for example in tqdm(dataset.select(range(min(max_samples, len(dataset))))):
        input_values = feature_extractor(example["audio"]["array"], sampling_rate=16000, return_tensors="pt").input_values.cuda()
        with torch.no_grad():
            outputs = model(input_values).last_hidden_state.squeeze(0).cpu().numpy()
        all_embeddings.append(outputs)
    return np.concatenate(all_embeddings, axis=0)  # (total_frames, hidden_dim)

# Collect embeddings
print("Extracting embeddings...")
mal_embeddings = extract_embeddings(mal_dataset)
tel_embeddings = extract_embeddings(tel_dataset)

# Combine and cluster to get token ids
all_embeddings = np.concatenate([mal_embeddings, tel_embeddings], axis=0)
print("Running KMeans clustering...")
kmeans = KMeans(n_clusters=100, random_state=42, n_init='auto').fit(all_embeddings)

def get_token_distribution(embeddings, kmeans_model):
    token_ids = kmeans_model.predict(embeddings)
    counter = Counter(token_ids)
    freq_vector = np.zeros(kmeans_model.n_clusters)
    for token_id, count in counter.items():
        freq_vector[token_id] = count
    return freq_vector / freq_vector.sum()

# Token distributions
print("Computing token distributions...")
mal_freq = get_token_distribution(mal_embeddings, kmeans)
tel_freq = get_token_distribution(tel_embeddings, kmeans)

# Cosine similarity
similarity = cosine_similarity([mal_freq], [tel_freq])[0][0]
print(f"ATDS (KMeans-based) between Malayalam and Telugu: {similarity:.4f}")
