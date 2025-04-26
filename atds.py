from datasets import load_dataset, load_from_disk, Dataset
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2CTCTokenizer
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Load the Wav2Vec2CTCTokenizer separately (if it exists)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-xls-r-300m")

# Load the pre-trained or fine-tuned model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xls-r-300m")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").eval()

# Function to process audio data from Hugging Face dataset
def extract_audio_features(dataset, processor, model):
    features = []
    
    for example in dataset:
        # Process the audio file (assuming it is in 'audio' field and has 'array' type)
        audio_input = processor(example['audio']['array'], return_tensors="pt", sampling_rate=16000)
        
        # Extract features using the model
        with torch.no_grad():
            outputs = model(**audio_input)
        
        # Get the hidden states (features) of the model
        features.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())  # Mean pooling
        
    return np.vstack(features)

# Step 1: Load the Hugging Face dataset for Malayalam and Tamil
# malayalam_dataset = load_dataset("path_to_your_malayalam_dataset", split="train")
# tamil_dataset = load_dataset("path_to_your_tamil_dataset", split="train")

# Load the Malayalam data
malayalam_dataset = load_from_disk("cptmal_audio_trans_dataset")

# Function to compute duration of each audio sample
def compute_durations(batch):
    batch["duration"] = [len(a["array"]) / a["sampling_rate"] for a in batch["audio"]]
    return batch

# Compute durations
malayalam_dataset = malayalam_dataset.map(compute_durations, batched=True)

selected_samples = []
total_duration = 0.0

for sample in malayalam_dataset:
    if total_duration + sample["duration"] > (3600 * 3.75): #3.75 hours
        break
    selected_samples.append(sample)
    total_duration += sample["duration"]
    
print("Total duration: ", total_duration)

malayalam_dataset = Dataset.from_list(selected_samples)

# Load the Tamil data
tamil_dataset = load_from_disk("tammal_IS_audio_dataset")

# Function to compute duration of each audio sample
def compute_durations(batch):
    batch["duration"] = [len(a["array"]) / a["sampling_rate"] for a in batch["audio"]]
    return batch

# Compute durations
tamil_dataset = tamil_dataset.map(compute_durations, batched=True)

selected_samples = []
total_duration = 0.0

for sample in tamil_dataset:
    if total_duration + sample["duration"] > (3600 * 3.75): #3.75 hours
        break
    selected_samples.append(sample)
    total_duration += sample["duration"]
    
print("Total duration: ", total_duration)

tamil_dataset = Dataset.from_list(selected_samples)

# Step 2: Extract features for both languages
malayalam_feats = extract_audio_features(malayalam_dataset, processor, model)
tamil_feats = extract_audio_features(tamil_dataset, processor, model)

# Step 3: Perform clustering to get token distribution
def cluster_features(features, num_clusters=100):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(features)
    return kmeans

malayalam_kmeans = cluster_features(malayalam_feats)
tamil_kmeans = cluster_features(tamil_feats)

# Step 4: Calculate the token frequency distribution for each language
def calculate_token_distribution(kmeans):
    token_distribution = np.bincount(kmeans.labels_)
    return token_distribution / token_distribution.sum()

malayalam_token_distribution = calculate_token_distribution(malayalam_kmeans)
tamil_token_distribution = calculate_token_distribution(tamil_kmeans)

# Step 5: Calculate cosine similarity (ATDS) between the two distributions
atds_score = cosine_similarity([malayalam_token_distribution], [tamil_token_distribution])
print(f"Acoustic Token Distribution Similarity (ATDS) score: {atds_score[0][0]}")
