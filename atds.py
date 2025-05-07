import torch
import numpy as np
from datasets import load_dataset, load_from_disk, Dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from tqdm import tqdm

malayalam_dataset = load_from_disk("cptmal_audio_trans_dataset") #load language 1 data

def compute_durations(batch): #compute duration of audio sample
    batch["duration"] = [len(a["array"]) / a["sampling_rate"] for a in batch["audio"]]
    return batch

malayalam_dataset = malayalam_dataset.map(compute_durations, batched=True)

selected_samples_m = []
total_duration_m = 0.0

for sample in malayalam_dataset:
    if total_duration_m + sample["duration"] > (3600 * 1): #1 hour
        break
    selected_samples_m.append(sample)
    total_duration_m += sample["duration"]
    
print("Total duration: ", total_duration_m)

malayalam_dataset = Dataset.from_list(selected_samples_m)

kannada_dataset = load_from_disk("kannada_IS_audio_dataset") #load language 2 data

kannada_dataset = kannada_dataset.map(compute_durations, batched=True)

selected_samples_k = []
total_duration_k = 0.0

for sample in kannada_dataset:
    if total_duration_k + sample["duration"] > (3600 * 1): #1 hour
        break
    selected_samples_k.append(sample)
    total_duration_k += sample["duration"]
    
print("Total duration: ", total_duration_k)

kannada_dataset = Dataset.from_list(selected_samples_k)

model_name = "facebook/wav2vec2-xls-r-300m"  # load pretrained model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)

model.eval()
model.cuda()


def extract_tokens(batch): #extract tokens using feature extractor
    inputs = feature_extractor(batch["audio"]["array"], sampling_rate=16_000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(inputs.input_values.cuda())

    hidden_states = outputs.last_hidden_state.squeeze(0).cpu().numpy()  # (time_steps, hidden_dim)
    
    token_ids = np.argmax(hidden_states, axis=-1)  # Shape: (time_steps,)
    
    return token_ids

def build_token_frequency(dataset, sample_size=500): #build token frequency distribution
    token_counter = Counter()
    for example in tqdm(dataset.select(range(min(sample_size, len(dataset))))):
        tokens = extract_tokens(example)
        token_counter.update(tokens.tolist())
    
    max_token_id = max(token_counter.keys())
    freq_vector = np.zeros(max_token_id + 1)
    for token_id, count in token_counter.items():
        freq_vector[token_id] = count
    
    freq_vector = freq_vector / freq_vector.sum() #normalize
    
    return freq_vector

malayalam_freq = build_token_frequency(malayalam_dataset)
kan_freq = build_token_frequency(kannada_dataset)

max_len = max(len(malayalam_freq), len(kan_freq))
malayalam_freq = np.pad(malayalam_freq, (0, max_len - len(malayalam_freq)))
kan_freq = np.pad(kan_freq, (0, max_len - len(kan_freq))) #padding

similarity = cosine_similarity([malayalam_freq], [kan_freq])[0][0]

print(f"ATDS (Acoustic Token Distribution Similarity) between Malayalam and Kannada: {similarity:.4f}")
