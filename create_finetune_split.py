from datasets import load_from_disk
from datasets import load_dataset, DatasetDict, ClassLabel, Audio, Dataset

###FINE-TUNING CODE

# Load the Malayalam data
mal_data = load_from_disk("cptmal_audio_trans_dataset")

# Function to compute duration of each audio sample
def compute_durations(batch):
    batch["duration"] = [len(a["array"]) / a["sampling_rate"] for a in batch["audio"]]
    return batch

# Compute durations
mal_data = mal_data.map(compute_durations, batched=True)

selected_samples = []
total_duration = 0.0

for sample in mal_data:
    if total_duration + sample["duration"] > (3600 * 3.75): #3.75 hours
        break
    selected_samples.append(sample)
    total_duration += sample["duration"]
    
print("Total duration: ", total_duration)

mal_data = Dataset.from_list(selected_samples)

# Split the dataset into training and test sets (80% train, 20% test)
mal_data_split = mal_data.train_test_split(test_size=0.2, seed=121) #ensuring same train split each time

mal_data_split.save_to_disk("finetune_split")
