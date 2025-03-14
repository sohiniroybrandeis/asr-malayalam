from datasets import load_dataset, DatasetDict, ClassLabel, Audio
from evaluate import load
import random
import pandas as pd
import IPython.display as ipd
from IPython.display import display, HTML
import re
import json
import numpy as np
import torchaudio
import torch
from transformers import AutoModelForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

mal_data_train = DatasetDict()
mal_data_test = DatasetDict()

# Load the Malayalam subset of Common Voice
mal_data_train = load_dataset("mozilla-foundation/common_voice_13_0", "ml", split="train+validation")
mal_data_test = load_dataset("mozilla-foundation/common_voice_13_0", "ml", split="test")

mal_data_train = mal_data_train.remove_columns(['client_id', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale'])
mal_data_test = mal_data_test.remove_columns(['client_id', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale'])

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    display(df)

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch

mal_data_train = mal_data_train.map(remove_special_characters)
mal_data_test = mal_data_test.map(remove_special_characters)

def extract_all_chars(batch):
  all_text = " ".join(batch["sentence"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = mal_data_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=mal_data_train.column_names)
vocab_test = mal_data_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=mal_data_test.column_names)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(vocab_list)}

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
len(vocab_dict)
# vocab_dict["|"] = vocab_dict.pop(" ")

with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

# print(vocab_dict)

tokenizer = Wav2Vec2CTCTokenizer("vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
# processor.save_pretrained("results")

mal_data_train = mal_data_train.cast_column("audio", Audio(sampling_rate=16_000))
mal_data_test = mal_data_test.cast_column("audio", Audio(sampling_rate=16_000))

def prepare_dataset(batch):
    audio = batch["audio"]
    
    # If the audio is not sampled at 16000 Hz, resample it
    if audio["sampling_rate"] != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=audio["sampling_rate"], new_freq=16000)
        audio_tensor = torch.tensor(audio["array"], dtype=torch.float32)  # Convert to torch tensor (float32)
        audio_tensor = resampler(audio_tensor)  # Apply the resampler to the audio tensor
        audio["sampling_rate"] = 16000  # Update the sampling rate after resampling
    else:
        audio_tensor = torch.tensor(audio["array"], dtype=torch.float32)  # Convert to torch tensor (float32)

    # Process the audio using the Wav2Vec2Processor
    batch["input_values"] = processor(audio_tensor, sampling_rate=16000).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids

    return batch

mal_data_train = mal_data_train.map(prepare_dataset, remove_columns=["audio"])  # Keep input_values
mal_data_test = mal_data_test.map(prepare_dataset, remove_columns=["audio"])

# sample = mal_data_train[0]
# plt.plot(sample["input_values"])
# plt.title("Sample Audio Input")
# plt.show() #added to view audio- looks pretty reasonable


class DataCollatorCTCWithPadding:
    def __init__(
        self,
        processor: Wav2Vec2Processor,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        max_length_labels: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        pad_to_multiple_of_labels: Optional[int] = None,
    ):
        self.processor = processor
        self.padding = padding
        self.max_length = max_length
        self.max_length_labels = max_length_labels
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_to_multiple_of_labels = pad_to_multiple_of_labels

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Separate input values and labels
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features if feature["labels"] is not None]

        # Pad inputs
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # Replace padding tokens with -100 for loss calculation
        labels = labels_batch["input_ids"]
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        batch["labels"] = labels

        return batch

# Initialize collator
data_collator = DataCollatorCTCWithPadding(processor=processor)

wer_metric = load("wer")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Create a copy of label_ids to avoid modifying the original array
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(label_ids, group_tokens=False)

    # Compute Word Error Rate (WER)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-xls-r-300m", 
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.0,
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

model.freeze_feature_extractor()


training_args = TrainingArguments(
    output_dir="./results/",
    group_by_length=False,
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=10,
    fp16=True,
    gradient_checkpointing=False,
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    learning_rate=5e-5,
    weight_decay=0.005,
    warmup_steps=1000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=mal_data_train.shuffle(),
    eval_dataset=mal_data_test,
    tokenizer=processor.feature_extractor,
)


trainer.train()

processor = Wav2Vec2Processor.from_pretrained("results")
model = Wav2Vec2ForCTC.from_pretrained("results/checkpoint-320")

def map_to_result(batch):
  with torch.no_grad():
    input_values = torch.tensor(batch["input_values"], device="cpu").unsqueeze(0)
    logits = model(input_values).logits

  pred_ids = torch.argmax(logits, dim=-1)

  batch["pred_str"] = processor.batch_decode(pred_ids)[0]
  batch["sentence"] = processor.decode(batch["labels"], group_tokens=False)
  
  return batch

results = mal_data_test.map(map_to_result, remove_columns = [col for col in mal_data_test.column_names if col != "sentence"])

print(results.to_pandas())

print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["sentence"])))
