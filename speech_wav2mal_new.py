from datasets import load_dataset, DatasetDict, ClassLabel, Audio, Dataset
from evaluate import load
import random
import pandas as pd
import re
import json
import numpy as np
import torch
from transformers import AutoModelForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# Load the Malayalam subset of Common Voice
mal_data_train = load_dataset("mozilla-foundation/common_voice_14_0", "ml", split="train+validation")
mal_data_test = load_dataset("mozilla-foundation/common_voice_14_0", "ml", split="test")

# Function to compute duration of each audio sample
def compute_durations(batch):
    batch["duration"] = [len(a["array"]) / a["sampling_rate"] for a in batch["audio"]]
    return batch

# Compute durations
mal_data_train = mal_data_train.map(compute_durations, batched=True)

selected_samples = []
total_duration = 0.0

for sample in mal_data_train:
    if total_duration + sample["duration"] > 3600:
        break
    selected_samples.append(sample)
    total_duration += sample["duration"]

mal_data_train = Dataset.from_list(selected_samples)

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

# show_random_elements(mal_data_train.remove_columns(["path", "audio", "segment", "variant"]), num_examples=10)

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch

mal_data_train = mal_data_train.map(remove_special_characters)
mal_data_test = mal_data_test.map(remove_special_characters)

# show_random_elements(mal_data_train.remove_columns(["path","audio", "segment", "variant"]))

def extract_all_chars(batch):
  all_text = " ".join(batch["sentence"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

vocab_train = mal_data_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=mal_data_train.column_names)
vocab_test = mal_data_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=mal_data_test.column_names)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
repo_name = "wav2vec2-large-xls-r-300m-malayalam-results"
# print(tokenizer.tokenize("മലയാളം ഒരു മനോഹരമായ ഭാഷയാണ്"))
tokenizer.save_pretrained(repo_name)

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# mal_data_train[0]["path"]

# mal_data_train[0]["audio"]

mal_data_train = mal_data_train.cast_column("audio", Audio(sampling_rate=16_000))
mal_data_test = mal_data_test.cast_column("audio", Audio(sampling_rate=16_000))

# mal_data_train[0]["audio"]

# rand_int = random.randint(0, len(mal_data_train)-1)

# print(mal_data_train[rand_int]["sentence"])
# ipd.Audio(data=mal_data_train[rand_int]["audio"]["array"], autoplay=True, rate=16000)

# rand_int = random.randint(0, len(mal_data_train)-1)

# print("Target text:", mal_data_train[rand_int]["sentence"])
# print("Input array shape:", mal_data_train[rand_int]["audio"]["array"].shape)
# print("Sampling rate:", mal_data_train[rand_int]["audio"]["sampling_rate"])

def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

mal_data_train = mal_data_train.map(prepare_dataset, remove_columns=mal_data_train.column_names)
mal_data_test = mal_data_test.map(prepare_dataset, remove_columns=mal_data_test.column_names)


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
    
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

cer_metric = load("cer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    label_str = [s.replace(processor.tokenizer.pad_token, '') for s in label_str]  # Remove padding

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}

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
  output_dir=repo_name,
  group_by_length=True,
  per_device_train_batch_size=16,
  gradient_accumulation_steps=2,
  evaluation_strategy="steps",
  num_train_epochs=30,
  gradient_checkpointing=True,
  fp16=True,
  save_steps=400,
  eval_steps=400,
  logging_steps=400,
  learning_rate=3e-4,
  warmup_steps=500,
  save_total_limit=2,
  push_to_hub=False,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=mal_data_train,
    eval_dataset=mal_data_test,
    tokenizer=processor 
)

trainer.train()


model = Wav2Vec2ForCTC.from_pretrained(repo_name+"/checkpoint-840").to("cuda")

processor = Wav2Vec2Processor.from_pretrained(repo_name+"/checkpoint-840")

input_dict = processor(mal_data_test[0]["input_values"], return_tensors="pt", padding=True)

logits = model(input_dict.input_values.to("cuda")).logits

pred_ids = torch.argmax(logits, dim=-1)[0]

mal_data_test_transcription = load_dataset("mozilla-foundation/common_voice_14_0", "ml", split="test")

print("Prediction:")
print(processor.decode(pred_ids))

print("\nReference:")
print(mal_data_test_transcription[0]["sentence"].lower())

def map_to_result(batch):
  with torch.no_grad():
    input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
    logits = model(input_values).logits

  pred_ids = torch.argmax(logits, dim=-1)
  batch["pred_str"] = processor.batch_decode(pred_ids)[0]
#   batch["text"] = processor.decode(batch["labels"], group_tokens=False)
  batch["text"] = processor.decode([id for id in batch["labels"] if id != -100], group_tokens=False)  # Remove -100
  
  return batch

results = mal_data_test.map(map_to_result, remove_columns=mal_data_test.column_names)

print(results["pred_str"])
print(results["text"])

print("Test CER: {:.3f}".format(cer_metric.compute(predictions=results["pred_str"], references=results["text"])))
