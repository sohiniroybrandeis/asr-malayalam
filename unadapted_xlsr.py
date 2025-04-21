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
from datasets import load_from_disk

###FINE-TUNING CODE

mal_data_split = load_from_disk("finetune_split")

# Extract the training and test sets
mal_data_train = mal_data_split['train']
mal_data_test = mal_data_split['test']

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def remove_special_characters(batch):
    batch["transcription"] = re.sub(chars_to_ignore_regex, '', batch["transcription"]).lower()
    return batch

mal_data_train = mal_data_train.map(remove_special_characters)
mal_data_test = mal_data_test.map(remove_special_characters)


def extract_all_chars(batch):
  all_text = " ".join(batch["transcription"])
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
tokenizer.save_pretrained(repo_name)

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

mal_data_train = mal_data_train.cast_column("audio", Audio(sampling_rate=16_000))
mal_data_test = mal_data_test.cast_column("audio", Audio(sampling_rate=16_000))


def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        # batch["labels"] = processor(batch["sentence"]).input_ids
        batch["labels"] = processor(batch["transcription"]).input_ids
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

    # Decode predicted sequences
    pred_str = processor.batch_decode(pred_ids)

    # Decode label sequences (removing -100s directly)
    label_ids = pred.label_ids
    label_str = []

    for label in label_ids:
        label = [id for id in label if id != -100]
        decoded = processor.tokenizer.decode(label, group_tokens=True)
        label_str.append(decoded)

    # Compute CER
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


model = Wav2Vec2ForCTC.from_pretrained(repo_name+"/checkpoint-1950").to("cuda")

processor = Wav2Vec2Processor.from_pretrained(repo_name+"/checkpoint-1950")

input_dict = processor(mal_data_test[0]["input_values"], return_tensors="pt", padding=True)

logits = model(input_dict.input_values.to("cuda")).logits

pred_ids = torch.argmax(logits, dim=-1)[0]

mal_data_test_transcription = mal_data_split['test']


print("Prediction:")
print(processor.decode(pred_ids))

print("\nReference:")
print(mal_data_test_transcription[0]["transcription"].lower())

def map_to_result(batch):
  with torch.no_grad():
    input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
    logits = model(input_values).logits

  pred_ids = torch.argmax(logits, dim=-1)
  batch["pred_str"] = processor.batch_decode(pred_ids)[0]
  batch["text"] = processor.decode([id for id in batch["labels"] if id != -100], group_tokens=False)  # Remove -100
  
  return batch

results = mal_data_test.map(map_to_result, remove_columns=mal_data_test.column_names)

print("Test CER: {:.3f}".format(cer_metric.compute(predictions=results["pred_str"], references=results["text"])))
