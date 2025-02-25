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
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

mal_data_train = DatasetDict()
mal_data_test = DatasetDict()

# Load the Malayalam subset of Common Voice
mal_data_train = load_dataset("mozilla-foundation/common_voice_13_0", "ml", split="train+validation")
mal_data_test = load_dataset("mozilla-foundation/common_voice_13_0", "ml", split="test")

# Inspect the first example
# print(mal_data[0])
# print(mal_data["train"][0])

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

# show_random_elements(mal_data_train.remove_columns(["path", "audio"]))

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    return batch

# mal_data = mal_data.map(remove_special_characters)
mal_data_train = mal_data_train.map(remove_special_characters)
mal_data_test = mal_data_test.map(remove_special_characters)

# show_random_elements(mal_data_train.remove_columns(["path", "audio"]))
# # show_random_elements(mal_data["train"])

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
# tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
# # print("vocab len", len(tokenizer.get_vocab()))

# # print("Tokenizer special tokens map:", tokenizer.special_tokens_map)
# # print("All special tokens:", tokenizer.all_special_tokens)

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# # rand_int = random.randint(0, len(mal_data["train"]))

# # print("Target text:", mal_data["train"][rand_int]["sentence"])
# # print("Input array shape:", np.asarray(mal_data["train"][rand_int]["audio"]["array"]).shape)
# # print("Sampling rate:", mal_data["train"][rand_int]["audio"]["sampling_rate"])

# print(mal_data_train[0]["audio"])

mal_data_train = mal_data_train.cast_column("audio", Audio(sampling_rate=16_000))
mal_data_test = mal_data_test.cast_column("audio", Audio(sampling_rate=16_000))

# rand_int = random.randint(0, len(mal_data_train)-1)

# print(mal_data_train[rand_int]["sentence"])
# audio_array = mal_data_train[rand_int]["audio"]["array"]

# # Normalize and convert to int16
# audio_int16 = np.int16(audio_array * 32767)

# # Save as WAV file
# write('test.wav', 16000, audio_int16)
# print(ipd.Audio(data=mal_data_train[rand_int]["audio"]["array"], autoplay=True, rate=16000))

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

mal_data_train = mal_data_train.map(prepare_dataset, remove_columns=mal_data_train.column_names)
mal_data_test = mal_data_test.map(prepare_dataset, remove_columns=mal_data_test.column_names)
# # Apply the dataset transformation

# # sample = mal_data["train"][0]
# # plt.plot(sample["input_values"])
# # plt.title("Sample Audio Input")
# # plt.show()

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
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
    
data_collator = DataCollatorCTCWithPadding(processor=processor)

wer_metric = load("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id


    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer.get_vocab()),  # Fix: Use .get_vocab()
)

# model = Wav2Vec2ForCTC.from_pretrained(
#     "facebook/wav2vec2-base", 
#     ctc_loss_reduction="mean", 
#     pad_token_id=processor.tokenizer.pad_token_id,
#     vocab_size=len(tokenizer.get_vocab())
# )
# # print("Vocab", vocab_dict)
# # print("Pad token ID:", processor.tokenizer.pad_token_id)
# # print("Vocabulary size:", len(tokenizer.get_vocab()))

model.freeze_feature_extractor()


training_args = TrainingArguments(
  output_dir="./results/",
  group_by_length=False,
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=10,
  fp16=True,
  gradient_checkpointing=True, 
  save_steps=500,
  eval_steps=500,
  logging_steps=500,
  learning_rate=1e-4,
  weight_decay=0.005,
  warmup_steps=1000,
  save_total_limit=2,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=mal_data_train,
    eval_dataset=mal_data_test,
    tokenizer=processor.feature_extractor,
)


trainer.train()


# processor = Wav2Vec2Processor.from_pretrained("results")
# model = Wav2Vec2ForCTC.from_pretrained("results")

# def map_to_result(batch):
#   with torch.no_grad():
#     input_values = torch.tensor(batch["input_values"], device="cpu").unsqueeze(0)
#     logits = model(input_values).logits

#   pred_ids = torch.argmax(logits, dim=-1)
#   batch["pred_str"] = processor.batch_decode(pred_ids)[0]
#   batch["sentence"] = processor.decode(batch["labels"], group_tokens=False)
  
#   return batch

# results = mal_data_test.map(map_to_result, remove_columns=mal_data_test.column_names)

# print(results.to_pandas())

# print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["sentence"])))
