from datasets import load_dataset, DatasetDict, ClassLabel, Audio, Dataset
from evaluate import load
import random
import pandas as pd
import re
import gc
import json
import numpy as np
import torch
from transformers import Wav2Vec2ForPreTraining, Wav2Vec2Config, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import load_from_disk

###PRE-TRAINING CODE
# pt_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
#         "facebook/wav2vec2-xls-r-300m",
#         cache_dir="./cache/"
#     )
      
# pt_wav2vec_config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-xls-r-300m")

# # saving config to JSON file
# config_dict = pt_wav2vec_config.to_dict()
# with open(f"pt_wav2vec2_config.json", "w") as F:
# 	json.dump(config_dict, F, indent=4)

# pt_model = Wav2Vec2ForPreTraining(pt_wav2vec_config)

# pt_mal_train = load_from_disk("cptmal_audio_trans_dataset")

# sampling_rate = pt_feature_extractor.sampling_rate
# pt_mal_train = pt_mal_train.cast_column('audio', Audio(sampling_rate=sampling_rate))

# def get_input_values(batch): #original
# 	"""Normalizes input arrays using feature extractor."""
# 	sample = batch['audio']	
# 	batch["input_values"] = pt_feature_extractor(
# 		sample['array'], sampling_rate=sample['sampling_rate'],
# 		return_tensors='np',
#         return_attention_mask=True
# 		).input_values[0]
	
# 	# saving input_length for each sequence, might not be needed for this task.
# 	batch["input_length"] = [batch["input_values"].shape[0]/sample['sampling_rate']]

# 	# manually calling garbage collector to dispose off unallocated memory.
# 	gc.collect()
# 	return batch


# # applying get_input_values function to all the examples 
# pt_mal_train = pt_mal_train.map(
# 		get_input_values,
# 		remove_columns=pt_mal_train.column_names,
# 	)

# def get_seq_indices_not_too_short(dataset, min_length):
# 	"""Returns the list of indices of sequences that are 'good'
# 	meaning longer than min length."""
# 	good_indices = []
# 	all_input_lengths = dataset['input_length']
# 	for i in range(len(dataset)):
# 		if all_input_lengths[i][0] > min_length:
# 			good_indices.append(i)
# 	return good_indices

# # retaining the examples having lengths greater than 3 sec
# good_indices = get_seq_indices_not_too_short(pt_mal_train, 3)
# pt_mal_train = pt_mal_train.select(good_indices)

# # Split the dataset into training and test sets (95% train, 5% test)
# train_test_split = pt_mal_train.train_test_split(test_size=0.05)

# # Extract the training and test sets
# pt_train = train_test_split['train']
# pt_test = train_test_split['test']

# @dataclass
# class DataCollatorForPretraining:

# 	model: Wav2Vec2ForPreTraining
# 	feature_extractor: Wav2Vec2FeatureExtractor
# 	padding: Union[bool, str] = "longest"

# 	def __call__(
# 			self,
# 			features: List[Dict[str, Union[List[int], torch.Tensor]]]
# 		) -> Dict[str, torch.Tensor]:

# 		input_features = [{"input_values": feature["input_values"]} for feature in features]
# 		batch = self.feature_extractor.pad(
# 			input_features,
# 			padding=self.padding,
# 			return_tensors="pt",
# 		)

# 		device = batch['input_values'].device
# 		batch_size, input_seq_len = batch['input_values'].shape

# 		seq_len = self.model._get_feat_extract_output_lengths(input_seq_len).item()

# 		# to avoid computing loss on padded inputs
# 		if batch.get("attention_mask") is not None:
# 			sub_attention_mask = self.model._get_feature_vector_attention_mask(
# 				seq_len, batch["attention_mask"]
# 			)

# 		features_shape = (batch_size, seq_len)

# 		# sample randomly masked indices
# 		mask_time_indices = _compute_mask_indices(
# 			features_shape,
# 			self.model.config.mask_time_prob,
# 			self.model.config.mask_time_length,
# 			attention_mask=sub_attention_mask,
# 		)

# 		# sample negative indices
# 		sampled_negative_indices = _sample_negative_indices(
# 			features_shape,
# 			self.model.config.num_negatives,
# 			mask_time_indices=mask_time_indices,
# 		)

# 		batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
# 		batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

# 		return batch
      
# pt_data_collator = DataCollatorForPretraining(model=pt_model, feature_extractor=pt_feature_extractor)
     
# class CustomTrainer(Trainer):
# 	def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
# 		# If no evaluation dataset is provided, use the default one
# 		eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

# 		# Set up evaluation
# 		self.model.eval()
# 		output = {}
		
# 		# Prepare for accumulation
# 		total_loss = 0.0
# 		total_contrastive_loss = 0.0
# 		total_diversity_loss = 0.0
# 		num_batches = 0

# 		# Create a DataLoader for evaluation
# 		dataloader = self.get_eval_dataloader(eval_dataset)

# 		# Iterate over the DataLoader
# 		for step, batch in enumerate(dataloader):
# 			# Move batch to device
# 			batch = {k: v.to(self.args.device) for k, v in batch.items()}

# 			# Forward pass
# 			with torch.no_grad():
# 				outputs = self.model(**batch)

# 			# Extract loss
# 			loss = outputs.get('loss', None)
# 			contrastive_loss = outputs.get('contrastive_loss', None)
# 			diversity_loss = outputs.get('diversity_loss', None)
# 			if loss is not None:
# 				total_loss += loss.item()
# 				total_contrastive_loss += contrastive_loss.item()
# 				total_diversity_loss += diversity_loss.item() 
# 				num_batches += 1

# 		# Compute average loss
# 		avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
# 		avg_contrastive_loss = total_contrastive_loss / num_batches if num_batches > 0 else float('nan')
# 		avg_diversity_loss = total_diversity_loss / num_batches if num_batches > 0 else float('nan')

# 		# Compute additional metrics
# 		metrics = {
# 			f"{metric_key_prefix}_loss": avg_loss,
# 			f"{metric_key_prefix}_constrast_loss": avg_contrastive_loss,
# 			f"{metric_key_prefix}_div_loss": avg_diversity_loss,
# 		}

# 		# Report metrics
# 		self.log(metrics)

# 		return metrics
      
# training_args = TrainingArguments(
# 		output_dir='wav2vec2-pretraining-res',
# 		gradient_checkpointing=False, 
# 		group_by_length=True,   # groups examples of comparable lengths together
# 		gradient_accumulation_steps=1,
# 		per_device_eval_batch_size=4,
# 		num_train_epochs=10,
# 		per_device_train_batch_size=4,
		
# 		# logging...
# 		logging_strategy='steps',
# 		logging_steps=10,

# 		# save and eval strategy...
# 		save_strategy='steps',
# 		save_steps=100,
# 		save_total_limit=2,
# 		eval_strategy='steps',
# 		eval_steps=100,

# 		learning_rate=1e-4,
# 		weight_decay=0.005,
# 		warmup_ratio=0.1,
		
# 		fp16=True,  # use this only if it is supported by you GPU
# 		report_to=["tensorboard"],
# 		load_best_model_at_end=True,
# 		metric_for_best_model="loss",
# 		# prediction_loss_only=True,
# 		greater_is_better=False,
# 		push_to_hub=False,
# 		)

# pt_trainer = CustomTrainer(
#     model=pt_model,
#     data_collator=pt_data_collator,
#     args=training_args,
#     train_dataset=pt_train,
#     eval_dataset=pt_test,
#     tokenizer=pt_feature_extractor,
# )
# print(f"Starting training...!")
# torch.cuda.empty_cache()
# pt_trainer.train()

###FINE-TUNING CODE

# Load the Malayalam data
mal_data = load_from_disk("cptmal_audio_trans_dataset")

# Split the dataset into training and test sets (80% train, 20% test)
mal_data_split = mal_data.train_test_split(test_size=0.2, seed=121) #ensuring same train split each time

# Extract the training and test sets
mal_data_train = mal_data_split['train']
mal_data_test = mal_data_split['test']

# Function to compute duration of each audio sample
def compute_durations(batch):
    batch["duration"] = [len(a["array"]) / a["sampling_rate"] for a in batch["audio"]]
    return batch

# Compute durations
mal_data_train = mal_data_train.map(compute_durations, batched=True)

selected_samples = []
total_duration = 0.0

for sample in mal_data_train:
    if total_duration + sample["duration"] > (3600 * 3): #three hours
        break
    selected_samples.append(sample)
    total_duration += sample["duration"]

mal_data_train = Dataset.from_list(selected_samples)

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def remove_special_characters(batch):
    # batch["sentence"] = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower()
    batch["transcription"] = re.sub(chars_to_ignore_regex, '', batch["transcription"]).lower()
    return batch

mal_data_train = mal_data_train.map(remove_special_characters)
mal_data_test = mal_data_test.map(remove_special_characters)


def extract_all_chars(batch):
#   all_text = " ".join(batch["sentence"])
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
repo_name = "cpt1-wav2vec2-large-xls-r-300m-malayalam-results"
tokenizer.save_pretrained(repo_name)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("wav2vec2-pretraining-res/checkpoint-47600")
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

# def compute_metrics(pred):
#     pred_logits = pred.predictions
#     pred_ids = np.argmax(pred_logits, axis=-1)

#     pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

#     pred_str = processor.batch_decode(pred_ids)
#     # we do not want to group tokens when computing the metrics
#     label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

#     label_str = [s.replace(processor.tokenizer.pad_token, '') for s in label_str]  # Remove padding

#     cer = cer_metric.compute(predictions=pred_str, references=label_str)

#     return {"cer": cer}

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
        decoded = processor.tokenizer.decode(label, group_tokens=False)
        label_str.append(decoded)

    # Compute CER
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


model = Wav2Vec2ForCTC.from_pretrained(
    "wav2vec2-pretraining-res/checkpoint-47600", 
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

# trainer.train()


model = Wav2Vec2ForCTC.from_pretrained(repo_name+"/checkpoint-840").to("cuda")

processor = Wav2Vec2Processor.from_pretrained(repo_name+"/checkpoint-840")

input_dict = processor(mal_data_test[0]["input_values"], return_tensors="pt", padding=True)

logits = model(input_dict.input_values.to("cuda")).logits

pred_ids = torch.argmax(logits, dim=-1)[0]

mal_data_test_transcription = mal_data_split['test']

sample = mal_data_split["train"]
input_values = processor(sample["audio"]["array"], sampling_rate=16000, return_tensors="pt").input_values

with torch.no_grad():
    logits = model(input_values).logits

pred_ids = torch.argmax(logits, dim=-1)[0].tolist()
print("Pred token ids:", pred_ids)
print("Pred decoded:", processor.decode(pred_ids, group_tokens=False))

label_ids = [id for id in sample["labels"] if id != -100]
print("Label decoded:", processor.decode(label_ids, group_tokens=False))

# print("Prediction:")
# print(processor.decode(pred_ids))

# print("\nReference:")
# # print(mal_data_test_transcription[0]["sentence"].lower())
# print(mal_data_test_transcription[0]["transcription"].lower())

def map_to_result(batch):
  with torch.no_grad():
    input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
    logits = model(input_values).logits

  pred_ids = torch.argmax(logits, dim=-1)
  batch["pred_str"] = processor.batch_decode(pred_ids)[0]
  batch["text"] = processor.decode([id for id in batch["labels"] if id != -100], group_tokens=False)  # Remove -100
  
  return batch

results = mal_data_test.map(map_to_result, remove_columns=mal_data_test.column_names)

# print(results["pred_str"])
# print(results["text"])

print("Test CER: {:.3f}".format(cer_metric.compute(predictions=results["pred_str"], references=results["text"])))
