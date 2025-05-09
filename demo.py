import torch
import asyncio
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jiwer import cer
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from googletrans import Translator

# --- 1. Paths and inputs
model_dir = "cpt4-wav2vec2-large-xls-r-300m-mal30-results/checkpoint-1950"
audio_path = "demo/sonia_set1_11.wav"
reference_text = "മൂന്നാം റൗണ്ടിൽ ജർമൻ താരം ആഞ്ചലിക് കെർബറാണ് ഷറപ്പോവയെ വീഴ്ത്തിയത്"

# --- 2. Load model and processor
processor = Wav2Vec2Processor.from_pretrained(model_dir)
model = Wav2Vec2ForCTC.from_pretrained(model_dir)
model.eval()

# --- 3. Load and preprocess audio
speech, sr = librosa.load(audio_path, sr=16000)
input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values

# --- 4. Inference
with torch.no_grad():
    logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
predicted_text = processor.decode(predicted_ids[0])

# --- 5. Transliteration
romanized_ref = transliterate(reference_text, sanscript.MALAYALAM, sanscript.ITRANS)
romanized_pred = transliterate(predicted_text, sanscript.MALAYALAM, sanscript.ITRANS)

# --- 8. Output everything
print("\n--- Reference ---")
print(f"Malayalam:   {reference_text}")
print(f"Romanized:   {romanized_ref}")

print("\n--- Prediction ---")
print(f"Malayalam:   {predicted_text}")
print(f"Romanized:   {romanized_pred}")

# print(f"\n--- CER: {cer_score:.4f}")
