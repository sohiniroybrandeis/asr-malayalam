import sys
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from jiwer import cer

# 1. Get model, audio file, and reference transcription from command-line
model_dir = sys.argv[1]
audio_path = sys.argv[2]
reference_text = sys.argv[3]  # Give the correct transcription as an argument

# 2. Load model and processor
processor = Wav2Vec2Processor.from_pretrained(model_dir)
model = Wav2Vec2ForCTC.from_pretrained(model_dir)
model.eval()

# 3. Load and preprocess audio
speech, sr = librosa.load(audio_path, sr=16000)
input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values

# 4. Run inference
with torch.no_grad():
    logits = model(input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
predicted_text = processor.decode(predicted_ids[0])

# 5. Compute CER
cer_score = cer(reference_text.lower(), predicted_text.lower())

# 6. Output results
print("\n--- Comparison ---")
print(f"Reference:   {reference_text}")
print(f"Prediction:  {predicted_text}")
print(f"\nCharacter Error Rate (CER): {cer_score:.4f}")
