from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch

# Cargar modelo y processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

# Cargar las primeras 5 muestras del split test
ds = load_dataset("mispeech/speechocean762", split="test")

for i in range(5):
    audio = ds[i]["audio"]["array"]
    text = ds[i]["text"]
    # Tokenizar
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
    # Inferencia
    with torch.no_grad():
        logits = model(input_values).logits
    # Argmax y decodificación
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    print(f"\n--- Muestra {i+1} ---")
    print(f"Texto real: {text}")
    print(f"Transcripción fonética: {transcription[0]}") 