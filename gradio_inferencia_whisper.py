import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import gradio as gr
import os

# Ruta al modelo entrenado
MODEL_DIR = "./whisper-pronunciation-assessment-local"

def load_model():
    processor = WhisperProcessor.from_pretrained(MODEL_DIR)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

processor, model, device = load_model()


def infer(audio):
    # audio: (sr, data) tuple from gradio
    if audio is None:
        return "No audio recibido."
    sampling_rate, data = audio
    # Procesar el audio
    inputs = processor(audio=data, sampling_rate=sampling_rate, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    # Generar predicción
    with torch.no_grad():
        predicted_ids = model.generate(input_features, max_length=256)
    # Decodificar salida cruda (sin skip_special_tokens)
    raw_output = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=False)[0]
    return raw_output


demo = gr.Interface(
    fn=infer,
    inputs=gr.Audio(source="upload", type="numpy", label="Sube tu audio"),
    outputs=gr.Textbox(label="Salida cruda del modelo"),
    title="Inferencia Whisper Pronunciation Tuning",
    description="Sube un audio y obtén la salida cruda del modelo entrenado."
)

if __name__ == "__main__":
    demo.launch() 