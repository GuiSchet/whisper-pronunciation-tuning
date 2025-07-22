#!/usr/bin/env python3
"""
Script rápido para probar la configuración del modelo antes del entrenamiento completo.
Esto ayuda a identificar problemas sin perder tiempo en un entrenamiento largo.
"""

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from dotenv import load_dotenv
import os

def test_model_setup():
    """
    Prueba rápida del setup del modelo y tokenización
    """
    print("🧪 Iniciando prueba rápida del modelo...")
    
    # Cargar configuración
    load_dotenv()
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token:
        print("❌ Error: No se encontró HUGGING_FACE_TOKEN")
        return False
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"📱 Dispositivo: {device}")
    
    try:
        # 1. Cargar modelo y procesador
        print("\n1️⃣ Cargando modelo base...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        model.to(device)
        print("✅ Modelo cargado correctamente")
        
        # 2. Probar tokenización de fonemas
        print("\n2️⃣ Probando tokenización de fonemas...")
        test_phoneme_labels = [
            "W 2.0 IY0 2.0 K 2.0",
            "AO0 1.8 L 1.8 IH0 2.0",
            "EH0 1.0 R 1.0"
        ]
        
        for i, label in enumerate(test_phoneme_labels):
            tokens = processor.tokenizer.tokenize(label)
            token_ids = processor.tokenizer.convert_tokens_to_ids(tokens)
            decoded = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
            
            print(f"   Test {i+1}: '{label}'")
            print(f"   Tokens: {tokens}")
            print(f"   Decodificado: '{decoded}'")
            print(f"   ¿Coincide?: {'✅' if decoded.strip() == label.strip() else '❌'}")
        
        # 3. Cargar muestra del dataset
        print("\n3️⃣ Cargando muestra del dataset...")
        dataset = load_dataset("mispeech/speechocean762", "default", streaming=True, token=hf_token)
        sample = next(iter(dataset["train"]))
        
        # Procesar muestra como en el entrenamiento
        audio_array = sample["audio"]["array"]
        words = sample['words']
        
        phonemes = []
        scores = []
        for word in words:
            phonemes.extend(word['phones'])
            scores.extend(word['phones-accuracy'])
        
        # Limitar a primeros 10 fonemas para prueba
        phonemes = phonemes[:10]
        scores = scores[:10]
        label_str = " ".join([f"{p} {s}" for p, s in zip(phonemes, scores)])
        
        print(f"   Texto original: {sample['text']}")
        print(f"   Label generado: {label_str}")
        
        # 4. Probar procesamiento de audio
        print("\n4️⃣ Probando procesamiento de audio...")
        inputs = processor(audio=audio_array, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        print(f"   Shape de input_features: {input_features.shape}")
        
        # 5. Probar tokenización del label
        label_tokens = processor.tokenizer(
            label_str, 
            padding="max_length", 
            max_length=128, 
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        print(f"   Shape de label_tokens: {label_tokens.shape}")
        print(f"   Primeros 10 token IDs: {label_tokens[0][:10].tolist()}")
        
        # 6. Probar generación inicial (sin entrenamiento)
        print("\n5️⃣ Probando generación inicial...")
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                max_length=50,
                min_length=5,
                num_beams=1,
                do_sample=False,
                early_stopping=True,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        
        output = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        print(f"   Salida inicial del modelo: '{output}'")
        
        # Verificar si contiene números (indicativo de scores)
        has_numbers = any(char.isdigit() for char in output)
        print(f"   ¿Contiene números?: {'✅' if has_numbers else '❌'}")
        
        # 7. Verificar configuración de entrenamiento
        print("\n6️⃣ Verificando configuración de entrenamiento...")
        
        # Simular un step de entrenamiento
        model.train()
        labels = label_tokens.clone().to(device)  # IMPORTANTE: Mover labels a GPU
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        try:
            with torch.no_grad():
                outputs = model(input_features=input_features, labels=labels)
                loss = outputs.loss
                print(f"   Loss inicial: {loss.item():.4f}")
                print("✅ Forward pass exitoso")
        except Exception as e:
            print(f"❌ Error en forward pass: {e}")
            return False
        
        print("\n🎉 ¡Todas las pruebas pasaron! El setup está listo para entrenamiento.")
        return True
        
    except Exception as e:
        print(f"❌ Error durante las pruebas: {e}")
        return False

def main():
    """Función principal"""
    print("=" * 60)
    print("🧪 PRUEBA RÁPIDA DEL MODELO WHISPER PRONUNCIATION")
    print("=" * 60)
    
    success = test_model_setup()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ RESULTADO: Todo listo para entrenamiento!")
        print("💡 Puedes ejecutar: uv run python trainer.py")
    else:
        print("❌ RESULTADO: Hay problemas que resolver antes del entrenamiento")
    print("=" * 60)

if __name__ == "__main__":
    main() 