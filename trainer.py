import os
import torch
from dotenv import load_dotenv
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)


# --- CONFIGURACIÓN PRINCIPAL ---
# Puedes ajustar estos parámetros
MODEL_NAME = "openai/whisper-base"
DATASET_NAME = "mispeech/speechocean762"
OUTPUT_DIR = "./whisper-pronunciation-assessment-local"
# Usar todo el dataset, sin límites de muestras
TRAIN_SAMPLES = None  # None para usar todo
TEST_SAMPLES = None

def main():
    """
    Función principal que encapsula todo el proceso de entrenamiento.
    """
    # 1. CARGA DE VARIABLES DE ENTORNO Y CONFIGURACIÓN DEL DISPOSITIVO
    print("--- Paso 1: Configuración Inicial ---")
    load_dotenv()
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token:
        raise ValueError(
            "No se encontró el token de Hugging Face. "
            "Asegúrate de tener un archivo .env con HUGGING_FACE_TOKEN='hf_...'"
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo detectado: {device}")
    if device == "cpu":
        print("ADVERTENCIA: No se detectó una GPU. El entrenamiento será extremadamente lento.")

    # 2. CARGA DEL MODELO, PROCESADOR Y DATASET
    print("\n--- Paso 2: Cargando Modelo, Procesador y Dataset ---")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(device)

    # Configuraciones críticas para el fine-tuning
    model.config.use_cache = False
    model.config.forced_decoder_ids = None

    print(f"Cargando dataset '{DATASET_NAME}'...")
    raw_dataset = load_dataset(DATASET_NAME, "default", streaming=True, token=hf_token)

    # Convertir el dataset de streaming a un dataset en memoria para procesarlo
    train_subset = list(raw_dataset["train"]) if TRAIN_SAMPLES is None else list(raw_dataset["train"].take(TRAIN_SAMPLES))
    test_subset = list(raw_dataset["test"]) if TEST_SAMPLES is None else list(raw_dataset["test"].take(TEST_SAMPLES))
    dataset = DatasetDict({
        "train": Dataset.from_list(train_subset),
        "test": Dataset.from_list(test_subset)
    })

    print("Ejemplo de un dato del dataset:")
    print(dataset["train"][0]['text'])

    # 3. PREPROCESAMIENTO DE LOS DATOS
    print("\n--- Paso 3: Preprocesando Datos ---")

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = processor(audio=audio_arrays, sampling_rate=16000, return_tensors="pt")

        target_labels = []
        for words in examples['words']:
            phonemes = []
            scores = []
            for word in words:
                phonemes.extend(word['phones'])
                scores.extend(word['phones-accuracy'])
            label_str = " ".join([f"{p} {s}" for p, s in zip(phonemes, scores)])
            target_labels.append(label_str)

        labels = processor.tokenizer(target_labels, padding="max_length", max_length=256, truncation=True).input_ids
        inputs["labels"] = [[-100 if token == processor.tokenizer.pad_token_id else token for token in label] for label in labels]
        return inputs

    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=16,
        remove_columns=dataset["train"].column_names,
        num_proc=1,  # Solo 1 proceso para evitar sobrecargar la RAM
    )
    print("Preprocesamiento completado.")

    # 4. ENTRENAMIENTO DEL MODELO
    print("\n--- Paso 4: Configurando el Entrenamiento ---")

    # Ajustes optimizados para RTX 3060
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,  # Ajusta a 32 si tienes suficiente VRAM, o reduce si da error
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=500,
        num_train_epochs=10,  # Entrena por epochs, no por steps
        gradient_checkpointing=False,  # Ahorra memoria
        fp16=True,  # Mixed precision
        eval_strategy="steps",
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        generation_max_length=256,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=3,  # No llenar el disco
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["test"],
        tokenizer=processor,  # Usa el processor completo
    )

    print("¡Iniciando entrenamiento!")
    try:
        trainer.train()
        print("\n¡Entrenamiento completado exitosamente!")
    except torch.cuda.OutOfMemoryError:
        print("\n¡ERROR FATAL: CUDA out of memory!")
        print("La GPU no tiene suficiente memoria para el tamaño de lote actual.")
        print("Solución: Reduce 'per_device_train_batch_size' en los TrainingArguments y vuelve a intentarlo.")
        return # Salir del script

    # 5. GUARDADO FINAL DEL MODELO
    print(f"\n--- Paso 5: Guardando el modelo final en '{OUTPUT_DIR}' ---")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("¡Modelo y procesador guardados!")


if __name__ == "__main__":
    main()