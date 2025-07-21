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
TRAIN_SAMPLES = 800  # Reducido para un ejemplo rápido. ¡Auméntalo para un modelo real!
TEST_SAMPLES = 200

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
    train_subset = list(raw_dataset["train"].take(TRAIN_SAMPLES))
    test_subset = list(raw_dataset["test"].take(TEST_SAMPLES))
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
        num_proc=4, # Acelera el preprocesamiento usando múltiples núcleos de CPU
    )
    print("Preprocesamiento completado.")

    # 4. ENTRENAMIENTO DEL MODELO
    print("\n--- Paso 4: Configurando el Entrenamiento ---")

    # Si recibes un error de "CUDA out of memory", la primera cosa que debes
    # reducir es 'per_device_train_batch_size' (por ejemplo, a 4, 2 o incluso 1).
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        warmup_steps=100,
        max_steps=1000,
        gradient_checkpointing=False, # ¡Clave para ahorrar memoria de la GPU!
        eval_strategy="steps",  # <-- CORRECTO según la doc oficial
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=256,
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["test"],
        tokenizer=processor.feature_extractor,
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