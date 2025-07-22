# unsloth_trainer.py
"""
Trainer basado en UnsloThAI para Whisper, adaptado a mispeech/speechocean762
- Usa UnsloTh para acelerar entrenamiento
- Preprocesa el dataset para extraer phonemas y scores
- Incluye validación específica de pronunciación
- Early stopping y logging avanzado
"""
import os
import torch
from datasets import load_dataset, DatasetDict, Dataset
from dotenv import load_dotenv
from transformers import GenerationConfig, EarlyStoppingCallback
import wandb
import logging

# --- UnsloTh imports ---
try:
    from unsloth import FastWhisperForConditionalGeneration, FastWhisperProcessor, FastSeq2SeqTrainer, FastSeq2SeqTrainingArguments
except ImportError:
    raise ImportError("Debes instalar UnsloTh: pip install unsloth")

# --- CONFIGURACIÓN PRINCIPAL ---
MODEL_NAME = "openai/whisper-base"
DATASET_NAME = "mispeech/speechocean762"
OUTPUT_DIR = "./whisper-pronunciation-assessment-local"
TRAIN_SAMPLES = 2000
TEST_SAMPLES = 400

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_wandb():
    try:
        wandb.init(
            project="whisper-pronunciation-tuning",
            name="unsloth-phoneme-score-training",
            config={
                "model": MODEL_NAME,
                "dataset": DATASET_NAME,
                "train_samples": TRAIN_SAMPLES,
                "test_samples": TEST_SAMPLES,
            }
        )
        return True
    except Exception as e:
        logger.warning(f"No se pudo inicializar wandb: {e}")
        return False

def validate_model_output(model, processor, sample_input, device):
    model.eval()
    with torch.no_grad():
        try:
            if hasattr(sample_input, 'to'):
                sample_input = sample_input.to(device)
            predicted_ids = model.generate(
                sample_input,
                max_length=50,
                min_length=10,
                num_beams=1,
                do_sample=False,
                early_stopping=True,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
            output = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            has_numbers = any(char.isdigit() for char in output)
            logger.info(f"Validación del modelo - Salida: '{output[:100]}...'")
            logger.info(f"Contiene números (scores): {has_numbers}")
            return output, has_numbers
        except Exception as e:
            logger.error(f"Error en validación: {e}")
            return "", False
        finally:
            model.train()

def main():
    print("--- Paso 1: Configuración Inicial (UnsloTh) ---")
    load_dotenv()
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token:
        raise ValueError("No se encontró el token de Hugging Face. Agrega HUGGING_FACE_TOKEN al .env")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo detectado: {device}")
    use_wandb = setup_wandb()
    if device == "cpu":
        print("ADVERTENCIA: No se detectó una GPU. El entrenamiento será muy lento.")

    # --- Paso 2: Carga de modelo UnsloTh ---
    print("\n--- Paso 2: Cargando modelo UnsloTh ---")
    processor = FastWhisperProcessor.from_pretrained(MODEL_NAME)
    model = FastWhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(device)
    model.config.use_cache = False
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.begin_suppress_tokens = []
    generation_config = GenerationConfig(
        max_length=256,
        min_length=20,
        num_beams=1,
        do_sample=False,
        early_stopping=True,
        pad_token_id=processor.tokenizer.eos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        forced_decoder_ids=None,
        suppress_tokens=[],
        begin_suppress_tokens=[],
    )
    model.generation_config = generation_config

    print(f"Cargando dataset '{DATASET_NAME}' con samples limitados...")
    raw_dataset = load_dataset(DATASET_NAME, "default", streaming=True, token=hf_token)
    train_subset = list(raw_dataset["train"].take(TRAIN_SAMPLES))
    test_subset = list(raw_dataset["test"].take(TEST_SAMPLES))
    dataset = DatasetDict({
        "train": Dataset.from_list(train_subset),
        "test": Dataset.from_list(test_subset)
    })
    print(f"Dataset cargado: {len(train_subset)} train, {len(test_subset)} test")
    sample = dataset["train"][0]
    print(f"Texto: {sample['text']}")
    words = sample['words']
    phonemes = []
    scores = []
    for word in words:
        phonemes.extend(word['phones'])
        scores.extend(word['phones-accuracy'])
    example_label = " ".join([f"{p} {s}" for p, s in zip(phonemes[:5], scores[:5])])
    print(f"Label ejemplo: {example_label}")

    print("\n--- Paso 3: Preprocesando Datos (UnsloTh) ---")
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
            max_phonemes = 20
            phonemes = phonemes[:max_phonemes]
            scores = scores[:max_phonemes]
            label_str = " ".join([f"{p} {s}" for p, s in zip(phonemes, scores)])
            target_labels.append(label_str)
        labels = processor.tokenizer(
            target_labels,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        ).input_ids
        inputs["labels"] = [
            [-100 if token == processor.tokenizer.pad_token_id else token for token in label]
            for label in labels
        ]
        return inputs

    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=8,
        remove_columns=dataset["train"].column_names,
        num_proc=1,
    )
    print("Preprocesamiento completado.")

    print("\n--- Validación Inicial del Modelo ---")
    sample_audio = processed_dataset["test"][0]["input_features"].unsqueeze(0)
    initial_output, has_numbers = validate_model_output(model, processor, sample_audio, device)
    print(f"Salida inicial del modelo: {initial_output}")

    print("\n--- Paso 4: Entrenamiento Optimizado UnsloTh ---")
    training_args = FastSeq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_steps=200,
        num_train_epochs=15,
        max_steps=-1,
        fp16=True,
        gradient_checkpointing=True,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        report_to="wandb" if use_wandb else None,
        remove_unused_columns=False,
        label_smoothing_factor=0.1,
    )

    class PronunciationTrainer(FastSeq2SeqTrainer):
        def evaluate(self, **kwargs):
            result = super().evaluate(**kwargs)
            sample_input = self.eval_dataset[0]["input_features"].unsqueeze(0)
            output, has_numbers = validate_model_output(
                self.model, self.tokenizer, sample_input, self.args.device
            )
            result["eval_sample_output"] = output[:100]
            result["eval_has_numbers"] = has_numbers
            logger.info(f"Eval - Sample output: {output[:100]}")
            logger.info(f"Eval - Has numbers: {has_numbers}")
            return result

    trainer = PronunciationTrainer(
        args=training_args,
        model=model,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["test"],
        tokenizer=processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("¡Iniciando entrenamiento UnsloTh!")
    try:
        trainer.train()
        print("\n¡Entrenamiento completado exitosamente!")
        print("\n--- Validación Final ---")
        final_output, final_has_numbers = validate_model_output(
            model, processor, sample_audio, device
        )
        print(f"Salida final: {final_output}")
        print(f"Tiene números (scores): {final_has_numbers}")
        if final_has_numbers:
            print("✅ ¡El modelo parece haber aprendido a generar scores!")
        else:
            print("❌ El modelo aún no genera scores correctamente.")
    except torch.cuda.OutOfMemoryError:
        print("\n¡ERROR FATAL: CUDA out of memory! Reduce per_device_train_batch_size a 4 y vuelve a intentar.")
        return
    except Exception as e:
        print(f"\nError durante entrenamiento: {e}")
        return

    print(f"\n--- Paso 5: Guardando el modelo mejorado en '{OUTPUT_DIR}' ---")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("¡Modelo y procesador guardados!")
    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main() 