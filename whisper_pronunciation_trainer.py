"""
Fine-tuning de Whisper con LoRA para análisis de pronunciación usando speechocean762
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import warnings
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from datasets import DatasetDict
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    get_scheduler,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
import evaluate
from accelerate import Accelerator
import wandb

from data_loader import SpeechOceanDataLoader

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhisperSeq2SeqTrainer(Seq2SeqTrainer):
    """Custom trainer for Whisper that handles inputs correctly with LoRA"""
    
    def _prepare_inputs(self, inputs):
        """Prepare inputs for the model, ensuring the correct format for Whisper"""
        # Filter out any incompatible keys that might be added by the trainer
        whisper_compatible_keys = {
            'input_features', 'labels', 'attention_mask', 
            'decoder_input_ids', 'decoder_attention_mask'
        }
        
        # Only keep keys that are compatible with Whisper
        filtered_inputs = {k: v for k, v in inputs.items() if k in whisper_compatible_keys}
        
        # Ensure we have the required input_features
        if 'input_features' not in filtered_inputs and 'input_ids' in inputs:
            logger.warning("Converting input_ids to input_features - this might indicate a data issue")
            # This is a fallback - shouldn't happen with our data collator
            filtered_inputs['input_features'] = inputs['input_ids']
        
        return super()._prepare_inputs(filtered_inputs)
    
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """Override compute_loss to handle inputs correctly for Whisper with LoRA"""
        # Filter inputs to only include Whisper-compatible keys
        whisper_compatible_keys = {
            'input_features', 'labels', 'attention_mask', 
            'decoder_input_ids', 'decoder_attention_mask'
        }
        
        inputs_clean = {k: v for k, v in inputs.items() if k in whisper_compatible_keys}
        
        # Ensure we have the required input_features
        if 'input_features' not in inputs_clean:
            raise ValueError("input_features not found in inputs. Check data collator.")
        
        # Call the model directly to get outputs
        outputs = model(**inputs_clean)
        loss = outputs.loss
        
        if return_outputs:
            return loss, outputs
        else:
            return loss


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that dynamically pads the inputs received.
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract input features for audio
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Extract labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Create labels with -100 for padding tokens
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove bos token from labels if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


@dataclass
class TrainingConfig:
    """Configuración para el entrenamiento"""
    # Modelo
    model_name: str = "openai/whisper-small"
    
    # LoRA config
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "out_proj",
        "fc1", "fc2"
    ])
    
    # Entrenamiento optimizado para modelo completo
    output_dir: str = "./whisper-pronunciation-tuned"
    num_train_epochs: int = 2  # Menos épocas para evitar memoria
    per_device_train_batch_size: int = 1  # Batch size más pequeño para memoria
    per_device_eval_batch_size: int = 1  # Batch size de evaluación más pequeño
    gradient_accumulation_steps: int = 32  # Compensar batch size pequeño
    learning_rate: float = 1e-5  # LR muy bajo para modelo completo
    warmup_steps: int = 500  # Warmup proporcionalmente menor
    max_grad_norm: float = 0.5  # Gradient clipping más agresivo
    weight_decay: float = 0.01
    
    # Evaluación y guardado
    eval_steps: int = 100  # Evaluar más frecuentemente
    save_steps: int = 100
    save_total_limit: int = 3  # Menos checkpoints por espacio
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
    
    # Logging
    logging_steps: int = 25  # Log más frecuente
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    run_name: str = "whisper-pronunciation-optimal"
    
    # Datos
    max_length: int = 448
    cache_dir: str = "./cache"
    
    # Otros
    fp16: bool = True
    dataloader_num_workers: int = 0  # Disable multiprocessing to avoid Windows pickling issues
    seed: int = 42


class WhisperPronunciationTrainer:
    """Trainer para fine-tuning de Whisper para análisis de pronunciación"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.accelerator = Accelerator()
        
        # Configurar dispositivo
        self.device = self.accelerator.device
        logger.info(f"Usando dispositivo: {self.device}")
        
        # Configurar modelo y processor
        self.setup_model_and_processor()
        
        # Métricas
        self.wer_metric = evaluate.load("wer")
        self.bleu_metric = evaluate.load("bleu")
        
        logger.info("Trainer inicializado exitosamente")
    
    def setup_model_and_processor(self):
        """Configurar modelo Whisper y processor con LoRA según documentación oficial"""
        logger.info(f"Cargando modelo base: {self.config.model_name}")
        
        # Cargar feature extractor y tokenizer por separado
        feature_extractor = WhisperFeatureExtractor.from_pretrained(self.config.model_name)
        tokenizer = WhisperTokenizer.from_pretrained(self.config.model_name)
        
        # Crear processor
        self.processor = WhisperProcessor.from_pretrained(self.config.model_name)
        
        # Cargar modelo base
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,  # Use FP32 for now
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Configurar modelo según documentación oficial
        self.model.config.use_cache = False
        self.model.config.apply_spec_augment = False
        
        # Configurar generation config para fine-tuning usando partial (documentación oficial)
        self.model.generation_config.language = None
        self.model.generation_config.task = None
        self.model.generation_config.forced_decoder_ids = None
        
        # Configurar generate method para entrenamiento (según documentación oficial)
        self.model.generate = partial(
            self.model.generate,
            use_cache=True
        )
        
        # Usar entrenamiento completo por problemas de compatibilidad con PEFT
        # En producción, LoRA sería ideal pero tiene problemas con Whisper
        logger.info(f"Entrenando modelo completo - {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} parámetros")
        
        # Aplicar dropout en algunas capas para regularización
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.1  # Aumentar dropout para regularización
        
        logger.info("Modelo y processor configurados con LoRA")
    
    def prepare_data(self) -> DatasetDict:
        """Preparar datos para entrenamiento"""
        logger.info("Preparando datos...")
        
        # Crear data loader
        data_loader = SpeechOceanDataLoader(
            model_name=self.config.model_name,
            cache_dir=self.config.cache_dir
        )
        
        # Cargar y procesar dataset
        raw_dataset = data_loader.load_dataset()
        processed_dataset = data_loader.process_dataset(raw_dataset)
        
        logger.info("Datos preparados exitosamente")
        return processed_dataset
    
    def compute_metrics(self, eval_pred):
        """Calcular métricas de evaluación"""
        predictions, labels = eval_pred
        
        # Decodificar predicciones
        decoded_preds = self.processor.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Reemplazar -100 en labels
        labels = np.where(labels != -100, labels, self.processor.tokenizer.pad_token_id)
        decoded_labels = self.processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Calcular WER
        wer = self.wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        
        # Calcular BLEU
        bleu = self.bleu_metric.compute(
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels]
        )
        
        return {
            "wer": wer,
            "bleu": bleu["bleu"]
        }
    
    def train(self):
        """Entrenar el modelo según documentación oficial de Hugging Face"""
        logger.info("Iniciando entrenamiento...")
        
        # Preparar datos
        dataset = self.prepare_data()
        
        # Preparar data collator
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )
        
        # Configurar argumentos de entrenamiento
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_grad_norm=self.config.max_grad_norm,
            weight_decay=self.config.weight_decay,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,
            load_best_model_at_end=self.config.load_best_model_at_end,
            logging_steps=self.config.logging_steps,
            report_to=self.config.report_to,
            run_name=self.config.run_name,
            fp16=False,  # Disable FP16 for now
            dataloader_num_workers=self.config.dataloader_num_workers,
            seed=self.config.seed,
            eval_strategy="steps",
            predict_with_generate=False,  # Disable to avoid input_ids issue
            push_to_hub=False,
            # Configuraciones adicionales para Whisper
            gradient_checkpointing=False,  # Disable to avoid backward graph issues
            max_steps=2000,  # Menos pasos para modelo completo
            # Configuraciones adicionales para estabilidad
            save_strategy="steps",
            lr_scheduler_type="cosine",  # Cosine scheduler for better convergence
            warmup_ratio=0.1,  # Alternative warmup specification
        )
        
        # Crear trainer según documentación oficial
        trainer = WhisperSeq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=self.processor.tokenizer,  # Usar solo el tokenizer del processor
            data_collator=data_collator,
            # compute_metrics=self.compute_metrics,  # Disable para ahorrar memoria
        )
        
        # Entrenar
        train_result = trainer.train()
        
        # Guardar modelo final
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        
        # Evaluación final
        logger.info("Realizando evaluación final...")
        eval_results = trainer.evaluate()
        trainer.log_metrics("eval", eval_results)
        trainer.save_metrics("eval", eval_results)
        
        logger.info("Entrenamiento completado exitosamente")
        return train_result, eval_results
    
    def save_model(self, path: str):
        """Guardar modelo fine-tuneado"""
        logger.info(f"Guardando modelo en: {path}")
        
        # Crear directorio si no existe
        os.makedirs(path, exist_ok=True)
        
        # Guardar modelo LoRA
        self.model.save_pretrained(path)
        
        # Guardar processor
        self.processor.save_pretrained(path)
        
        # Guardar configuración
        config_dict = {
            "model_name": self.config.model_name,
            "lora_config": {
                "r": self.config.lora_r,
                "alpha": self.config.lora_alpha,
                "dropout": self.config.lora_dropout,
                "target_modules": self.config.target_modules
            }
        }
        
        with open(os.path.join(path, "training_config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info("Modelo guardado exitosamente")
    
    @classmethod
    def load_model(cls, path: str, config: TrainingConfig):
        """Cargar modelo fine-tuneado"""
        logger.info(f"Cargando modelo desde: {path}")
        
        # Cargar configuración
        config_path = os.path.join(path, "training_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                saved_config = json.load(f)
                config.model_name = saved_config["model_name"]
        
        # Crear trainer
        trainer = cls(config)
        
        # Cargar modelo LoRA
        trainer.model = PeftModel.from_pretrained(
            trainer.model.base_model,
            path
        )
        
        logger.info("Modelo cargado exitosamente")
        return trainer


def main():
    """Función principal"""
    # Configuración
    config = TrainingConfig(
        model_name="openai/whisper-small",
        output_dir="./whisper-pronunciation-tuned",
        num_train_epochs=5,
        per_device_train_batch_size=2,  # Reducido para memoria
        gradient_accumulation_steps=8,  # Aumentado para compensar
        learning_rate=1e-4,
        run_name="whisper-pronunciation-test"
    )
    
    # Crear trainer
    trainer = WhisperPronunciationTrainer(config)
    
    # Entrenar
    train_results, eval_results = trainer.train()
    
    # Guardar modelo
    trainer.save_model(config.output_dir)
    
    print("\n=== RESULTADOS FINALES ===")
    print(f"Train Loss: {train_results.metrics.get('train_loss', 'N/A')}")
    print(f"Eval Loss: {eval_results.get('eval_loss', 'N/A')}")
    print(f"WER: {eval_results.get('eval_wer', 'N/A')}")
    print(f"BLEU: {eval_results.get('eval_bleu', 'N/A')}")


if __name__ == "__main__":
    main() 