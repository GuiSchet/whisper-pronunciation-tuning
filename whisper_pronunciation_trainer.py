"""
Fine-tuning de Whisper con LoRA para análisis de pronunciación usando speechocean762
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from datasets import DatasetDict
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    WhisperTokenizer,
    get_scheduler
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
    
    # Entrenamiento
    output_dir: str = "./whisper-pronunciation-tuned"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # Evaluación y guardado
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
    
    # Logging
    logging_steps: int = 50
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    run_name: str = "whisper-pronunciation-finetuning"
    
    # Datos
    max_length: int = 448
    cache_dir: str = "./cache"
    
    # Otros
    fp16: bool = True
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
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
        """Configurar modelo Whisper y processor con LoRA"""
        logger.info(f"Cargando modelo base: {self.config.model_name}")
        
        # Cargar modelo base
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Configurar LoRA
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules
        )
        
        # Aplicar LoRA al modelo
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        # Cargar processor y tokenizer
        self.processor = WhisperProcessor.from_pretrained(self.config.model_name)
        self.tokenizer = WhisperTokenizer.from_pretrained(self.config.model_name)
        
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
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Reemplazar -100 en labels
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
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
    
    def collate_fn(self, features):
        """Data collator personalizado"""
        batch = {}
        
        # Stack input features
        input_features = torch.stack([f["input_features"] for f in features])
        batch["input_features"] = input_features
        
        # Procesar labels
        labels = [f["labels"] for f in features]
        labels = torch.stack(labels)
        batch["labels"] = labels
        
        return batch

    def create_data_collator(self):
        """Crear data collator personalizado"""
        return self.collate_fn
    
    def train(self):
        """Entrenar el modelo"""
        logger.info("Iniciando entrenamiento...")
        
        # Preparar datos
        dataset = self.prepare_data()
        
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
            fp16=self.config.fp16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            remove_unused_columns=self.config.remove_unused_columns,
            seed=self.config.seed,
            eval_strategy="steps",
            predict_with_generate=True,
            generation_max_length=self.config.max_length,
            push_to_hub=False
        )
        
        # Crear trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=self.processor.feature_extractor,
            data_collator=self.create_data_collator(),
            compute_metrics=self.compute_metrics
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