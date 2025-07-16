"""
Data loader para el dataset speechocean762 optimizado para fine-tuning de Whisper
para análisis de pronunciación.
"""

import os
import json
from typing import Dict, List, Optional, Union, Tuple
import logging
from dataclasses import dataclass

import torch
import librosa
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import WhisperProcessor, WhisperTokenizer
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PronunciationAnalysis:
    """Clase para estructurar el análisis de pronunciación"""
    transcript: str
    accuracy_score: float
    fluency_score: float
    completeness_score: float
    prosodic_score: float
    total_score: float
    word_errors: List[Dict]
    phoneme_errors: List[Dict]
    mispronunciations: List[Dict]


class SpeechOceanDataLoader:
    """
    DataLoader para el dataset speechocean762 optimizado para fine-tuning de Whisper
    para análisis de pronunciación.
    """
    
    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        target_sampling_rate: int = 16000,
        max_duration: float = 30.0,
        cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.target_sampling_rate = target_sampling_rate
        self.max_duration = max_duration
        self.cache_dir = cache_dir
        
        # Inicializar processor y tokenizer de Whisper
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.tokenizer = WhisperTokenizer.from_pretrained(model_name)
        
        logger.info(f"Inicializando DataLoader con modelo: {model_name}")
    
    def load_dataset(self) -> DatasetDict:
        """Cargar el dataset speechocean762 desde Hugging Face"""
        try:
            logger.info("Cargando dataset speechocean762...")
            
            # Cargar dataset sin modificar el feature de audio
            dataset = load_dataset(
                "mispeech/speechocean762", 
                cache_dir=self.cache_dir
            )
            
            logger.info(f"Dataset cargado exitosamente. Splits disponibles: {list(dataset.keys())}")
            
            # Mostrar estadísticas del dataset
            for split_name, split_data in dataset.items():
                logger.info(f"{split_name}: {len(split_data)} muestras")
            
            return dataset
        except Exception as e:
            logger.error(f"Error cargando dataset: {e}")
            raise
    
    def preprocess_audio(self, audio: Dict) -> np.ndarray:
        """Preprocesar audio para Whisper"""
        try:
            # El audio ahora viene ya decodificado automáticamente por el dataset
            if isinstance(audio, dict):
                if "array" in audio:
                    # Audio ya decodificado
                    audio_array = audio["array"]
                    sampling_rate = audio["sampling_rate"]
                elif "path" in audio:
                    # Fallback: usar librosa directamente si no está decodificado
                    audio_array, sampling_rate = librosa.load(audio["path"], sr=None)
                else:
                    raise ValueError("Audio dict no contiene 'array' ni 'path'")
            else:
                # Si es una ruta directa
                audio_array, sampling_rate = librosa.load(audio, sr=None)
        except Exception as e:
            logger.warning(f"Error cargando audio: {e}")
            # Fallback: crear audio sintético
            logger.warning("Usando audio sintético como fallback")
            audio_array = np.random.randn(16000)  # 1 segundo de ruido
            sampling_rate = 16000
        
        # Resamplear si es necesario (aunque ahora el dataset debería manejar esto)
        if sampling_rate != self.target_sampling_rate:
            audio_array = librosa.resample(
                audio_array, 
                orig_sr=sampling_rate, 
                target_sr=self.target_sampling_rate
            )
        
        # Truncar o hacer padding si es necesario
        max_samples = int(self.max_duration * self.target_sampling_rate)
        if len(audio_array) > max_samples:
            audio_array = audio_array[:max_samples]
        elif len(audio_array) < max_samples:
            # Hacer padding con zeros
            padding = max_samples - len(audio_array)
            audio_array = np.pad(audio_array, (0, padding), mode='constant')
        
        return audio_array
    
    def create_pronunciation_analysis_text(self, example: Dict) -> str:
        """
        Crear texto de análisis de pronunciación basado en las anotaciones del dataset
        """
        text = example.get("text", "")
        accuracy = example.get("accuracy", 0.0)
        fluency = example.get("fluency", 0.0)
        completeness = example.get("completeness", 0.0)
        prosodic = example.get("prosodic", 0.0)
        total = example.get("total", 0.0)
        words = example.get("words", [])
        
        # Crear análisis detallado
        analysis_parts = [
            f"TRANSCRIPCIÓN: {text}",
            f"PUNTUACIÓN GENERAL: {total}/10",
            f"PRECISIÓN: {accuracy}/10",
            f"FLUIDEZ: {fluency}/10", 
            f"COMPLETITUD: {completeness:.1f}/1.0",
            f"PROSODIA: {prosodic}/10"
        ]
        
        # Analizar errores por palabra
        word_errors = []
        phoneme_errors = []
        
        for word_info in words:
            word_text = word_info["text"]
            word_accuracy = word_info["accuracy"]
            word_total = word_info["total"]
            phones_accuracy = word_info.get("phones-accuracy", [])
            mispronunciations = word_info.get("mispronunciations", [])
            
            if word_accuracy < 8:  # Palabras con errores significativos
                word_errors.append({
                    "word": word_text,
                    "accuracy": word_accuracy,
                    "total": word_total
                })
            
            # Analizar errores de fonemas
            if mispronunciations:
                for mispron in mispronunciations:
                    phoneme_errors.append({
                        "canonical": mispron["canonical-phone"],
                        "pronounced": mispron["pronounced-phone"],
                        "word": word_text,
                        "index": mispron["index"]
                    })
        
        # Agregar análisis de errores
        if word_errors:
            analysis_parts.append("\nERRORES DE PALABRAS:")
            for error in word_errors:
                analysis_parts.append(
                    f"- {error['word']}: {error['accuracy']}/10 (Necesita práctica)"
                )
        
        if phoneme_errors:
            analysis_parts.append("\nERRORES DE FONEMAS:")
            for error in phoneme_errors:
                analysis_parts.append(
                    f"- En '{error['word']}': '{error['canonical']}' pronunciado como '{error['pronounced']}'"
                )
        
        # Recomendaciones
        analysis_parts.append("\nRECOMENDACIONES:")
        if accuracy < 6:
            analysis_parts.append("- Enfócate en la pronunciación correcta de fonemas individuales")
        if fluency < 6:
            analysis_parts.append("- Practica la fluidez evitando pausas largas y repeticiones")
        if completeness < 0.8:
            analysis_parts.append("- Asegúrate de pronunciar todas las palabras claramente")
        if prosodic < 6:
            analysis_parts.append("- Trabaja en la entonación y el ritmo natural del inglés")
        
        return "\n".join(analysis_parts)
    
    def prepare_training_data(self, example: Dict) -> Dict:
        """Preparar un ejemplo para el entrenamiento"""
        # Preprocesar audio
        audio_array = self.preprocess_audio(example["audio"])
        
        # Procesar audio con el processor de Whisper
        inputs = self.processor(
            audio_array,
            sampling_rate=self.target_sampling_rate,
            return_tensors="pt"
        )
        
        # Crear texto objetivo con análisis de pronunciación
        target_text = self.create_pronunciation_analysis_text(example)
        
        # Tokenizar texto objetivo - solo extraer input_ids para labels
        labels_output = self.tokenizer(
            target_text,
            max_length=448,  # Máximo para Whisper
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Asegurarse de que los tensores tienen la forma correcta
        input_features = inputs.input_features.squeeze(0)
        labels = labels_output.input_ids.squeeze(0)

        result = {
            "input_features": input_features,
            "labels": labels,
            "original_text": example.get("text", ""),
            "target_analysis": target_text,
            "scores": {
                "accuracy": example.get("accuracy", 0.0),
                "fluency": example.get("fluency", 0.0),
                "completeness": example.get("completeness", 0.0),
                "prosodic": example.get("prosodic", 0.0),
                "total": example.get("total", 0.0)
            }
        }
        
        return result
    
    def process_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """Procesar todo el dataset para entrenamiento"""
        logger.info("Procesando dataset para entrenamiento...")
        
        processed_dataset = {}
        for split_name, split_data in dataset.items():
            logger.info(f"Procesando split: {split_name}")
            
            # Aplicar preprocesamiento
            processed_split = split_data.map(
                self.prepare_training_data,
                remove_columns=split_data.column_names,  # Remover columnas originales
                desc=f"Procesando {split_name}",
                load_from_cache_file=False
            )
            
            processed_dataset[split_name] = processed_split
        
        logger.info("Procesamiento completado")
        return DatasetDict(processed_dataset)
    
    def get_sample_analysis(self, dataset: Dataset, index: int = 0) -> Dict:
        """Obtener análisis de muestra para verificación"""
        example = dataset[index]
        
        # Manejar casos donde los datos podrían ser listas o tensors
        input_features = example["input_features"]
        labels = example["labels"]
        
        # Convertir a numpy arrays si son listas para obtener shape
        if isinstance(input_features, list):
            input_features = np.array(input_features)
        if isinstance(labels, list):
            labels = np.array(labels)
        
        return {
            "original_text": example.get("original_text", "N/A"),
            "target_analysis": example.get("target_analysis", "N/A"),
            "scores": example.get("scores", {}),
            "input_shape": input_features.shape if hasattr(input_features, 'shape') else f"List length: {len(input_features)}",
            "labels_shape": labels.shape if hasattr(labels, 'shape') else f"List length: {len(labels)}"
        }


def inspect_dataset_sample(dataset: DatasetDict):
    """Inspeccionar una muestra del dataset para debug"""
    train_split = dataset["train"]
    sample = train_split[0]
    
    print("\n=== INSPECCIÓN DEL DATASET ===")
    print(f"Campos disponibles: {list(sample.keys())}")
    print(f"Tipo de audio: {type(sample['audio'])}")
    print(f"Contenido de audio: {sample['audio']}")
    
    for field, value in sample.items():
        if field != 'audio':  # Skip audio para no imprimir datos binarios
            print(f"{field}: {value} (tipo: {type(value)})")
    print("=" * 40)


def main():
    """Función principal para probar el data loader"""
    # Crear data loader
    data_loader = SpeechOceanDataLoader(
        model_name="openai/whisper-small",
        cache_dir="./cache"
    )
    
    # Cargar dataset
    dataset = data_loader.load_dataset()
    
    # Inspeccionar dataset antes de procesar
    inspect_dataset_sample(dataset)
    
    # Procesar una muestra pequeña para prueba
    test_dataset = dataset["test"].select(range(5))
    processed_test = test_dataset.map(
        data_loader.prepare_training_data,
        remove_columns=test_dataset.column_names
    )
    
    # Mostrar muestra
    sample = data_loader.get_sample_analysis(processed_test, 0)
    print("\n=== MUESTRA DE ANÁLISIS ===")
    print(f"Texto original: {sample['original_text']}")
    print(f"Análisis objetivo: {sample['target_analysis']}")
    print(f"Scores: {sample['scores']}")
    print(f"Shape input: {sample['input_shape']}")
    print(f"Shape labels: {sample['labels_shape']}")


if __name__ == "__main__":
    main() 