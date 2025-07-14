"""
Analizador de pronunciación usando Whisper fine-tuneado para generar 
análisis detallado de errores de pronunciación en inglés.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import re

import torch
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperTokenizer, WhisperForConditionalGeneration
from peft import PeftModel
import matplotlib.pyplot as plt
import seaborn as sns

from whisper_pronunciation_trainer import TrainingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PronunciationScore:
    """Clase para almacenar puntuaciones de pronunciación"""
    overall: float
    accuracy: float
    fluency: float
    completeness: float
    prosodic: float


@dataclass
class WordError:
    """Clase para errores de palabra"""
    word: str
    accuracy_score: float
    error_type: str
    suggestion: str


@dataclass
class PhonemeError:
    """Clase para errores de fonema"""
    canonical_phone: str
    pronounced_phone: str
    word: str
    position: int
    severity: str


@dataclass
class PronunciationAnalysisResult:
    """Resultado completo del análisis de pronunciación"""
    transcript: str
    generated_analysis: str
    scores: PronunciationScore
    word_errors: List[WordError]
    phoneme_errors: List[PhonemeError]
    recommendations: List[str]
    confidence: float


class PronunciationAnalyzer:
    """
    Analizador de pronunciación que usa Whisper fine-tuneado 
    para generar análisis detallado de errores de pronunciación.
    """
    
    def __init__(
        self,
        model_path: str,
        base_model: str = "openai/whisper-small",
        device: str = "auto"
    ):
        self.model_path = model_path
        self.base_model = base_model
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Inicializando analizador en dispositivo: {self.device}")
        
        # Cargar modelo y componentes
        self.load_model()
        
        logger.info("Analizador de pronunciación inicializado exitosamente")
    
    def load_model(self):
        """Cargar modelo fine-tuneado y componentes"""
        logger.info(f"Cargando modelo desde: {self.model_path}")
        
        # Cargar configuración guardada
        config_path = os.path.join(self.model_path, "training_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                saved_config = json.load(f)
                self.base_model = saved_config.get("model_name", self.base_model)
        
        # Cargar modelo base
        self.base_model_instance = WhisperForConditionalGeneration.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map={"": self.device}
        )
        
        # Cargar adaptador LoRA
        self.model = PeftModel.from_pretrained(
            self.base_model_instance,
            self.model_path
        )
        self.model.eval()
        
        # Cargar processor y tokenizer
        self.processor = WhisperProcessor.from_pretrained(self.base_model)
        self.tokenizer = WhisperTokenizer.from_pretrained(self.base_model)
        
        logger.info("Modelo cargado exitosamente")
    
    def preprocess_audio(
        self,
        audio_path: str = None,
        audio_array: np.ndarray = None,
        sampling_rate: int = 16000,
        max_duration: float = 30.0
    ) -> np.ndarray:
        """Preprocesar audio para el modelo"""
        
        if audio_path is not None:
            # Cargar desde archivo
            audio_array, sr = librosa.load(audio_path, sr=sampling_rate)
        elif audio_array is not None:
            # Usar array proporcionado
            sr = sampling_rate
        else:
            raise ValueError("Se debe proporcionar audio_path o audio_array")
        
        # Resamplear si es necesario
        if sr != sampling_rate:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=sampling_rate)
        
        # Truncar si es muy largo
        max_samples = int(max_duration * sampling_rate)
        if len(audio_array) > max_samples:
            audio_array = audio_array[:max_samples]
        
        return audio_array
    
    def generate_analysis(
        self,
        audio_path: str = None,
        audio_array: np.ndarray = None,
        sampling_rate: int = 16000,
        max_new_tokens: int = 400,
        temperature: float = 0.1,
        do_sample: bool = True
    ) -> str:
        """Generar análisis de pronunciación usando el modelo fine-tuneado"""
        
        # Preprocesar audio
        audio = self.preprocess_audio(audio_path, audio_array, sampling_rate)
        
        # Procesar con el processor de Whisper
        inputs = self.processor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )
        
        # Mover a dispositivo
        input_features = inputs.input_features.to(self.device)
        
        # Generar análisis
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_features,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decodificar resultado
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return generated_text
    
    def parse_analysis(self, analysis_text: str) -> PronunciationAnalysisResult:
        """Parsear el texto de análisis generado para extraer información estructurada"""
        
        # Inicializar resultado
        result = PronunciationAnalysisResult(
            transcript="",
            generated_analysis=analysis_text,
            scores=PronunciationScore(0, 0, 0, 0, 0),
            word_errors=[],
            phoneme_errors=[],
            recommendations=[],
            confidence=0.8  # Default confidence
        )
        
        try:
            # Extraer transcripción
            transcript_match = re.search(r"TRANSCRIPCIÓN:\s*(.+)", analysis_text)
            if transcript_match:
                result.transcript = transcript_match.group(1).strip()
            
            # Extraer puntuaciones
            scores_patterns = {
                "overall": r"PUNTUACIÓN GENERAL:\s*(\d+(?:\.\d+)?)",
                "accuracy": r"PRECISIÓN:\s*(\d+(?:\.\d+)?)",
                "fluency": r"FLUIDEZ:\s*(\d+(?:\.\d+)?)",
                "completeness": r"COMPLETITUD:\s*(\d+(?:\.\d+)?)",
                "prosodic": r"PROSODIA:\s*(\d+(?:\.\d+)?)"
            }
            
            for score_name, pattern in scores_patterns.items():
                match = re.search(pattern, analysis_text)
                if match:
                    score_value = float(match.group(1))
                    setattr(result.scores, score_name, score_value)
            
            # Extraer errores de palabras
            word_errors_section = re.search(
                r"ERRORES DE PALABRAS:(.*?)(?=\n[A-Z]|\nRECOMENDACIONES:|$)",
                analysis_text,
                re.DOTALL
            )
            if word_errors_section:
                word_error_lines = word_errors_section.group(1).strip().split('\n')
                for line in word_error_lines:
                    line = line.strip()
                    if line.startswith('-'):
                        # Parsear línea de error: "- WORD: score/10 (descripción)"
                        word_match = re.search(r"-\s*(\w+):\s*(\d+(?:\.\d+)?)/10\s*\((.+)\)", line)
                        if word_match:
                            word, accuracy, description = word_match.groups()
                            result.word_errors.append(WordError(
                                word=word,
                                accuracy_score=float(accuracy),
                                error_type="accuracy",
                                suggestion=description
                            ))
            
            # Extraer errores de fonemas
            phoneme_errors_section = re.search(
                r"ERRORES DE FONEMAS:(.*?)(?=\n[A-Z]|\nRECOMENDACIONES:|$)",
                analysis_text,
                re.DOTALL
            )
            if phoneme_errors_section:
                phoneme_error_lines = phoneme_errors_section.group(1).strip().split('\n')
                for line in phoneme_error_lines:
                    line = line.strip()
                    if line.startswith('-'):
                        # Parsear línea: "- En 'WORD': 'CANONICAL' pronunciado como 'PRONOUNCED'"
                        phoneme_match = re.search(
                            r"-\s*En\s+'(\w+)':\s+'([^']+)'\s+pronunciado\s+como\s+'([^']+)'",
                            line
                        )
                        if phoneme_match:
                            word, canonical, pronounced = phoneme_match.groups()
                            result.phoneme_errors.append(PhonemeError(
                                canonical_phone=canonical,
                                pronounced_phone=pronounced,
                                word=word,
                                position=0,  # No tenemos posición específica
                                severity="medium"
                            ))
            
            # Extraer recomendaciones
            recommendations_section = re.search(
                r"RECOMENDACIONES:(.*?)$",
                analysis_text,
                re.DOTALL
            )
            if recommendations_section:
                rec_lines = recommendations_section.group(1).strip().split('\n')
                for line in rec_lines:
                    line = line.strip()
                    if line.startswith('-'):
                        result.recommendations.append(line[1:].strip())
        
        except Exception as e:
            logger.warning(f"Error parseando análisis: {e}")
        
        return result
    
    def analyze_pronunciation(
        self,
        audio_path: str = None,
        audio_array: np.ndarray = None,
        sampling_rate: int = 16000,
        return_raw: bool = False
    ) -> Union[str, PronunciationAnalysisResult]:
        """
        Análisis completo de pronunciación
        
        Args:
            audio_path: Ruta al archivo de audio
            audio_array: Array de audio NumPy
            sampling_rate: Frecuencia de muestreo
            return_raw: Si True, retorna solo el texto generado
            
        Returns:
            Análisis estructurado o texto raw
        """
        
        # Generar análisis
        analysis_text = self.generate_analysis(
            audio_path=audio_path,
            audio_array=audio_array,
            sampling_rate=sampling_rate
        )
        
        if return_raw:
            return analysis_text
        
        # Parsear y estructurar resultado
        result = self.parse_analysis(analysis_text)
        
        return result
    
    def create_report(self, result: PronunciationAnalysisResult) -> str:
        """Crear reporte formateado del análisis"""
        
        report_lines = [
            "=" * 60,
            "REPORTE DE ANÁLISIS DE PRONUNCIACIÓN",
            "=" * 60,
            "",
            f"📝 TRANSCRIPCIÓN: {result.transcript}",
            "",
            "📊 PUNTUACIONES:",
            f"   • General: {result.scores.overall}/10",
            f"   • Precisión: {result.scores.accuracy}/10",
            f"   • Fluidez: {result.scores.fluency}/10",
            f"   • Completitud: {result.scores.completeness}/1.0",
            f"   • Prosodia: {result.scores.prosodic}/10",
            ""
        ]
        
        if result.word_errors:
            report_lines.extend([
                "⚠️  ERRORES DE PALABRAS:",
                ""
            ])
            for error in result.word_errors:
                report_lines.append(
                    f"   • {error.word}: {error.accuracy_score}/10 - {error.suggestion}"
                )
            report_lines.append("")
        
        if result.phoneme_errors:
            report_lines.extend([
                "🔊 ERRORES DE FONEMAS:",
                ""
            ])
            for error in result.phoneme_errors:
                report_lines.append(
                    f"   • En '{error.word}': '{error.canonical_phone}' → '{error.pronounced_phone}'"
                )
            report_lines.append("")
        
        if result.recommendations:
            report_lines.extend([
                "💡 RECOMENDACIONES:",
                ""
            ])
            for rec in result.recommendations:
                report_lines.append(f"   • {rec}")
            report_lines.append("")
        
        report_lines.extend([
            f"🎯 Confianza del análisis: {result.confidence:.1%}",
            "=" * 60
        ])
        
        return "\n".join(report_lines)
    
    def batch_analyze(
        self,
        audio_files: List[str],
        output_dir: str = "./analysis_results"
    ) -> List[PronunciationAnalysisResult]:
        """Analizar múltiples archivos de audio"""
        
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        for i, audio_file in enumerate(audio_files):
            logger.info(f"Analizando archivo {i+1}/{len(audio_files)}: {audio_file}")
            
            try:
                # Analizar
                result = self.analyze_pronunciation(audio_path=audio_file)
                results.append(result)
                
                # Guardar reporte individual
                report = self.create_report(result)
                output_file = os.path.join(
                    output_dir,
                    f"analysis_{i+1}_{os.path.basename(audio_file)}.txt"
                )
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(report)
                
            except Exception as e:
                logger.error(f"Error analizando {audio_file}: {e}")
        
        return results


def main():
    """Función principal para prueba"""
    # Configurar rutas
    model_path = "./whisper-pronunciation-tuned"
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado en: {model_path}")
        print("   Por favor, entrena el modelo primero ejecutando whisper_pronunciation_trainer.py")
        return
    
    # Crear analizador
    analyzer = PronunciationAnalyzer(
        model_path=model_path,
        base_model="openai/whisper-small"
    )
    
    print("✅ Analizador de pronunciación listo!")
    print("\nPara usar el analizador:")
    print("1. analyzer.analyze_pronunciation(audio_path='mi_audio.wav')")
    print("2. analyzer.create_report(result)")
    
    # Ejemplo con audio de prueba (si existe)
    test_audio = "test_audio.wav"
    if os.path.exists(test_audio):
        print(f"\n🎵 Analizando audio de prueba: {test_audio}")
        
        result = analyzer.analyze_pronunciation(audio_path=test_audio)
        report = analyzer.create_report(result)
        
        print("\n" + report)
    else:
        print(f"\n💡 Para probar, coloca un archivo de audio como '{test_audio}' en el directorio actual")


if __name__ == "__main__":
    main() 