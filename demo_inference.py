"""
Script de demostración para probar el análisis de pronunciación
con el modelo Whisper fine-tuneado usando muestras reales del dataset.
"""

import os
import sys
import logging
import argparse
from typing import List, Dict
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa

from pronunciation_analyzer import PronunciationAnalyzer, PronunciationAnalysisResult
from data_loader import SpeechOceanDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importación opcional de IPython para notebooks
try:
    from IPython.display import Audio, display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False


class PronunciationDemo:
    """
    Clase de demostración para mostrar las capacidades del analizador
    de pronunciación con ejemplos reales del dataset speechocean762.
    """
    
    def __init__(self, model_path: str, base_model: str = "openai/whisper-small"):
        self.model_path = model_path
        self.base_model = base_model
        
        # Verificar que el modelo existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        
        # Inicializar analizador
        logger.info("Cargando analizador de pronunciación...")
        self.analyzer = PronunciationAnalyzer(
            model_path=model_path,
            base_model=base_model
        )
        logger.info("✅ Analizador cargado exitosamente")
        
        # Inicializar data loader para acceder al dataset
        logger.info("Cargando dataset speechocean762...")
        self.data_loader = SpeechOceanDataLoader(
            model_name=base_model,
            cache_dir="./cache"
        )
        self.dataset = None
    
    def load_real_samples(self, num_samples: int = 3) -> List[Dict]:
        """Cargar muestras reales del dataset speechocean762"""
        try:
            if self.dataset is None:
                self.dataset = self.data_loader.load_dataset()
            
            # Usar muestras del conjunto de test
            test_dataset = self.dataset["test"]
            
            # Seleccionar muestras variadas (diferentes scores)
            samples = []
            indices = [0, len(test_dataset)//4, len(test_dataset)//2]  # Muestras distribuidas
            
            for i, idx in enumerate(indices[:num_samples]):
                if idx < len(test_dataset):
                    sample = test_dataset[idx]
                    
                    # Procesar audio
                    try:
                        audio_array = self.data_loader.preprocess_audio(sample["audio"])
                        
                        sample_info = {
                            "id": idx,
                            "text": sample["text"],
                            "audio_array": audio_array,
                            "real_scores": {
                                "accuracy": sample.get("accuracy", 0.0),
                                "fluency": sample.get("fluency", 0.0),
                                "completeness": sample.get("completeness", 0.0),
                                "prosodic": sample.get("prosodic", 0.0),
                                "total": sample.get("total", 0.0)
                            },
                            "words": sample.get("words", []),
                            "description": f"Muestra real #{idx+1} del dataset"
                        }
                        samples.append(sample_info)
                        logger.info(f"✅ Cargada muestra {i+1}: '{sample['text'][:50]}...'")
                        
                    except Exception as e:
                        logger.warning(f"Error procesando muestra {idx}: {e}")
                        continue
            
            if not samples:
                logger.warning("No se pudieron cargar muestras del dataset, usando audio sintético")
                return self.get_fallback_samples()
                
            return samples
            
        except Exception as e:
            logger.error(f"Error cargando dataset: {e}")
            logger.info("Usando muestras sintéticas como fallback...")
            return self.get_fallback_samples()
    
    def get_fallback_samples(self) -> List[Dict]:
        """Muestras sintéticas como fallback si no se puede cargar el dataset"""
        logger.info("Generando muestras sintéticas...")
        
        sample_texts = [
            "Hello world",
            "The quick brown fox jumps over the lazy dog",  
            "Pronunciation is very important for communication"
        ]
        
        samples = []
        for i, text in enumerate(sample_texts):
            # Generar audio sintético mejorado
            audio_array = self.generate_speech_like_audio(text)
            
            sample_info = {
                "id": f"synthetic_{i}",
                "text": text,
                "audio_array": audio_array,
                "real_scores": None,  # No hay scores reales para sintético
                "words": [],
                "description": f"Muestra sintética #{i+1}"
            }
            samples.append(sample_info)
        
        return samples
    
    def generate_speech_like_audio(self, text: str, duration: float = None) -> np.ndarray:
        """Generar audio sintético mejorado (solo como fallback)"""
        if duration is None:
            duration = max(1.0, len(text) * 0.08)  # ~0.08s por carácter
        
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Crear audio más realista con formantes
        fundamental_freq = 120 + np.random.uniform(-20, 20)
        audio = np.zeros_like(t)
        
        # Simular palabras
        words = text.split()
        word_duration = duration / len(words)
        
        for i, word in enumerate(words):
            start_time = i * word_duration
            end_time = (i + 1) * word_duration
            
            start_idx = int(start_time * sample_rate)
            end_idx = int(end_time * sample_rate)
            
            if start_idx < len(audio) and end_idx <= len(audio):
                word_t = t[start_idx:end_idx]
                
                # Formantes variando por palabra
                f1 = 500 + np.random.uniform(-100, 200)
                f2 = 1500 + np.random.uniform(-300, 500)
                
                word_audio = np.zeros_like(word_t)
                
                # Múltiples armónicos
                for harmonic in range(1, 5):
                    freq = fundamental_freq * harmonic
                    if freq < sample_rate // 2:
                        amplitude = 1.0 / harmonic
                        word_audio += amplitude * np.sin(2 * np.pi * freq * word_t)
                
                # Agregar formantes
                word_audio += 0.3 * np.sin(2 * np.pi * f1 * word_t)
                word_audio += 0.2 * np.sin(2 * np.pi * f2 * word_t)
                
                # Envolvente natural
                envelope = np.exp(-2 * word_t / word_duration)
                word_audio *= envelope
                
                audio[start_idx:end_idx] = word_audio
        
        # Ruido y filtrado
        noise = np.random.normal(0, 0.05, len(audio))
        audio = audio + noise
        
        # Normalizar
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.7
        
        return audio
    
    def analyze_real_samples(self) -> List[Dict]:
        """Analizar muestras reales del dataset"""
        
        # Cargar muestras reales
        samples = self.load_real_samples()
        
        results = []
        
        for i, sample in enumerate(samples):
            logger.info(f"Analizando muestra {i+1}: '{sample['text'][:50]}{'...' if len(sample['text']) > 50 else ''}'")
            
            try:
                # Analizar pronunciación con el modelo
                analysis = self.analyzer.analyze_pronunciation(
                    audio_array=sample["audio_array"],
                    sampling_rate=16000,
                    expected_text=sample["text"]
                )
                
                result = {
                    "sample_info": sample,
                    "analysis": analysis,
                    "real_scores": sample["real_scores"],  # Scores reales del dataset
                    "audio": sample["audio_array"]
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error analizando muestra {i+1}: {e}")
        
        return results
    
    def create_visualization(self, results: List[Dict]) -> None:
        """Crear visualizaciones de los resultados"""
        
        if not results:
            logger.warning("No hay resultados para visualizar")
            return
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Análisis de Pronunciación - Resultados', fontsize=16, fontweight='bold')
        
        # Extraer datos para visualización
        sample_names = [f"Muestra {i+1}" for i in range(len(results))]
        accuracy_scores = []
        fluency_scores = []
        completeness_scores = []
        prosodic_scores = []
        
        for result in results:
            analysis = result["analysis"]
            accuracy_scores.append(analysis.scores.accuracy)
            fluency_scores.append(analysis.scores.fluency)
            completeness_scores.append(analysis.scores.completeness * 10)  # Escalar a 0-10
            prosodic_scores.append(analysis.scores.prosodic)
        
        # Gráfico 1: Puntuaciones por categoría
        ax1 = axes[0, 0]
        x = np.arange(len(sample_names))
        width = 0.2
        
        ax1.bar(x - 1.5*width, accuracy_scores, width, label='Precisión', alpha=0.8)
        ax1.bar(x - 0.5*width, fluency_scores, width, label='Fluidez', alpha=0.8)
        ax1.bar(x + 0.5*width, completeness_scores, width, label='Completitud', alpha=0.8)
        ax1.bar(x + 1.5*width, prosodic_scores, width, label='Prosodia', alpha=0.8)
        
        ax1.set_xlabel('Muestras')
        ax1.set_ylabel('Puntuación (0-10)')
        ax1.set_title('Puntuaciones por Categoría')
        ax1.set_xticks(x)
        ax1.set_xticklabels(sample_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Puntuación general
        ax2 = axes[0, 1]
        overall_scores = [result["analysis"].scores.overall for result in results]
        bars = ax2.bar(sample_names, overall_scores, color='skyblue', alpha=0.7)
        ax2.set_xlabel('Muestras')
        ax2.set_ylabel('Puntuación General (0-10)')
        ax2.set_title('Puntuación General por Muestra')
        ax2.grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bar, score in zip(bars, overall_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{score:.1f}', ha='center', va='bottom')
        
        # Gráfico 3: Mapa de calor de todas las puntuaciones
        ax3 = axes[1, 0]
        score_matrix = np.array([
            accuracy_scores,
            fluency_scores,
            completeness_scores,
            prosodic_scores
        ])
        
        im = ax3.imshow(score_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=10)
        ax3.set_xticks(range(len(sample_names)))
        ax3.set_xticklabels(sample_names, rotation=45)
        ax3.set_yticks(range(4))
        ax3.set_yticklabels(['Precisión', 'Fluidez', 'Completitud', 'Prosodia'])
        ax3.set_title('Mapa de Calor - Todas las Puntuaciones')
        
        # Añadir valores al mapa de calor
        for i in range(4):
            for j in range(len(sample_names)):
                text = ax3.text(j, i, f'{score_matrix[i, j]:.1f}',
                               ha="center", va="center", color="black", fontweight="bold")
        
        # Añadir barra de colores
        plt.colorbar(im, ax=ax3, label='Puntuación')
        
        # Gráfico 4: Número de errores por muestra
        ax4 = axes[1, 1]
        word_errors_count = [len(result["analysis"].word_errors) for result in results]
        phoneme_errors_count = [len(result["analysis"].phoneme_errors) for result in results]
        
        x = np.arange(len(sample_names))
        width = 0.35
        
        ax4.bar(x - width/2, word_errors_count, width, label='Errores de Palabra', alpha=0.8)
        ax4.bar(x + width/2, phoneme_errors_count, width, label='Errores de Fonema', alpha=0.8)
        
        ax4.set_xlabel('Muestras')
        ax4.set_ylabel('Número de Errores')
        ax4.set_title('Errores Detectados por Muestra')
        ax4.set_xticks(x)
        ax4.set_xticklabels(sample_names, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_detailed_results(self, results: List[Dict]) -> None:
        """Imprimir resultados detallados"""
        
        print("\n" + "="*80)
        print("RESULTADOS DETALLADOS DEL ANÁLISIS DE PRONUNCIACIÓN")
        print("="*80)
        
        for i, result in enumerate(results):
            sample_info = result["sample_info"]
            analysis = result["analysis"]
            
            print(f"\n🎯 MUESTRA {i+1}: {sample_info['description']}")
            print(f"   Texto: '{sample_info['text']}'")
            print("-" * 60)
            
            # Crear reporte
            report = self.analyzer.create_report(analysis)
            print(report)
            
            # Información adicional
            print(f"\n📈 ESTADÍSTICAS ADICIONALES:")
            print(f"   • Número de palabras: {len(sample_info['text'].split())}")
            print(f"   • Errores de palabra detectados: {len(analysis.word_errors)}")
            print(f"   • Errores de fonema detectados: {len(analysis.phoneme_errors)}")
            print(f"   • Recomendaciones generadas: {len(analysis.recommendations)}")
            
            if i < len(results) - 1:
                print("\n" + "~" * 60)
    
    def save_results_summary(self, results: List[Dict], output_file: str = "demo_results.json"):
        """Guardar resumen de resultados en JSON"""
        
        import datetime
        summary = {
            "demo_info": {
                "model_path": self.model_path,
                "base_model": self.base_model,
                "total_samples": len(results),
                "timestamp": str(datetime.datetime.now())
            },
            "results": []
        }
        
        for i, result in enumerate(results):
            sample_summary = {
                "sample_id": i + 1,
                "text": result["sample_info"]["text"],
                "description": result["sample_info"]["description"],
                "scores": {
                    "overall": result["analysis"].scores.overall,
                    "accuracy": result["analysis"].scores.accuracy,
                    "fluency": result["analysis"].scores.fluency,
                    "completeness": result["analysis"].scores.completeness,
                    "prosodic": result["analysis"].scores.prosodic
                },
                "errors": {
                    "word_errors": len(result["analysis"].word_errors),
                    "phoneme_errors": len(result["analysis"].phoneme_errors)
                },
                "recommendations_count": len(result["analysis"].recommendations),
                "confidence": result["analysis"].confidence
            }
            summary["results"].append(sample_summary)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Resumen guardado en: {output_file}")
    
    def run_demo(self, save_results: bool = True, show_visualizations: bool = True):
        """Ejecutar demostración completa"""
        
        print("\n🎤 DEMOSTRACIÓN DE ANÁLISIS DE PRONUNCIACIÓN")
        print("=" * 60)
        print("Este demo muestra las capacidades del modelo Whisper fine-tuneado")
        print("para análisis de pronunciación en inglés usando el dataset speechocean762.")
        print("=" * 60)
        
        # Analizar muestras
        logger.info("Generando y analizando muestras de audio...")
        results = self.analyze_real_samples()
        
        if not results:
            logger.error("❌ No se pudieron generar resultados")
            return
        
        # Mostrar resultados detallados con comparación
        self.display_detailed_results_with_comparison(results)
        
        # Crear visualizaciones de comparación
        if show_visualizations:
            try:
                logger.info("Creando visualizaciones de comparación...")
                self.create_comparison_visualization(results)
            except Exception as e:
                logger.warning(f"No se pudieron crear visualizaciones: {e}")
        
        # Guardar resultados
        if save_results:
            try:
                self.save_results_summary(results)
            except Exception as e:
                logger.warning(f"No se pudo guardar el resumen: {e}")
        
        # Mostrar resumen final con métricas de comparación
        self.show_final_summary_with_comparison(results)

    def display_detailed_results_with_comparison(self, results: List[Dict]) -> None:
        """Mostrar resultados detallados con comparación entre scores reales y predichos"""
        
        print("\n" + "="*90)
        print("RESULTADOS DETALLADOS CON COMPARACIÓN REAL vs PREDICHO")
        print("="*90)
        
        for i, result in enumerate(results):
            sample_info = result["sample_info"]
            analysis = result["analysis"]
            real_scores = result.get("real_scores")
            
            print(f"\n🎯 MUESTRA {i+1}: {sample_info['description']}")
            print(f"   Texto: '{sample_info['text']}'")
            print("-" * 70)
            
            # Mostrar comparación de scores si hay datos reales
            if real_scores:
                print("📊 COMPARACIÓN DE PUNTUACIONES (Real vs Predicho):")
                print(f"   • General:     {real_scores['total']:.1f}/10  →  {analysis.scores.overall:.1f}/10")
                print(f"   • Precisión:   {real_scores['accuracy']:.1f}/10  →  {analysis.scores.accuracy:.1f}/10") 
                print(f"   • Fluidez:     {real_scores['fluency']:.1f}/10  →  {analysis.scores.fluency:.1f}/10")
                print(f"   • Completitud: {real_scores['completeness']:.1f}/1.0  →  {analysis.scores.completeness:.1f}/1.0")
                print(f"   • Prosodia:    {real_scores['prosodic']:.1f}/10  →  {analysis.scores.prosodic:.1f}/10")
                
                # Calcular diferencias
                diff_accuracy = abs(real_scores['accuracy'] - analysis.scores.accuracy)
                diff_fluency = abs(real_scores['fluency'] - analysis.scores.fluency)
                diff_total = abs(real_scores['total'] - analysis.scores.overall)
                
                print(f"\n📈 DIFERENCIAS (Error Absoluto):")
                print(f"   • Precisión: {diff_accuracy:.1f}")
                print(f"   • Fluidez: {diff_fluency:.1f}")
                print(f"   • General: {diff_total:.1f}")
                
            else:
                print("📊 PUNTUACIONES PREDICHAS:")
                print(f"   • General: {analysis.scores.overall:.1f}/10")
                print(f"   • Precisión: {analysis.scores.accuracy:.1f}/10")
                print(f"   • Fluidez: {analysis.scores.fluency:.1f}/10")
                print(f"   • Completitud: {analysis.scores.completeness:.1f}/1.0")
                print(f"   • Prosodia: {analysis.scores.prosodic:.1f}/10")
            
            # Mostrar transcript
            print(f"\n📝 TRANSCRIPCIÓN: {analysis.transcript}")
            
            # Mostrar errores detectados
            if analysis.word_errors:
                print(f"\n⚠️  ERRORES DE PALABRAS DETECTADOS:")
                for error in analysis.word_errors:
                    print(f"   • {error.word}: {error.accuracy_score:.1f}/10 - {error.suggestion}")
            
            if analysis.phoneme_errors:
                print(f"\n🔊 ERRORES DE FONEMAS DETECTADOS:")
                for error in analysis.phoneme_errors:
                    print(f"   • En '{error.word}': /{error.canonical_phone}/ → /{error.pronounced_phone}/")
            
            # Mostrar recomendaciones
            if analysis.recommendations:
                print(f"\n💡 RECOMENDACIONES:")
                for rec in analysis.recommendations:
                    print(f"   {rec}")
            
            print(f"\n🎯 Confianza del análisis: {analysis.confidence:.1%}")
            print("~" * 70)
    
    def create_comparison_visualization(self, results: List[Dict]) -> None:
        """Crear visualizaciones comparando scores reales vs predichos"""
        
        # Filtrar solo resultados con scores reales
        real_results = [r for r in results if r.get("real_scores")]
        
        if not real_results:
            logger.info("No hay scores reales para comparar, usando visualización estándar")
            return self.create_visualization(results)
        
        # Configurar visualización
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comparación: Scores Reales vs Predichos del Modelo', fontsize=16, fontweight='bold')
        
        # Extraer datos para comparación
        sample_names = [f"Muestra {i+1}" for i in range(len(real_results))]
        
        real_accuracy = [r["real_scores"]["accuracy"] for r in real_results]
        pred_accuracy = [r["analysis"].scores.accuracy for r in real_results]
        
        real_fluency = [r["real_scores"]["fluency"] for r in real_results]
        pred_fluency = [r["analysis"].scores.fluency for r in real_results]
        
        real_total = [r["real_scores"]["total"] for r in real_results]
        pred_total = [r["analysis"].scores.overall for r in real_results]
        
        # Gráfico 1: Comparación de Precisión
        ax1 = axes[0, 0]
        x = np.arange(len(sample_names))
        width = 0.35
        
        ax1.bar(x - width/2, real_accuracy, width, label='Real', alpha=0.8, color='skyblue')
        ax1.bar(x + width/2, pred_accuracy, width, label='Predicho', alpha=0.8, color='orange')
        
        ax1.set_xlabel('Muestras')
        ax1.set_ylabel('Precisión')
        ax1.set_title('Precisión: Real vs Predicho')
        ax1.set_xticks(x)
        ax1.set_xticklabels(sample_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Comparación de Fluidez
        ax2 = axes[0, 1]
        ax2.bar(x - width/2, real_fluency, width, label='Real', alpha=0.8, color='lightgreen')
        ax2.bar(x + width/2, pred_fluency, width, label='Predicho', alpha=0.8, color='coral')
        
        ax2.set_xlabel('Muestras')
        ax2.set_ylabel('Fluidez')
        ax2.set_title('Fluidez: Real vs Predicho')
        ax2.set_xticks(x)
        ax2.set_xticklabels(sample_names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Scatter plot Real vs Predicho
        ax3 = axes[1, 0]
        ax3.scatter(real_total, pred_total, alpha=0.7, s=100)
        
        # Línea diagonal perfecta
        min_val = min(min(real_total), min(pred_total))
        max_val = max(max(real_total), max(pred_total))
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Predicción Perfecta')
        
        ax3.set_xlabel('Score Real')
        ax3.set_ylabel('Score Predicho')
        ax3.set_title('Correlación: Real vs Predicho')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Errores absolutos
        ax4 = axes[1, 1]
        errors = [abs(r - p) for r, p in zip(real_total, pred_total)]
        
        bars = ax4.bar(sample_names, errors, alpha=0.8, color='red')
        ax4.set_xlabel('Muestras')
        ax4.set_ylabel('Error Absoluto')
        ax4.set_title('Error Absoluto por Muestra')
        ax4.set_xticklabels(sample_names, rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Añadir valores encima de las barras
        for bar, error in zip(bars, errors):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{error:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def show_final_summary_with_comparison(self, results: List[Dict]) -> None:
        """Mostrar resumen final con métricas de comparación"""
        
        real_results = [r for r in results if r.get("real_scores")]
        
        print("\n" + "="*80)
        print("📊 RESUMEN FINAL DEL ANÁLISIS")
        print("="*80)
        
        if real_results:
            # Calcular métricas de error
            total_errors = []
            accuracy_errors = []
            fluency_errors = []
            
            for result in real_results:
                real = result["real_scores"]
                pred = result["analysis"].scores
                
                total_errors.append(abs(real["total"] - pred.overall))
                accuracy_errors.append(abs(real["accuracy"] - pred.accuracy))
                fluency_errors.append(abs(real["fluency"] - pred.fluency))
            
            print(f"🎯 **Métricas de Evaluación del Modelo:**")
            print(f"   • Muestras con datos reales: {len(real_results)}")
            print(f"   • Error Absoluto Medio (Total): {np.mean(total_errors):.2f}")
            print(f"   • Error Absoluto Medio (Precisión): {np.mean(accuracy_errors):.2f}")
            print(f"   • Error Absoluto Medio (Fluidez): {np.mean(fluency_errors):.2f}")
            print(f"   • Error Máximo: {np.max(total_errors):.2f}")
            print(f"   • Error Mínimo: {np.min(total_errors):.2f}")
        
        print(f"\n📈 **Resumen General:**")
        print(f"   • Total de muestras analizadas: {len(results)}")
        
        avg_confidence = np.mean([r["analysis"].confidence for r in results])
        print(f"   • Confianza promedio: {avg_confidence:.1%}")
        
        total_word_errors = sum(len(r["analysis"].word_errors) for r in results)
        total_phoneme_errors = sum(len(r["analysis"].phoneme_errors) for r in results)
        print(f"   • Total errores de palabras detectados: {total_word_errors}")
        print(f"   • Total errores de fonemas detectados: {total_phoneme_errors}")
        
        if real_results:
            avg_real_score = np.mean([r["real_scores"]["total"] for r in real_results])
            avg_pred_score = np.mean([r["analysis"].scores.overall for r in real_results])
            print(f"   • Score real promedio: {avg_real_score:.2f}/10")
            print(f"   • Score predicho promedio: {avg_pred_score:.2f}/10")
        
        print("\n✅ **Análisis completado con éxito!**")
        
        if real_results:
            print("🎯 **El modelo fue evaluado con datos reales del dataset speechocean762**")
            mae = np.mean(total_errors)
            if mae < 1.0:
                print("🟢 **Rendimiento: EXCELENTE** (Error < 1.0)")
            elif mae < 2.0:
                print("🟡 **Rendimiento: BUENO** (Error < 2.0)")
            else:
                print("🔴 **Rendimiento: NECESITA MEJORA** (Error > 2.0)")
        
        print("="*80)


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Demostración del analizador de pronunciación",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Ejecutar demo completo
  python demo_inference.py

  # Usar modelo personalizado
  python demo_inference.py --model-path ./mi_modelo --base-model openai/whisper-base

  # Solo mostrar resultados, sin visualizaciones
  python demo_inference.py --no-visualizations
        """
    )
    
    parser.add_argument(
        "--model-path",
        default="./whisper-pronunciation-tuned",
        help="Ruta del modelo fine-tuneado"
    )
    
    parser.add_argument(
        "--base-model",
        default="openai/whisper-small",
        help="Modelo base de Whisper usado"
    )
    
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Desactivar visualizaciones"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="No guardar resultados"
    )
    
    args = parser.parse_args()
    
    try:
        # Crear y ejecutar demo
        demo = PronunciationDemo(
            model_path=args.model_path,
            base_model=args.base_model
        )
        
        demo.run_demo(
            save_results=not args.no_save,
            show_visualizations=not args.no_visualizations
        )
        
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
        logger.info("💡 Asegúrate de haber entrenado el modelo primero:")
        logger.info("   python main.py train")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Error ejecutando demo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 