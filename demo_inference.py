"""
Script de demostración para probar el análisis de pronunciación
con el modelo Whisper fine-tuneado.
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
from IPython.display import Audio, display
import librosa

from pronunciation_analyzer import PronunciationAnalyzer, PronunciationAnalysisResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PronunciationDemo:
    """
    Clase de demostración para mostrar las capacidades del analizador
    de pronunciación con ejemplos interactivos.
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
    
    def generate_sample_audio(
        self,
        text: str = "Hello world",
        duration: float = 2.0,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Generar audio de muestra sintético para pruebas
        (En una implementación real, usarías archivos de audio reales)
        """
        # Generar señal sinusoidal simple como placeholder
        t = np.linspace(0, duration, int(sample_rate * duration))
        frequencies = [440, 523, 659, 784]  # Notas musicales
        
        audio = np.zeros_like(t)
        for i, freq in enumerate(frequencies):
            segment_start = i * len(t) // len(frequencies)
            segment_end = (i + 1) * len(t) // len(frequencies)
            audio[segment_start:segment_end] = np.sin(2 * np.pi * freq * t[segment_start:segment_end])
        
        # Añadir algo de ruido para hacerlo más realista
        noise = np.random.normal(0, 0.1, len(audio))
        audio = audio * 0.8 + noise * 0.2
        
        # Normalizar
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio
    
    def analyze_sample_texts(self) -> List[Dict]:
        """Analizar textos de muestra con audio sintético"""
        
        sample_texts = [
            {
                "text": "Hello world",
                "description": "Saludo básico en inglés",
                "audio_file": None  # Se generará sintéticamente
            },
            {
                "text": "The quick brown fox jumps over the lazy dog",
                "description": "Pangrama clásico en inglés",
                "audio_file": None
            },
            {
                "text": "Pronunciation is very important for communication",
                "description": "Frase sobre pronunciación",
                "audio_file": None
            }
        ]
        
        results = []
        
        for i, sample in enumerate(sample_texts):
            logger.info(f"Analizando muestra {i+1}: '{sample['text']}'")
            
            try:
                # Generar audio sintético
                audio_array = self.generate_sample_audio(
                    text=sample["text"],
                    duration=len(sample["text"]) * 0.1  # Duración basada en longitud del texto
                )
                
                # Analizar pronunciación
                analysis = self.analyzer.analyze_pronunciation(
                    audio_array=audio_array,
                    sampling_rate=16000
                )
                
                result = {
                    "sample_info": sample,
                    "analysis": analysis,
                    "audio": audio_array
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
        results = self.analyze_sample_texts()
        
        if not results:
            logger.error("❌ No se pudieron generar resultados")
            return
        
        # Mostrar resultados detallados
        self.print_detailed_results(results)
        
        # Crear visualizaciones
        if show_visualizations:
            try:
                logger.info("Creando visualizaciones...")
                self.create_visualization(results)
            except Exception as e:
                logger.warning(f"No se pudieron crear visualizaciones: {e}")
        
        # Guardar resultados
        if save_results:
            try:
                self.save_results_summary(results)
            except Exception as e:
                logger.warning(f"No se pudo guardar el resumen: {e}")
        
        print(f"\n✅ Demo completada exitosamente!")
        print(f"   📊 Muestras analizadas: {len(results)}")
        print(f"   🎯 Puntuación promedio: {np.mean([r['analysis'].scores.overall for r in results]):.2f}/10")


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