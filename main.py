"""
Sistema de Fine-tuning de Whisper para Análisis de Pronunciación en Inglés

Este script principal proporciona una interfaz para entrenar y usar el modelo
de análisis de pronunciación basado en Whisper y el dataset speechocean762.
"""

import os
import argparse
import logging
import sys
from typing import Dict, List, Optional
import json

import torch
from dataclasses import asdict

from whisper_pronunciation_trainer import WhisperPronunciationTrainer, TrainingConfig
from pronunciation_analyzer import PronunciationAnalyzer
from data_loader import SpeechOceanDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Crear directorios necesarios"""
    directories = [
        "./cache",
        "./whisper-pronunciation-tuned",
        "./logs",
        "./analysis_results"
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Directorio creado/verificado: {dir_path}")


def train_model(args):
    """Entrenar el modelo de análisis de pronunciación"""
    logger.info("🚀 Iniciando entrenamiento del modelo de análisis de pronunciación")
    
    # Configurar parámetros de entrenamiento
    config = TrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        cache_dir=args.cache_dir,
        run_name=f"whisper-pronunciation-{args.model_name.split('/')[-1]}",
        fp16=args.fp16,
        seed=args.seed
    )
    
    logger.info("📋 Configuración de entrenamiento:")
    config_dict = asdict(config)
    for key, value in config_dict.items():
        if not isinstance(value, list):
            logger.info(f"   {key}: {value}")
    
    try:
        # Crear trainer
        trainer = WhisperPronunciationTrainer(config)
        
        # Entrenar modelo
        train_results, eval_results = trainer.train()
        
        # Guardar modelo y configuración
        trainer.save_model(config.output_dir)
        
        # Guardar métricas finales
        metrics_file = os.path.join(config.output_dir, "final_metrics.json")
        final_metrics = {
            "train_loss": train_results.metrics.get("train_loss"),
            "eval_loss": eval_results.get("eval_loss"),
            "eval_wer": eval_results.get("eval_wer"),
            "eval_bleu": eval_results.get("eval_bleu"),
            "model_name": config.model_name,
            "epochs": config.num_train_epochs
        }
        
        with open(metrics_file, "w") as f:
            json.dump(final_metrics, f, indent=2)
        
        logger.info("✅ Entrenamiento completado exitosamente!")
        logger.info(f"📊 Métricas finales:")
        for metric, value in final_metrics.items():
            if value is not None and isinstance(value, (int, float)):
                logger.info(f"   {metric}: {value:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        return False


def analyze_audio(args):
    """Analizar pronunciación de archivo(s) de audio"""
    logger.info("🎵 Iniciando análisis de pronunciación")
    
    # Verificar que el modelo existe
    if not os.path.exists(args.model_path):
        logger.error(f"❌ Modelo no encontrado en: {args.model_path}")
        logger.info("   Primero entrena el modelo usando: python main.py train")
        return False
    
    try:
        # Crear analizador
        analyzer = PronunciationAnalyzer(
            model_path=args.model_path,
            base_model=args.base_model
        )
        
        if args.audio_file:
            # Analizar archivo único
            logger.info(f"📂 Analizando archivo: {args.audio_file}")
            
            if not os.path.exists(args.audio_file):
                logger.error(f"❌ Archivo de audio no encontrado: {args.audio_file}")
                return False
            
            # Realizar análisis
            result = analyzer.analyze_pronunciation(audio_path=args.audio_file)
            
            # Crear y mostrar reporte
            report = analyzer.create_report(result)
            print("\n" + report)
            
            # Guardar reporte si se especifica directorio
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                output_file = os.path.join(
                    args.output_dir,
                    f"analysis_{os.path.basename(args.audio_file)}.txt"
                )
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(report)
                logger.info(f"💾 Reporte guardado en: {output_file}")
            
        elif args.audio_dir:
            # Analizar directorio de archivos
            logger.info(f"📂 Analizando directorio: {args.audio_dir}")
            
            if not os.path.exists(args.audio_dir):
                logger.error(f"❌ Directorio no encontrado: {args.audio_dir}")
                return False
            
            # Encontrar archivos de audio
            audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
            audio_files = []
            
            for file in os.listdir(args.audio_dir):
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_files.append(os.path.join(args.audio_dir, file))
            
            if not audio_files:
                logger.error(f"❌ No se encontraron archivos de audio en: {args.audio_dir}")
                return False
            
            logger.info(f"🎵 Encontrados {len(audio_files)} archivos de audio")
            
            # Analizar en lote
            output_dir = args.output_dir or "./analysis_results"
            results = analyzer.batch_analyze(audio_files, output_dir)
            
            logger.info(f"✅ Análisis completado. {len(results)} archivos procesados.")
            logger.info(f"📁 Resultados guardados en: {output_dir}")
        
        else:
            logger.error("❌ Debes especificar --audio-file o --audio-dir")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error durante el análisis: {e}")
        return False


def test_data_loader(args):
    """Probar el cargador de datos"""
    logger.info("🧪 Probando cargador de datos speechocean762")
    
    try:
        # Crear data loader
        data_loader = SpeechOceanDataLoader(
            model_name=args.model_name,
            cache_dir=args.cache_dir
        )
        
        # Cargar dataset
        dataset = data_loader.load_dataset()
        
        # Procesar una muestra pequeña
        test_size = min(5, len(dataset["test"]))
        test_dataset = dataset["test"].select(range(test_size))
        processed_test = test_dataset.map(
            data_loader.prepare_training_data,
            remove_columns=test_dataset.column_names
        )
        
        # Mostrar información de muestra
        sample = data_loader.get_sample_analysis(processed_test, 0)
        
        print("\n" + "="*50)
        print("PRUEBA DEL CARGADOR DE DATOS")
        print("="*50)
        print(f"Texto original: {sample['original_text']}")
        print(f"\nAnálisis objetivo:\n{sample['target_analysis']}")
        print(f"\nScores: {sample['scores']}")
        print(f"Shape input: {sample['input_shape']}")
        print(f"Shape labels: {sample['labels_shape']}")
        print("="*50)
        
        logger.info("✅ Prueba del cargador de datos completada exitosamente")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error probando cargador de datos: {e}")
        return False


def show_model_info(args):
    """Mostrar información del modelo entrenado"""
    if not os.path.exists(args.model_path):
        logger.error(f"❌ Modelo no encontrado en: {args.model_path}")
        return False
    
    try:
        # Cargar configuración
        config_file = os.path.join(args.model_path, "training_config.json")
        metrics_file = os.path.join(args.model_path, "final_metrics.json")
        
        print("\n" + "="*60)
        print("INFORMACIÓN DEL MODELO")
        print("="*60)
        
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
            
            print("📋 Configuración del modelo:")
            for key, value in config.items():
                print(f"   {key}: {value}")
        
        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            
            print("\n📊 Métricas finales:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and value is not None:
                    print(f"   {key}: {value:.4f}")
                elif value is not None:
                    print(f"   {key}: {value}")
        
        print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error mostrando información del modelo: {e}")
        return False


def main():
    """Función principal con interfaz de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Sistema de Fine-tuning de Whisper para Análisis de Pronunciación",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Entrenar modelo
  python main.py train --epochs 5 --batch-size 2

  # Analizar un archivo de audio
  python main.py analyze --audio-file mi_audio.wav

  # Analizar directorio de audios
  python main.py analyze --audio-dir ./audios --output-dir ./resultados

  # Probar cargador de datos
  python main.py test-data

  # Ver información del modelo
  python main.py info
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")
    
    # Comando de entrenamiento
    train_parser = subparsers.add_parser("train", help="Entrenar modelo")
    train_parser.add_argument("--model-name", default="openai/whisper-small",
                             help="Modelo base de Whisper a usar")
    train_parser.add_argument("--output-dir", default="./whisper-pronunciation-tuned",
                             help="Directorio de salida del modelo")
    train_parser.add_argument("--epochs", type=int, default=5,
                             help="Número de épocas de entrenamiento")
    train_parser.add_argument("--batch-size", type=int, default=2,
                             help="Tamaño de lote para entrenamiento")
    train_parser.add_argument("--eval-batch-size", type=int, default=4,
                             help="Tamaño de lote para evaluación")
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=8,
                             help="Pasos de acumulación de gradiente")
    train_parser.add_argument("--learning-rate", type=float, default=1e-4,
                             help="Tasa de aprendizaje")
    train_parser.add_argument("--warmup-steps", type=int, default=500,
                             help="Pasos de warmup")
    train_parser.add_argument("--lora-r", type=int, default=32,
                             help="Rank de LoRA")
    train_parser.add_argument("--lora-alpha", type=int, default=64,
                             help="Alpha de LoRA")
    train_parser.add_argument("--lora-dropout", type=float, default=0.1,
                             help="Dropout de LoRA")
    train_parser.add_argument("--cache-dir", default="./cache",
                             help="Directorio de caché")
    train_parser.add_argument("--fp16", action="store_true", default=True,
                             help="Usar precisión mixta FP16")
    train_parser.add_argument("--seed", type=int, default=42,
                             help="Semilla aleatoria")
    
    # Comando de análisis
    analyze_parser = subparsers.add_parser("analyze", help="Analizar pronunciación")
    analyze_parser.add_argument("--model-path", default="./whisper-pronunciation-tuned",
                               help="Ruta del modelo entrenado")
    analyze_parser.add_argument("--base-model", default="openai/whisper-small",
                               help="Modelo base de Whisper")
    analyze_parser.add_argument("--audio-file", help="Archivo de audio a analizar")
    analyze_parser.add_argument("--audio-dir", help="Directorio con archivos de audio")
    analyze_parser.add_argument("--output-dir", help="Directorio para guardar resultados")
    
    # Comando de prueba de datos
    test_parser = subparsers.add_parser("test-data", help="Probar cargador de datos")
    test_parser.add_argument("--model-name", default="openai/whisper-small",
                            help="Modelo de Whisper para procesar")
    test_parser.add_argument("--cache-dir", default="./cache",
                            help="Directorio de caché")
    
    # Comando de información
    info_parser = subparsers.add_parser("info", help="Mostrar información del modelo")
    info_parser.add_argument("--model-path", default="./whisper-pronunciation-tuned",
                            help="Ruta del modelo entrenado")
    
    args = parser.parse_args()
    
    # Configurar directorios
    setup_directories()
    
    # Ejecutar comando
    if args.command == "train":
        success = train_model(args)
    elif args.command == "analyze":
        success = analyze_audio(args)
    elif args.command == "test-data":
        success = test_data_loader(args)
    elif args.command == "info":
        success = show_model_info(args)
    else:
        parser.print_help()
        success = False
    
    if success:
        logger.info("🎉 Operación completada exitosamente!")
        sys.exit(0)
    else:
        logger.error("💥 Operación falló")
        sys.exit(1)


if __name__ == "__main__":
    main()
