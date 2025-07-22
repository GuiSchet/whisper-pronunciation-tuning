import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import gradio as gr
import os
import re
from datasets import load_dataset
from dotenv import load_dotenv
import warnings

# Suprimir warnings no críticos
warnings.filterwarnings("ignore")

# Ruta al modelo entrenado
MODEL_DIR = "./whisper-pronunciation-assessment-local"

def load_model():
    try:
        processor = WhisperProcessor.from_pretrained(MODEL_DIR)
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        # Configurar el modelo para inferencia segura
        model.eval()
        
        # Configurar el generador para Whisper
        model.generation_config.forced_decoder_ids = None
        model.generation_config.suppress_tokens = []
        model.generation_config.begin_suppress_tokens = []
        
        return processor, model, device
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return None, None, None

processor, model, device = load_model()

if processor is None or model is None:
    print("ERROR: No se pudo cargar el modelo. Verificar la ruta del modelo.")
    exit(1)


def debug_tokenizer_phonemes():
    """
    Función para verificar cómo el tokenizer maneja los símbolos fonéticos
    """
    # Ejemplos de labels como los del entrenamiento
    test_labels = [
        "W 2.0 IY0 2.0 K 2.0",
        "AO0 1.8 L 1.8 IH0 2.0", 
        "EH0 1.0 R 1.0"
    ]
    
    debug_info = "🔍 **ANÁLISIS DEL TOKENIZER CON FONEMAS:**\n\n"
    
    for i, label in enumerate(test_labels):
        debug_info += f"**Ejemplo {i+1}:** `{label}`\n"
        
        # Tokenizar
        tokens = processor.tokenizer.tokenize(label)
        token_ids = processor.tokenizer.convert_tokens_to_ids(tokens)
        
        debug_info += f"- Tokens: {tokens}\n"
        debug_info += f"- Token IDs: {token_ids}\n"
        
        # Decodificar de vuelta
        decoded = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
        debug_info += f"- Decodificado: `{decoded}`\n"
        debug_info += f"- ¿Coincide?: {'✅' if decoded.strip() == label.strip() else '❌'}\n\n"
    
    # Verificar tokens individuales problemáticos
    problematic_tokens = ["IY0", "AO0", "EH0", "2.0", "1.8", "1.0"]
    debug_info += "**Tokens individuales problemáticos:**\n"
    
    for token in problematic_tokens:
        try:
            tokenized = processor.tokenizer.tokenize(token)
            debug_info += f"- `{token}` → {tokenized}\n"
        except Exception as e:
            debug_info += f"- `{token}` → ERROR: {e}\n"
    
    return debug_info


def debug_training_data():
    """
    Función para examinar cómo se veían realmente los datos de entrenamiento
    """
    try:
        load_dotenv()
        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if not hf_token:
            return "No se encontró token de Hugging Face"
        
        # Cargar una muestra del dataset para ver el formato
        dataset = load_dataset("mispeech/speechocean762", "default", streaming=True, token=hf_token)
        sample = next(iter(dataset["train"]))
        
        # Recrear el procesamiento de labels como en el trainer
        words = sample['words']
        phonemes = []
        scores = []
        for word in words:
            phonemes.extend(word['phones'])
            scores.extend(word['phones-accuracy'])
        
        label_str = " ".join([f"{p} {s}" for p, s in zip(phonemes, scores)])
        
        # NUEVO: Verificar tokenización
        tokens = processor.tokenizer.tokenize(label_str)
        token_ids = processor.tokenizer.convert_tokens_to_ids(tokens)
        decoded_back = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
        
        debug_info = f"""
🔍 **DATOS DE ENTRENAMIENTO EJEMPLO:**
Texto original: {sample['text']}

Fonemas: {phonemes[:10]}...
Scores: {scores[:10]}...

Label procesado para entrenamiento:
{label_str[:200]}...

**🚨 ANÁLISIS DE TOKENIZACIÓN:**
Tokens: {tokens[:20]}...
Token IDs: {token_ids[:20]}...
Decodificado de vuelta: {decoded_back[:200]}...
¿Coincide con original?: {'✅' if decoded_back.strip() == label_str.strip() else '❌'}

Primeros 5 pares fonema-score:
{[(p, s) for p, s in zip(phonemes[:5], scores[:5])]}
        """
        return debug_info
    except Exception as e:
        return f"Error cargando datos de debug: {e}"


def print_tokenizer_vocab(processor):
    vocab = processor.tokenizer.get_vocab()
    print(f"Tamaño del vocabulario: {len(vocab)}")
    # Ordenar por ID
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    print("Primeros 20 tokens del vocabulario:")
    for token, idx in sorted_vocab[:20]:
        print(f"ID: {idx}\tToken: {repr(token)}")

print_tokenizer_vocab(processor)


def debug_model_output(raw_output):
    """
    Función para debugging detallado de la salida del modelo
    """
    debug_info = f"""
🔍 **DEBUG DETALLADO DE SALIDA:**

Salida cruda completa:
'{raw_output}'

Longitud: {len(raw_output)} caracteres

Contiene tokens especiales:
- <|startoftranscript|>: {'Sí' if '<|startoftranscript|>' in raw_output else 'No'}
- <|endoftext|>: {'Sí' if '<|endoftext|>' in raw_output else 'No'}
- <|notimestamps|>: {'Sí' if '<|notimestamps|>' in raw_output else 'No'}

Después de limpiar tokens especiales:
"""
    
    cleaned = raw_output.strip()
    for token in ['<|startoftranscript|>', '<|endoftext|>', '<|notimestamps|>']:
        cleaned = cleaned.replace(token, '')
    cleaned = cleaned.strip()
    
    debug_info += f"'{cleaned}'\n\n"
    debug_info += f"Tokens separados por espacio: {cleaned.split()}\n"
    debug_info += f"Número de tokens: {len(cleaned.split())}\n"
    
    return debug_info


def safe_generate(model, processor, input_features, strategy_name, **kwargs):
    """
    Función segura para generar con manejo de errores
    """
    try:
        with torch.no_grad():
            # Limpiar cache de CUDA si es necesario
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            predicted_ids = model.generate(
                input_features,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                **kwargs
            )
            
            # Validar que los IDs estén dentro del rango del vocabulario
            vocab_size = len(processor.tokenizer.get_vocab())
            max_id = predicted_ids.max().item()
            
            if max_id >= vocab_size:
                return f"ERROR: Token ID {max_id} fuera del rango del vocabulario (max: {vocab_size-1})"
            
            raw_output = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=False)[0]
            return raw_output
            
    except Exception as e:
        return f"ERROR en {strategy_name}: {str(e)}"


def parse_pronunciation_output(raw_text):
    """
    Parsea la salida del modelo que contiene fonemas y scores alternados.
    Formato esperado: "phoneme1 score1 phoneme2 score2 ..."
    """
    # Limpiar la salida (eliminar tokens especiales y espacios extra)
    cleaned_text = raw_text.strip()
    # Remover tokens especiales comunes de Whisper
    for token in ['<|startoftranscript|>', '<|endoftext|>', '<|notimestamps|>']:
        cleaned_text = cleaned_text.replace(token, '')
    
    cleaned_text = cleaned_text.strip()
    
    if not cleaned_text or cleaned_text.startswith("ERROR"):
        return [], f"Error en la generación: {cleaned_text}"
    
    # Dividir en tokens
    tokens = cleaned_text.split()
    
    phonemes_and_scores = []
    analysis = []
    
    # Procesar tokens de a pares (fonema, score)
    i = 0
    while i < len(tokens) - 1:
        phoneme = tokens[i]
        score_text = tokens[i + 1]
        
        try:
            # Intentar convertir el score a float
            score = float(score_text)
            phonemes_and_scores.append((phoneme, score))
            
            # Análisis del score
            if score >= 0.8:
                quality = "Excelente"
            elif score >= 0.6:
                quality = "Buena"
            elif score >= 0.4:
                quality = "Regular"
            else:
                quality = "Necesita mejora"
            
            analysis.append(f"/{phoneme}/ → {score:.2f} ({quality})")
            
        except ValueError:
            # Si no puede convertir a float, tratar como fonema individual
            analysis.append(f"/{phoneme}/ → Sin score")
            phonemes_and_scores.append((phoneme, None))
        
        i += 2
    
    # Si queda un token impar al final
    if i == len(tokens) - 1:
        phoneme = tokens[i]
        analysis.append(f"/{phoneme}/ → Sin score")
        phonemes_and_scores.append((phoneme, None))
    
    return phonemes_and_scores, analysis


def calculate_pronunciation_metrics(phonemes_and_scores):
    """
    Calcula métricas generales de pronunciación.
    """
    if not phonemes_and_scores:
        return "No hay datos para calcular métricas."
    
    valid_scores = [score for _, score in phonemes_and_scores if score is not None]
    
    if not valid_scores:
        return "No se encontraron scores válidos."
    
    avg_score = sum(valid_scores) / len(valid_scores)
    min_score = min(valid_scores)
    max_score = max(valid_scores)
    
    # Contar por calidad
    excellent = sum(1 for score in valid_scores if score >= 0.8)
    good = sum(1 for score in valid_scores if 0.6 <= score < 0.8)
    regular = sum(1 for score in valid_scores if 0.4 <= score < 0.6)
    needs_improvement = sum(1 for score in valid_scores if score < 0.4)
    
    metrics = f"""📊 **Métricas de Pronunciación:**
• **Score promedio:** {avg_score:.2f}/1.0
• **Rango de scores:** {min_score:.2f} - {max_score:.2f}
• **Total de fonemas:** {len(valid_scores)}

🎯 **Distribución de calidad:**
• Excelente (≥0.8): {excellent} fonemas
• Buena (0.6-0.79): {good} fonemas  
• Regular (0.4-0.59): {regular} fonemas
• Necesita mejora (<0.4): {needs_improvement} fonemas
"""
    
    return metrics


def infer(audio):
    # audio: (sr, data) tuple from gradio
    if audio is None:
        return "No audio recibido.", "", ""
    
    sampling_rate, data = audio
    
    try:
        # Procesar el audio
        inputs = processor(audio=data, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        
        # INTENTAR MÚLTIPLES ESTRATEGIAS DE GENERACIÓN (SEGURAS)
        generation_results = {}
        
        # Estrategia 1: Configuración básica y segura
        raw_output_1 = safe_generate(
            model, processor, input_features, "Estrategia 1",
            max_length=128,  # Más conservador
            min_length=5,
            num_beams=1,
            do_sample=False,
            early_stopping=True
        )
        generation_results["Estrategia 1 (Básica)"] = raw_output_1
        
        # Estrategia 2: Solo si la primera funciona
        if not raw_output_1.startswith("ERROR"):
            raw_output_2 = safe_generate(
                model, processor, input_features, "Estrategia 2",
                max_length=256,
                min_length=10,
                num_beams=2,
                do_sample=False,
                early_stopping=True
            )
            generation_results["Estrategia 2 (Beam)"] = raw_output_2
        else:
            generation_results["Estrategia 2 (Beam)"] = "Saltada por error en Estrategia 1"
        
        # Usar la primera estrategia para el análisis
        raw_output = raw_output_1
        
        # Debug detallado
        debug_detailed = debug_model_output(raw_output)
        
        # Procesar la salida para extraer fonemas y scores
        phonemes_and_scores, analysis = parse_pronunciation_output(raw_output)
        
        # Calcular métricas
        metrics = calculate_pronunciation_metrics(phonemes_and_scores)
        
        # Formatear análisis detallado
        detailed_analysis = "\n".join(analysis) if analysis else "No se pudo analizar la pronunciación."
        
        # Agregar debug de datos de entrenamiento
        training_debug = debug_training_data()
        
        # NUEVO: Agregar análisis del tokenizer
        tokenizer_debug = debug_tokenizer_phonemes()
        
        # Formatear salida cruda para debugging
        debug_output = f"""**🔧 COMPARACIÓN DE ESTRATEGIAS DE GENERACIÓN:**

"""
        
        for strategy_name, output in generation_results.items():
            debug_output += f"**{strategy_name}:**\n```\n{output}\n```\n\n"
        
        debug_output += f"""
**📊 ANÁLISIS DETALLADO:**
{debug_detailed}

{training_debug}

{tokenizer_debug}

**💡 DIAGNÓSTICO FINAL:**
- Modelo cargado: ✅
- Generación segura: {'✅' if not raw_output.startswith('ERROR') else '❌'}
- Formato esperado: fonema score fonema score
- **PROBLEMA IDENTIFICADO**: El tokenizer de Whisper no puede manejar símbolos fonéticos correctamente
- **SOLUCIÓN REQUERIDA**: Usar un enfoque diferente (wav2vec2 o tokenizer personalizado)
"""
        
        return detailed_analysis, metrics, debug_output
        
    except Exception as e:
        error_msg = f"Error durante la inferencia: {str(e)}"
        return error_msg, "Error en métricas", f"Error completo: {error_msg}"


# Crear la interfaz con múltiples salidas
demo = gr.Interface(
    fn=infer,
    inputs=gr.Audio(type="numpy", label="Sube tu audio"),
    outputs=[
        gr.Textbox(label="📝 Análisis de Pronunciación por Fonema", lines=10),
        gr.Textbox(label="📊 Métricas Generales", lines=8),
        gr.Textbox(label="🔧 Salida Cruda (Debug)", lines=15)
    ],
    title="🎯 Whisper Pronunciation Assessment",
    description="""
    **Sube un audio en inglés y obtén un análisis detallado de pronunciación.**
    
    El modelo analiza cada fonema y proporciona un score de 0.0 a 1.0:
    - **0.8-1.0**: Pronunciación excelente  
    - **0.6-0.79**: Pronunciación buena
    - **0.4-0.59**: Pronunciación regular
    - **0.0-0.39**: Necesita mejora
    """,
    examples=None,
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch() 