import os
import random
import soundfile as sf
from dotenv import load_dotenv
from datasets import load_dataset

def extract_audio_samples():
    """
    Extrae 5 audios aleatorios del dataset speechocean762 y los guarda 
    en la carpeta audio_samples
    """
    # 1. CONFIGURACIÓN
    print("--- Extrayendo muestras de audio del dataset ---")
    load_dotenv()
    hf_token = os.getenv("HUGGING_FACE_TOKEN")
    if not hf_token:
        raise ValueError(
            "No se encontró el token de Hugging Face. "
            "Asegúrate de tener un archivo .env con HUGGING_FACE_TOKEN='hf_...'"
        )
    
    DATASET_NAME = "mispeech/speechocean762"
    OUTPUT_FOLDER = "audio_samples"
    NUM_SAMPLES = 5
    
    # 2. CREAR CARPETA DE SALIDA
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Carpeta '{OUTPUT_FOLDER}' creada.")
    else:
        print(f"Usando carpeta existente '{OUTPUT_FOLDER}'.")
    
    # 3. CARGAR EL DATASET
    print(f"Cargando dataset '{DATASET_NAME}'...")
    try:
        # Cargar el dataset en modo streaming primero
        raw_dataset = load_dataset(DATASET_NAME, "default", streaming=True, token=hf_token)
        
        # Convertir una porción del dataset de entrenamiento a lista para poder seleccionar aleatoriamente
        print("Convirtiendo dataset a memoria (esto puede tomar un momento)...")
        train_data = []
        
        # Tomar los primeros 1000 elementos para tener una buena muestra
        for i, example in enumerate(raw_dataset["train"]):
            if i >= 1000:  # Limitar a 1000 para no cargar todo el dataset
                break
            train_data.append(example)
            if i % 100 == 0:
                print(f"Cargados {i+1} elementos...")
        
        print(f"Dataset cargado: {len(train_data)} muestras disponibles")
        
    except Exception as e:
        print(f"Error cargando el dataset: {e}")
        return
    
    # 4. SELECCIONAR MUESTRAS ALEATORIAS
    if len(train_data) < NUM_SAMPLES:
        print(f"Advertencia: Solo hay {len(train_data)} muestras disponibles, menor que {NUM_SAMPLES}")
        NUM_SAMPLES = len(train_data)
    
    random_samples = random.sample(train_data, NUM_SAMPLES)
    print(f"Seleccionadas {NUM_SAMPLES} muestras aleatorias")
    
    # 5. GUARDAR LOS AUDIOS
    for i, sample in enumerate(random_samples):
        try:
            # Obtener los datos de audio
            audio_data = sample["audio"]
            audio_array = audio_data["array"]
            sampling_rate = audio_data["sampling_rate"]
            
            # Crear nombre del archivo
            filename = f"sample_{i+1:02d}.wav"
            filepath = os.path.join(OUTPUT_FOLDER, filename)
            
            # Guardar el archivo de audio
            sf.write(filepath, audio_array, sampling_rate)
            
            # Mostrar información sobre la muestra
            text = sample.get("text", "No disponible")
            duration = len(audio_array) / sampling_rate
            
            print(f"✓ Guardado: {filename}")
            print(f"  - Texto: {text}")
            print(f"  - Duración: {duration:.2f} segundos")
            print(f"  - Sample rate: {sampling_rate} Hz")
            print()
            
        except Exception as e:
            print(f"Error guardando muestra {i+1}: {e}")
    
    print(f"¡Proceso completado! {NUM_SAMPLES} audios guardados en '{OUTPUT_FOLDER}/'")

if __name__ == "__main__":
    extract_audio_samples()
