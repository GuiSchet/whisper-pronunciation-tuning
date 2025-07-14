# Configuración de PyTorch con CUDA 12.1

## Requisitos Previos

### 1. GPU NVIDIA
- Verifica que tienes una GPU NVIDIA compatible
- Ejecuta `nvidia-smi` en la terminal para verificar

### 2. Drivers NVIDIA
- Descarga e instala los drivers más recientes desde: https://www.nvidia.com/drivers
- Reinicia tu sistema después de instalar

### 3. CUDA Toolkit (Opcional)
- No es necesario instalar CUDA Toolkit separadamente
- PyTorch incluye las librerías CUDA necesarias

## Instalación Automática

### Opción 1: Script Automático
```bash
uv run python install_pytorch_cuda.py
```

### Opción 2: Usando uv con configuración actualizada
```bash
# Desinstalar versión CPU
uv run pip uninstall torch torchvision torchaudio -y

# Instalar PyTorch con CUDA 12.1
uv run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verificar instalación
uv run python check_cuda.py
```

### Opción 3: Reinstalar completamente el entorno
```bash
# Eliminar entorno virtual
rm -rf .venv

# Reinstalar con PyTorch CUDA
uv sync
uv run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Verificación

### Verificar CUDA
```bash
uv run python check_cuda.py
```

### Verificar en código Python
```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

## Uso en el Proyecto

### Entrenamiento con GPU
```bash
# El modelo usará GPU automáticamente si está disponible
uv run python main.py train --epochs 5 --batch-size 4
```

### Probar dataset
```bash
uv run python main.py test-data
```

### Análisis de pronunciación
```bash
uv run python main.py analyze --audio-file audio.wav
```

## Solución de Problemas

### Error "CUDA out of memory"
- Reduce el batch size: `--batch-size 2` o `--batch-size 1`
- Reduce gradient accumulation: `--gradient-accumulation-steps 4`

### CUDA no detectado
1. Verificar drivers NVIDIA: `nvidia-smi`
2. Verificar instalación PyTorch: `uv run python check_cuda.py`
3. Reinstalar PyTorch: `uv run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

### Versión incorrecta de CUDA
- Este proyecto está configurado para CUDA 12.1
- Si tienes CUDA 11.8, cambia la URL a: `https://download.pytorch.org/whl/cu118`

## Configuración Avanzada

### Configurar GPU específica
```python
import torch
torch.cuda.set_device(0)  # Usar GPU 0
```

### Verificar memoria GPU
```python
import torch
if torch.cuda.is_available():
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
```

## Notas Importantes

1. **Drivers NVIDIA**: Deben ser compatible con CUDA 12.1
2. **Memoria GPU**: Necesitas al menos 6GB para entrenamiento básico
3. **Versiones**: PyTorch se actualiza frecuentemente, siempre verifica compatibilidad
 