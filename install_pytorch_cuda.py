#!/usr/bin/env python3
"""
Script para instalar PyTorch con CUDA 12.1 en el entorno virtual de uv
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Ejecutar comando y mostrar salida"""
    print(f"Ejecutando: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    return result.returncode == 0

def check_nvidia_gpu():
    """Verificar que hay una GPU NVIDIA disponible"""
    print("🔍 Verificando GPU NVIDIA...")
    
    # Verificar nvidia-smi
    result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ nvidia-smi no está disponible")
        print("   Instala los drivers NVIDIA desde: https://www.nvidia.com/drivers")
        return False
    
    print("✅ nvidia-smi encontrado")
    print(result.stdout)
    return True

def main():
    """Instalar PyTorch con CUDA 12.1"""
    print("🚀 Instalando PyTorch con CUDA 12.1...")
    
    # Verificar GPU
    if not check_nvidia_gpu():
        print("\n⚠️  Sin GPU NVIDIA detectada. Instalaremos PyTorch con CUDA de todas formas.")
        print("   Puedes usar CPU mientras tanto.")
        response = input("¿Continuar? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Desinstalar versiones existentes
    print("\n📦 Desinstalando versiones existentes de PyTorch...")
    run_command("uv run pip uninstall torch torchvision torchaudio -y")
    
    # Instalar PyTorch con CUDA 12.1 (última versión)
    print("\n🔧 Instalando PyTorch con CUDA 12.1...")
    cmd = 'uv run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121'
    success = run_command(cmd)
    
    if success:
        print("\n✅ Instalación completada. Verificando...")
        
        # Verificar instalación
        print("\n🔍 Verificando instalación...")
        verify_cmd = 'uv run python check_cuda.py'
        run_command(verify_cmd)
        
        print("\n" + "="*50)
        print("INSTRUCCIONES FINALES")
        print("="*50)
        print("1. Si CUDA está disponible, el modelo usará GPU automáticamente")
        print("2. Para verificar CUDA en cualquier momento: uv run python check_cuda.py")
        print("3. Para entrenar el modelo: uv run python main.py train")
        print("4. Para probar el dataset: uv run python main.py test-data")
        print("="*50)
    else:
        print("\n❌ Error en la instalación")
        print("Intenta manualmente:")
        print("uv run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

if __name__ == "__main__":
    main() 