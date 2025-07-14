#!/usr/bin/env python3
"""
Script para verificar que PyTorch CUDA funciona correctamente
"""

import torch
import sys

def check_cuda():
    print("=" * 50)
    print("VERIFICACIÓN DE CUDA")
    print("=" * 50)
    
    # Información básica de PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    
    # Verificar disponibilidad de CUDA
    print(f"\nCUDA disponible: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Número de GPUs: {torch.cuda.device_count()}")
        
        # Información de cada GPU
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {gpu_props.name}")
            print(f"  Memoria total: {gpu_props.total_memory / 1024**3:.2f} GB")
            print(f"  Memoria libre: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  Compute capability: {gpu_props.major}.{gpu_props.minor}")
        
        # Prueba básica de GPU
        print("\n" + "=" * 50)
        print("PRUEBA BÁSICA DE GPU")
        print("=" * 50)
        
        try:
            # Crear tensor en GPU
            device = torch.device("cuda:0")
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            
            # Operación matemática
            z = torch.mm(x, y)
            
            print(f"✅ Operación en GPU exitosa!")
            print(f"   Tensor shape: {z.shape}")
            print(f"   Device: {z.device}")
            
            # Verificar tiempo de ejecución
            import time
            start = time.time()
            for _ in range(100):
                z = torch.mm(x, y)
            torch.cuda.synchronize()
            end = time.time()
            
            print(f"   100 multiplicaciones de matrices (1000x1000): {end - start:.4f}s")
            
        except Exception as e:
            print(f"❌ Error en prueba de GPU: {e}")
    else:
        print("❌ CUDA no está disponible")
        print("   Verificar:")
        print("   1. Que tienes una GPU NVIDIA")
        print("   2. Que tienes los drivers NVIDIA instalados")
        print("   3. Que tienes CUDA Toolkit instalado")
        print("   4. Que PyTorch está instalado con soporte CUDA")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    check_cuda() 