#!/usr/bin/env python3
"""
Test script to verify GPU acceleration functionality.
"""

import sys
import os
sys.path.insert(0, "src")

from utils.device_config import setup_device
import numpy as np
import time

def test_device_setup():
    """Test device configuration and detection"""
    print("Testing GPU/CPU device configuration...")
    print("=" * 50)
    
    # Test GPU setup
    print("1. Testing GPU setup...")
    gpu_device = setup_device(force_cpu=False)
    print(f"   Backend: {gpu_device.backend}")
    print(f"   Using GPU: {gpu_device.use_gpu}")
    
    if gpu_device.use_gpu:
        used, total = gpu_device.get_memory_info()
        print(f"   GPU Memory: {used / 1e9:.1f}GB / {total / 1e9:.1f}GB")
    
    # Test CPU setup
    print("\n2. Testing CPU setup...")
    cpu_device = setup_device(force_cpu=True)
    print(f"   Backend: {cpu_device.backend}")
    print(f"   Using GPU: {cpu_device.use_gpu}")
    
    return gpu_device, cpu_device

def benchmark_operations(device, name):
    """Benchmark basic operations on device"""
    print(f"\n3. Benchmarking {name} operations...")
    
    # Test array creation
    start_time = time.time()
    large_array = device.zeros((10000, 1000), dtype=np.float32)
    creation_time = time.time() - start_time
    print(f"   Array creation (10M elements): {creation_time:.4f}s")
    
    # Test array operations
    start_time = time.time()
    array1 = device.random_random((5000, 2000))
    array2 = device.random_random((5000, 2000))
    result = device.maximum(array1, array2)
    sum_result = device.sum(result)
    if device.use_gpu:
        device.synchronize()
    operation_time = time.time() - start_time
    print(f"   Matrix operations: {operation_time:.4f}s")
    
    # Convert back to numpy for verification
    final_result = device.to_numpy(sum_result)
    print(f"   Final result (verification): {float(final_result):.2f}")
    
    return creation_time, operation_time

def test_cfr_components():
    """Test CFR components with GPU"""
    print("\n4. Testing CFR components...")
    
    try:
        from cfr.mccfr import InfoSet
        from utils.device_config import get_device_config
        
        device = get_device_config()
        
        # Test InfoSet creation
        infoset = InfoSet("test_key", 4, device)
        print(f"   Created InfoSet with {infoset.num_actions} actions")
        print(f"   Regret sum shape: {infoset.regret_sum.shape if hasattr(infoset.regret_sum, 'shape') else 'scalar'}")
        
        # Test strategy computation
        strategy = infoset.get_strategy(1.0)
        print(f"   Strategy computed: {strategy}")
        print(f"   Strategy sum: {np.sum(strategy):.6f}")
        
        print("   ✓ CFR components working correctly")
        
    except Exception as e:
        print(f"   ✗ Error testing CFR components: {e}")

def main():
    print("GPU Acceleration Test Suite")
    print("=" * 50)
    
    # Test device setup
    gpu_device, cpu_device = test_device_setup()
    
    # Benchmark GPU vs CPU if GPU is available
    if gpu_device.use_gpu:
        gpu_creation, gpu_operation = benchmark_operations(gpu_device, "GPU")
        cpu_creation, cpu_operation = benchmark_operations(cpu_device, "CPU")
        
        print(f"\n5. Performance Comparison:")
        print(f"   Array creation speedup: {cpu_creation / gpu_creation:.2f}x")
        print(f"   Operations speedup: {cpu_operation / gpu_operation:.2f}x")
    else:
        print("\n5. GPU not available, testing CPU only...")
        benchmark_operations(cpu_device, "CPU")
    
    # Test CFR components
    test_cfr_components()
    
    print("\n" + "=" * 50)
    print("Test completed!")
    
    if gpu_device.use_gpu:
        print("✓ GPU acceleration is available and working")
        print("  Use --cpu flag to force CPU-only mode if needed")
    else:
        print("⚠ GPU acceleration not available")
        print("  Install CuPy or PyTorch with CUDA support for GPU acceleration")
        print("  See requirements.txt for installation instructions")

if __name__ == "__main__":
    main()