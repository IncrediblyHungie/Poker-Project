"""
GPU/CPU device configuration and management.
Handles automatic detection and setup of compute devices.
"""

import numpy as np
import os
import logging
from typing import Tuple, Optional

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

# Try to import PyTorch for GPU support
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


class DeviceConfig:
    """Manages device configuration for GPU/CPU computation"""
    
    def __init__(self, force_cpu: bool = False, device_id: int = 0):
        self.force_cpu = force_cpu
        self.device_id = device_id
        self.use_gpu = False
        self.backend = 'numpy'
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        self._setup_device()
    
    def _setup_device(self):
        """Setup the best available compute device"""
        if self.force_cpu:
            self.logger.info("Forcing CPU computation")
            self._setup_cpu()
            return
        
        # Try to setup GPU
        if self._try_setup_gpu():
            self.logger.info(f"Using GPU acceleration with {self.backend}")
        else:
            self.logger.info("Falling back to CPU computation")
            self._setup_cpu()
    
    def _try_setup_gpu(self) -> bool:
        """Try to setup GPU acceleration"""
        # Try CuPy first (better for numerical computing)
        if CUPY_AVAILABLE and self._test_cupy():
            self.use_gpu = True
            self.backend = 'cupy'
            return True
        
        # Try PyTorch as fallback
        if TORCH_AVAILABLE and self._test_torch():
            self.use_gpu = True
            self.backend = 'torch'
            return True
        
        return False
    
    def _test_cupy(self) -> bool:
        """Test if CuPy is working properly"""
        try:
            cp.cuda.Device(self.device_id).use()
            # Simple test operation
            test_array = cp.random.random((100, 100))
            result = cp.sum(test_array)
            cp.cuda.Stream.null.synchronize()
            return True
        except Exception as e:
            self.logger.warning(f"CuPy GPU test failed: {e}")
            return False
    
    def _test_torch(self) -> bool:
        """Test if PyTorch GPU is working properly"""
        try:
            if not torch.cuda.is_available():
                return False
            
            device = torch.device(f'cuda:{self.device_id}')
            # Simple test operation
            test_tensor = torch.randn(100, 100, device=device)
            result = torch.sum(test_tensor)
            torch.cuda.synchronize()
            return True
        except Exception as e:
            self.logger.warning(f"PyTorch GPU test failed: {e}")
            return False
    
    def _setup_cpu(self):
        """Setup CPU computation"""
        self.use_gpu = False
        self.backend = 'numpy'
        
        # Set numpy threads for better CPU performance
        os.environ['OMP_NUM_THREADS'] = str(os.cpu_count() or 4)
        os.environ['MKL_NUM_THREADS'] = str(os.cpu_count() or 4)
    
    def array(self, data, dtype=None):
        """Create array on the appropriate device"""
        if self.backend == 'cupy':
            return cp.array(data, dtype=dtype)
        elif self.backend == 'torch':
            tensor = torch.tensor(data, dtype=dtype)
            if self.use_gpu:
                tensor = tensor.cuda(self.device_id)
            return tensor
        else:
            return np.array(data, dtype=dtype)
    
    def zeros(self, shape, dtype=None):
        """Create zeros array on the appropriate device"""
        if self.backend == 'cupy':
            return cp.zeros(shape, dtype=dtype)
        elif self.backend == 'torch':
            tensor = torch.zeros(shape, dtype=dtype)
            if self.use_gpu:
                tensor = tensor.cuda(self.device_id)
            return tensor
        else:
            return np.zeros(shape, dtype=dtype)
    
    def ones(self, shape, dtype=None):
        """Create ones array on the appropriate device"""
        if self.backend == 'cupy':
            return cp.ones(shape, dtype=dtype)
        elif self.backend == 'torch':
            tensor = torch.ones(shape, dtype=dtype)
            if self.use_gpu:
                tensor = tensor.cuda(self.device_id)
            return tensor
        else:
            return np.ones(shape, dtype=dtype)
    
    def to_numpy(self, array):
        """Convert array to numpy (for saving/interfacing)"""
        if self.backend == 'cupy':
            return cp.asnumpy(array)
        elif self.backend == 'torch':
            if hasattr(array, 'cpu'):
                return array.cpu().numpy()
            return array.numpy()
        else:
            return np.asarray(array)
    
    def maximum(self, arr1, arr2):
        """Element-wise maximum"""
        if self.backend == 'cupy':
            return cp.maximum(arr1, arr2)
        elif self.backend == 'torch':
            return torch.maximum(arr1, arr2)
        else:
            return np.maximum(arr1, arr2)
    
    def sum(self, array, axis=None):
        """Sum array elements"""
        if self.backend == 'cupy':
            return cp.sum(array, axis=axis)
        elif self.backend == 'torch':
            return torch.sum(array, dim=axis)
        else:
            return np.sum(array, axis=axis)
    
    def random_random(self, shape):
        """Generate random array"""
        if self.backend == 'cupy':
            return cp.random.random(shape)
        elif self.backend == 'torch':
            tensor = torch.rand(shape)
            if self.use_gpu:
                tensor = tensor.cuda(self.device_id)
            return tensor
        else:
            return np.random.random(shape)
    
    def synchronize(self):
        """Synchronize GPU operations"""
        if self.backend == 'cupy':
            cp.cuda.Stream.null.synchronize()
        elif self.backend == 'torch' and self.use_gpu:
            torch.cuda.synchronize()
    
    def get_memory_info(self) -> Tuple[int, int]:
        """Get GPU memory info (used, total) in bytes"""
        if not self.use_gpu:
            return 0, 0
        
        if self.backend == 'cupy':
            mempool = cp.get_default_memory_pool()
            return mempool.used_bytes(), mempool.total_bytes()
        elif self.backend == 'torch':
            return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        
        return 0, 0
    
    def clear_cache(self):
        """Clear GPU memory cache"""
        if not self.use_gpu:
            return
        
        if self.backend == 'cupy':
            cp.get_default_memory_pool().free_all_blocks()
        elif self.backend == 'torch':
            torch.cuda.empty_cache()


# Global device configuration
_device_config: Optional[DeviceConfig] = None


def get_device_config(force_cpu: bool = False, device_id: int = 0) -> DeviceConfig:
    """Get or create global device configuration"""
    global _device_config
    
    if _device_config is None:
        _device_config = DeviceConfig(force_cpu=force_cpu, device_id=device_id)
    
    return _device_config


def setup_device(force_cpu: bool = False, device_id: int = 0) -> DeviceConfig:
    """Setup device configuration"""
    global _device_config
    _device_config = DeviceConfig(force_cpu=force_cpu, device_id=device_id)
    return _device_config