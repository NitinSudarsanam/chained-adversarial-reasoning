"""Configure PyTorch memory settings for training."""

import os
import torch

# Set environment variable for memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print("Memory configuration:")
print(f"  PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")

if torch.cuda.is_available():
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Print memory info
    device = torch.cuda.current_device()
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    
    print(f"\nGPU Memory Status:")
    print(f"  Total: {total_memory:.2f} GiB")
    print(f"  Allocated: {allocated:.2f} GiB")
    print(f"  Reserved: {reserved:.2f} GiB")
    print(f"  Free: {total_memory - reserved:.2f} GiB")
    
    print("\n✓ Memory configuration complete")
else:
    print("\n⚠ CUDA not available")
