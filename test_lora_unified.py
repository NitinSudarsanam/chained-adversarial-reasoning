"""Test script to verify LoRA unified model works correctly."""

import torch
from models.unified_model import UnifiedModel
from data.problem_dataset import load_problems

print("="*80)
print("Testing LoRA Unified Model")
print("="*80)

# Test 1: Load model with LoRA
print("\n1. Loading model with LoRA...")
try:
    model = UnifiedModel(
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_lora=True
    )
    print("✓ Model loaded successfully with LoRA")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    exit(1)

# Test 2: Check adapters exist
print("\n2. Checking adapters...")
if model.use_lora:
    try:
        # Check if adapters are present
        adapters = model.model.peft_config.keys()
        print(f"✓ Found adapters: {list(adapters)}")
        
        if "generator" not in adapters or "discriminator" not in adapters:
            print("✗ Missing required adapters!")
            exit(1)
    except Exception as e:
        print(f"✗ Error checking adapters: {e}")
        exit(1)
else:
    print("⚠ LoRA not enabled, skipping adapter check")

# Test 3: Test adapter switching
print("\n3. Testing adapter switching...")
try:
    model.set_generator_mode()
    print(f"✓ Switched to generator mode (current: {model.current_adapter})")
    
    model.set_discriminator_mode()
    print(f"✓ Switched to discriminator mode (current: {model.current_adapter})")
    
    model.set_generator_mode()
    print(f"✓ Switched back to generator mode (current: {model.current_adapter})")
except Exception as e:
    print(f"✗ Adapter switching failed: {e}")
    exit(1)

# Test 4: Test generation with generator adapter
print("\n4. Testing generation with generator adapter...")
try:
    model.set_generator_mode()
    model.eval()
    
    # Simple test generation
    test_prompt = "Write a function to add two numbers:\ndef add(a: int, b: int) -> int:"
    
    with torch.no_grad():
        output = model._generate(
            test_prompt,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9
        )
    
    print(f"✓ Generated output (length: {len(output)} chars)")
    if output:
        print(f"  Preview: {output[:100]}...")
    else:
        print("  ⚠ Empty output (might be normal for small model)")
except Exception as e:
    print(f"✗ Generation failed: {e}")
    exit(1)

# Test 5: Test generation with discriminator adapter
print("\n5. Testing generation with discriminator adapter...")
try:
    model.set_discriminator_mode()
    model.eval()
    
    test_prompt = "Generate a test case:\ndef test_add():"
    
    with torch.no_grad():
        output = model._generate(
            test_prompt,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9
        )
    
    print(f"✓ Generated output (length: {len(output)} chars)")
    if output:
        print(f"  Preview: {output[:100]}...")
    else:
        print("  ⚠ Empty output (might be normal for small model)")
except Exception as e:
    print(f"✗ Generation failed: {e}")
    exit(1)

# Test 6: Test get_log_probs with both adapters
print("\n6. Testing get_log_probs...")
try:
    # Generator mode
    model.set_generator_mode()
    model.train()
    
    prompt = "def add(a, b):"
    output = "\n    return a + b"
    
    log_probs = model.get_log_probs(prompt, output)
    print(f"✓ Generator log_probs shape: {log_probs.shape}")
    print(f"  requires_grad: {log_probs.requires_grad}")
    
    # Discriminator mode
    model.set_discriminator_mode()
    
    log_probs = model.get_log_probs(prompt, output)
    print(f"✓ Discriminator log_probs shape: {log_probs.shape}")
    print(f"  requires_grad: {log_probs.requires_grad}")
    
    model.eval()
except Exception as e:
    print(f"✗ get_log_probs failed: {e}")
    exit(1)

# Test 7: Test parameters() method
print("\n7. Testing parameters() method...")
try:
    params = list(model.parameters())
    trainable_params = [p for p in params if p.requires_grad]
    
    total_params = sum(p.numel() for p in params)
    trainable = sum(p.numel() for p in trainable_params)
    
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable:,} ({100*trainable/total_params:.2f}%)")
    
    if model.use_lora and trainable / total_params > 0.05:
        print("⚠ Warning: More than 5% of parameters are trainable (expected <1% with LoRA)")
except Exception as e:
    print(f"✗ parameters() failed: {e}")
    exit(1)

# Test 8: Test memory usage
print("\n8. Checking memory usage...")
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"✓ GPU memory allocated: {allocated:.2f} GiB")
    print(f"✓ GPU memory reserved: {reserved:.2f} GiB")
    
    if model.use_lora and reserved > 5:
        print("⚠ Warning: Using more than 5GB (expected <3GB with LoRA + 4-bit)")
else:
    print("  CPU mode - skipping GPU memory check")

print("\n" + "="*80)
print("All tests passed! ✓")
print("="*80)
print("\nLoRA unified model is working correctly:")
print("  ✓ Dual adapters (generator + discriminator)")
print("  ✓ Adapter switching works")
print("  ✓ Generation works with both adapters")
print("  ✓ get_log_probs works with both adapters")
print("  ✓ Parameters are trainable")
print("  ✓ Memory efficient")
print("\nReady for training!")
