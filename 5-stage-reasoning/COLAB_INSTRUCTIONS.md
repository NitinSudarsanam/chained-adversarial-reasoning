# Instructions for Running in Google Colab

## The Problem

You're seeing this error:
```
AttributeError: 'LLMDiscriminator' object has no attribute 'train'
```

This happens because **Python caches imported modules**. Even though the code files have been updated with the bug fixes, your Colab runtime is still using the old cached versions.

## The Solution

You have **two options**:

### Option 1: Restart Runtime (RECOMMENDED)

1. In Colab, go to: **Runtime → Restart runtime**
2. Re-run all your setup cells (imports, installations, etc.)
3. Run your training script again

This is the cleanest solution and ensures all modules are reloaded.

### Option 2: Verify Fixes First

Before restarting, you can verify the fixes are in the files:

```python
# Run this in a Colab cell
!python verify_fixes.py
```

This will check if all the bug fixes are present in the code files.

## What Was Fixed

All these bugs have been fixed in the code files:

1. ✅ Added `train()`, `eval()`, `parameters()` methods to `LLMGenerator` and `LLMDiscriminator`
2. ✅ Fixed `get_log_probs()` to allow gradients (removed `torch.no_grad()`)
3. ✅ Added edge case handling for empty outputs and truncation
4. ✅ Fixed all model references in `adversarial_trainer.py` (6 locations)
5. ✅ Added empty input validation in sandbox
6. ✅ Added NaN/Inf handling in training loop

## After Restarting

After you restart the runtime, the training should work without AttributeErrors. The code will properly:
- Train models with gradient computation
- Handle edge cases gracefully
- Use the correct wrapper classes throughout

## If You Still See Errors

If you restart and still see the same error, it means the files weren't synced to Colab. In that case:

1. Check if you're editing files locally or in Colab
2. Make sure file changes are saved
3. If using Google Drive, ensure files are synced
4. You may need to re-upload the modified files to Colab

## Quick Test

After restarting, run this to confirm the methods exist:

```python
from models.discriminator import LLMDiscriminator
from models.generator import LLMGenerator

print("Discriminator methods:", [m for m in dir(LLMDiscriminator) if not m.startswith('_')])
print("Generator methods:", [m for m in dir(LLMGenerator) if not m.startswith('_')])

# Should see 'train', 'eval', 'parameters' in the output
```
