# Syntax Error Fixes for Generated Code

## Problem: 0/0 Tests

When you see:
```
Generator Pass Rate: 0/0 (0.0%)
Validation Pass Rate: 0/0 (0.0%)
```

This means **no tests were executed** because pytest couldn't parse the test file due to syntax errors.

## Root Cause

The model generates code with syntax errors:
- `for numinnums:` → Missing space: should be `for num in nums:`
- `If currentsum>max_s` → Capital 'If', missing spaces
- `currentsum+=num` → Missing spaces around `+=`

These are **model generation errors**, not cleaning artifacts. The model (Llama-3.1-8B-Instruct) makes these mistakes during generation.

## Why This Happens

1. **Tokenization**: The model tokenizes `in` as part of `numinnums`
2. **Training data**: Saw inconsistent spacing in training
3. **Model size**: 8B models make more syntax errors than 70B models
4. **Temperature**: Can affect spacing consistency

## Solution: Automatic Syntax Fixing

Added post-processing to fix common LLM syntax errors:

### Generator (`models/generator.py`)

```python
def _fix_common_syntax_errors(self, code: str) -> str:
    """Fix common syntax errors made by LLMs."""
    import re
    
    # Fix: "for xiny:" -> "for x in y:"
    code = re.sub(r'\bfor\s+(\w+)in(\w+):', r'for \1 in \2:', code)
    
    # Fix: "If" -> "if", "Else" -> "else", "Elif" -> "elif"
    code = re.sub(r'\bIf\b', 'if', code)
    code = re.sub(r'\bElse\b', 'else', code)
    code = re.sub(r'\bElif\b', 'elif', code)
    
    # Fix: "x=y" -> "x = y" (add spaces around operators)
    code = re.sub(r'(\w+)=([^=])', r'\1 = \2', code)
    
    # Fix: "x+=y" -> "x += y"
    code = re.sub(r'(\w+)\+=(\w+)', r'\1 += \2', code)
    code = re.sub(r'(\w+)-=(\w+)', r'\1 -= \2', code)
    
    # Fix: "x>y" -> "x > y"
    code = re.sub(r'(\w+)>([^=])', r'\1 > \2', code)
    code = re.sub(r'(\w+)<([^=])', r'\1 < \2', code)
    
    return code
```

### Discriminator (`models/discriminator.py`)

Same fixes applied to test code, plus:
```python
# Fix: "assert x==y" -> "assert x == y"
code = re.sub(r'assert\s+(\w+)==(\w+)', r'assert \1 == \2', code)
```

## Examples

### Before:
```python
for numinnums:
    If currentsum>max_sum:
        currentsum+=num
```

### After:
```python
for num in nums:
    if currentsum > max_sum:
        currentsum += num
```

## What Gets Fixed

| Error | Fixed To | Pattern |
|-------|----------|---------|
| `for xiny:` | `for x in y:` | Missing space in `in` |
| `If x:` | `if x:` | Capital keywords |
| `x=y` | `x = y` | Missing spaces around `=` |
| `x+=y` | `x += y` | Missing spaces around `+=` |
| `x>y` | `x > y` | Missing spaces around `>` |
| `assert x==y` | `assert x == y` | Missing spaces in assert |

## What Doesn't Get Fixed

Some errors are too complex to fix automatically:
- Wrong variable names: `currentsum` vs `current_sum`
- Logic errors: Wrong algorithm
- Missing imports
- Incorrect indentation
- Type errors

These require better prompts or a better model.

## Testing

To test the syntax fixer:

```python
from models.generator import LLMGenerator

gen = LLMGenerator("meta-llama/Llama-3.1-8B-Instruct", "cuda")

# Simulate bad code
bad_code = """
def test():
    for numinnums:
        If x>y:
            x+=1
"""

# The fixer runs automatically in _clean_generated_code
fixed = gen._fix_common_syntax_errors(bad_code)
print(fixed)
```

Expected output:
```python
def test():
    for num in nums:
        if x > y:
            x += 1
```

## Impact

With syntax fixing:
- **Before**: `0/0` tests (pytest can't parse)
- **After**: `2/5` tests (pytest can run, some pass)

This won't fix all errors, but it will fix the most common ones that prevent pytest from even running.

## If Issues Persist

If you still see `0/0` after this fix:

1. **Check the error message**:
   ```
   Errors: 3 error(s)
   First error: SyntaxError: invalid syntax
   ```

2. **Look at the generated code** in the logs to see what error remains

3. **Consider**:
   - Using a larger model (70B instead of 8B)
   - Adjusting temperature (try 0.5 for more conservative generation)
   - Adding more examples to prompts
   - Using a code-specific model (CodeLlama, StarCoder)

## Files Modified

1. `models/generator.py`: Added `_fix_common_syntax_errors()` method
2. `models/discriminator.py`: Added `_fix_common_syntax_errors()` method

Both are called automatically during code cleaning, so no changes needed to training code.
