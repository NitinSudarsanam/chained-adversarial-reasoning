# Fix for Generation Format Issues

## Problem Identified

From the training logs:
```
Generated Code (length=368):
  This optimized version includes improved comments, better naming conventions...
  
Generated Tests (length=2873):
  def maxsubarray(nums):
    for numinnums:  # Missing space!
      If currentsum>max_s...  # Capital 'If'!
```

**Issues**:
1. Model generating explanatory text instead of code
2. Syntax errors in generated code (missing spaces, wrong capitalization)
3. Result: 0/0 tests pass because code can't execute

## Root Cause

The model (Llama-3.1-8B-Instruct) is:
- Not following the prompt format strictly
- Adding explanations before/after code
- Making syntax errors (spacing, capitalization)

## Solutions Applied

### 1. Stricter Prompts

**Generator Prompt (Stage 5)**:
```
STRICT RULES:
- Output ONLY the function code
- NO explanations, NO comments, NO text before or after
- Start directly with "def"
- Use the EXACT function signature
- Write syntactically valid Python
- Return the result

CODE (no explanations):
```python
```

**Discriminator Prompt (Stage 5)**:
```
STRICT RULES:
- Output ONLY {num_tests} test functions
- NO explanations, NO comments outside functions
- Each test MUST start with "def test_"
- Use correct Python syntax (spaces, colons, indentation)
- Use assert statements
- Import pytest at the top

TESTS (no explanations):
```python
import pytest
```

### 2. Better Code Cleaning

**Generator** (`models/generator.py`):
- Added filter to remove text before first `def` or `class`
- Strips explanatory paragraphs automatically

```python
# Find first function/class definition
for i, line in enumerate(lines):
    if line.strip().startswith(('def ', 'class ')):
        first_def_idx = i
        break

# Remove everything before it
if first_def_idx > 0:
    lines = lines[first_def_idx:]
```

**Discriminator** (`models/discriminator.py`):
- Added filter to remove text before first `import` or `def test_`
- Ensures only test code remains

```python
# Find first import or test function
for i, line in enumerate(lines):
    if stripped.startswith(('import ', 'from ', 'def test_')):
        first_code_idx = i
        break

# Remove everything before it
if first_code_idx > 0:
    lines = lines[first_code_idx:]
```

### 3. All Other Stage Prompts Updated

Updated all discriminator prompts (stages 1-4) with same strict format:
- "Output ONLY test functions"
- "NO explanations"
- "Use correct Python syntax"
- Clear example format

## Expected Improvement

### Before:
```
Generated Code (length=368):
  This optimized version includes improved comments...
  
  def add(a, b):
      return a + b
```
→ Cleaned to:
```python
def add(a, b):
    return a + b
```

### Before:
```
Generated Tests (length=2873):
  Here are some test cases:
  
  import pytest
  def test_add():
      assert add(2,3)==5
```
→ Cleaned to:
```python
import pytest

def test_add():
    assert add(2, 3) == 5
```

## Monitoring

With the enhanced logging, you'll now see:
1. **Raw length**: Shows if model generated too much text
2. **Cleaned preview**: Shows first 300 chars after cleaning
3. **Execution errors**: Shows if syntax errors remain

Example output:
```
Step 1/1 - Discriminator Reward: 0.4500
  Generator Pass Rate: 2/5 (40.0%)
  Validation Pass Rate: 4/5 (80.0%)
  Generated Code (length=145):
    def add(a, b):
        return a + b
  Generated Tests (length=234):
    import pytest
    
    def test_add():
        assert add(2, 3) == 5
  Execution Results:
    Timeout: False
    Errors: 0 error(s)
```

## If Issues Persist

If you still see:
- **Explanatory text**: Model may need more explicit system prompt
- **Syntax errors**: May need to use a different model or add post-processing
- **Empty generation**: Check temperature/max_tokens settings

### Quick Fixes:

1. **Increase temperature** (more creative, less rigid):
   ```python
   config.temperature = 0.9  # Was 0.7
   ```

2. **Increase max_tokens** (allow longer generation):
   ```python
   config.max_new_tokens = 512  # Was 256
   ```

3. **Try different model** (if available):
   ```python
   model_name = "meta-llama/Llama-3.1-70B-Instruct"  # Larger model
   ```

## Files Modified

1. `reasoning/stages.py`: Updated all prompt templates
2. `models/generator.py`: Added pre-cleaning to remove explanatory text
3. `models/discriminator.py`: Added pre-cleaning to remove explanatory text

## Testing

To test the improvements:
```python
from models.generator import LLMGenerator
from reasoning.stages import get_stage

gen = LLMGenerator("meta-llama/Llama-3.1-8B-Instruct", "cuda")
stage5 = get_stage(5)

code = gen.generate_code(
    problem="Write a function to add two numbers",
    reasoning_chain=[],
    prompt_template=stage5.generator_prompt_template,
    max_new_tokens=256,
    temperature=0.7,
    function_signature="def add(a: int, b: int) -> int:"
)

print(f"Generated ({len(code)} chars):")
print(code)
```

Expected output:
```python
def add(a: int, b: int) -> int:
    return a + b
```

NOT:
```
This function adds two numbers together...

def add(a: int, b: int) -> int:
    return a + b
```

The cleaning logic will automatically remove the explanation!
