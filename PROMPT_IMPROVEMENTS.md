# Prompt Template Improvements

## Changes Made

Updated all prompt templates in `reasoning/stages.py` to enforce strict format adherence and improve output quality.

## Generator Prompt (Stage 5 - Code Generation)

### Before:
- Verbose instructions
- Unclear format expectations
- Mixed messaging about what to output

### After:
```
CRITICAL REQUIREMENTS:
1. Use the EXACT function signature provided above
2. Write ONLY the function implementation - no explanations, no comments outside the function
3. The function must be complete and executable
4. Return the computed result
5. Do NOT write multiple functions - only ONE function
6. Do NOT include test code or example usage
```

### Key Improvements:
- ✅ Explicit "ONLY the function" instruction
- ✅ Clear prohibition on multiple functions
- ✅ No test code or examples
- ✅ Must be executable
- ✅ Starts with code fence for easier extraction

## Discriminator Prompts (All Stages)

### Before:
- Vague instructions like "Generate 2-3 test cases"
- No format specification
- No examples

### After:
```
CRITICAL REQUIREMENTS:
1. Write ONLY pytest test functions - no explanations
2. Each test must start with "def test_"
3. Use assert statements to check correctness
4. Tests must be syntactically valid Python
5. [Stage-specific requirement]

Example format:
```python
import pytest

def test_basic():
    assert function_name(input) == expected_output
```
```

### Key Improvements:
- ✅ Explicit pytest format requirement
- ✅ Must start with "def test_"
- ✅ Must use assert statements
- ✅ Must be syntactically valid
- ✅ Example format provided
- ✅ Uses {num_tests} variable for consistency

## Stage-Specific Improvements

### Stage 1: Informal Reasoning Tests
**Focus**: Basic functionality
**Requirements**: Happy path, empty input, single element cases

### Stage 2: Structured Reasoning Tests
**Focus**: Edge cases
**Requirements**: Boundary conditions, identified edge cases

### Stage 3: Pseudocode Tests
**Focus**: Algorithmic correctness
**Requirements**: Loop boundaries, off-by-one errors, corner cases

### Stage 4: Constraints Tests
**Focus**: Constraint verification
**Requirements**: Constraint violations, stress tests, boundaries

### Stage 5: Final Code Tests
**Focus**: Adversarial testing
**Requirements**: Be adversarial, find bugs, test edge/corner cases, tricky inputs

## Expected Impact

### Reduced Empty Generation
**Before**: Models might generate explanations or nothing
**After**: Clear instructions to generate specific format

### Reduced Syntax Errors
**Before**: Models might generate invalid Python
**After**: Explicit requirement for "syntactically valid Python"

### Better Format Compliance
**Before**: Mixed output with explanations
**After**: Only code/tests, no explanations

### Improved Test Quality
**Before**: Vague "generate tests"
**After**: Specific requirements per stage (basic, edge, algorithmic, adversarial)

## Example Outputs

### Generator Output (Expected)
```python
def add(a, b):
    """Add two numbers."""
    return a + b
```

**NOT**:
```python
# Here's my solution:
def add(a, b):
    return a + b

# Example usage:
print(add(2, 3))  # Should output 5
```

### Discriminator Output (Expected)
```python
import pytest

def test_add_positive():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, -2) == -3

def test_add_zero():
    assert add(0, 5) == 5
```

**NOT**:
```python
Here are some test cases:

1. Test with positive numbers
2. Test with negative numbers

import pytest
def test_add():
    # This tests addition
    assert add(2, 3) == 5
```

## Monitoring

With the enhanced logging added earlier, you can now verify:

1. **Code length** - Should be 50-500 chars (not 0, not 5000)
2. **Test length** - Should be 100-800 chars (not 0, not 5000)
3. **Syntax errors** - Should decrease over time
4. **Format compliance** - Should see proper function definitions

## Testing the Prompts

To test if prompts work:

```python
from reasoning.stages import get_stage

# Test generator prompt
stage5 = get_stage(5)
print(stage5.generator_prompt_template.format(
    problem="Write a function to add two numbers",
    function_signature="def add(a: int, b: int) -> int:",
    previous_stages="[reasoning from previous stages]"
))

# Test discriminator prompt
print(stage5.discriminator_prompt_template.format(
    problem="Write a function to add two numbers",
    stage_output="def add(a, b): return a + b",
    num_tests=3
))
```

## Troubleshooting

### If models still generate explanations:
- Check if model is instruction-tuned (Llama-3.1-Instruct should work)
- Try increasing temperature slightly (0.7-0.9)
- Check if code extraction logic is working

### If models generate empty output:
- Check max_new_tokens is sufficient (>= 256)
- Check temperature is not too low (>= 0.5)
- Verify model is loaded correctly

### If tests are invalid:
- Check if pytest is imported
- Check if function names match
- Verify ground truth solution is correct

## Files Modified

- `reasoning/stages.py`: Updated all 5 stage prompt templates

## Next Steps

1. Run training with improved prompts
2. Monitor the detailed logs for format compliance
3. Check if syntax errors decrease
4. Verify rewards improve over time

The improved prompts should significantly reduce format-related issues and improve training quality!
