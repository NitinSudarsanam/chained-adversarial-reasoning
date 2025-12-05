# Debugging Zero Rewards - Enhanced Logging

## What Was Added

Enhanced logging after each training step to show:

1. **Generated Code** - First 300 characters of the code produced
2. **Generated Tests** - First 300 characters of the tests produced
3. **Execution Results** - Timeout status, errors, stderr output
4. **Reward Breakdown** - Detailed test pass/fail counts

## Example Output

### Successful Step (Non-Zero Reward)
```
Step 1/10 - Discriminator Reward: 0.4500 (gen_passed=2/5, val_passed=4/5)
  Generated Code (length=145):
    def add(a, b):
        """Add two numbers."""
        return a + b
    
    def multiply(a, b):
        return a * b
  Generated Tests (length=234):
    import pytest
    
    def test_add_positive():
        assert add(2, 3) == 5
    
    def test_add_negative():
        assert add(-1, -2) == -3
  Execution Results:
    Timeout: False
    Errors: 0 error(s)
```

### Failed Step - No Tests Generated
```
Step 1/10 - Discriminator Reward: 0.0000 (gen_passed=0/0, val_passed=0/0)
  Generated Code (length=145):
    def add(a, b):
        return a + b
  Generated Tests (length=0):
    
  Execution Results:
    Timeout: False
    Errors: 0 error(s)
```
**Diagnosis**: Discriminator generated empty tests → `num_total=0` → reward=0

### Failed Step - Invalid Tests
```
Step 1/10 - Discriminator Reward: 0.0000 (gen_passed=3/5, val_passed=0/5)
  Generated Code (length=145):
    def add(a, b):
        return a + b
  Generated Tests (length=234):
    import pytest
    
    def test_add():
        assert add(2, 3) == 6  # WRONG! Should be 5
  Execution Results:
    Timeout: False
    Errors: 1 error(s)
    First error: AssertionError: assert 5 == 6
```
**Diagnosis**: Tests have wrong assertions → fail against ground truth → `val_passed=0/5` → reward=0

### Failed Step - Code Syntax Error
```
Step 1/10 - Generator Reward: 0.0000 (passed=0/5)
  Generated Code (length=89):
    def add(a, b)
        return a + b  # Missing colon!
  Generated Tests (length=234):
    import pytest
    
    def test_add():
        assert add(2, 3) == 5
  Execution Results:
    Timeout: False
    Errors: 1 error(s)
    First error: SyntaxError: invalid syntax
    Stderr: File "<string>", line 1
```
**Diagnosis**: Code has syntax error → can't execute → `passed=0/5` → reward=0

### Failed Step - Code Timeout
```
Step 1/10 - Generator Reward: 0.0000 (passed=0/5)
  Generated Code (length=145):
    def add(a, b):
        while True:  # Infinite loop!
            pass
        return a + b
  Generated Tests (length=234):
    import pytest
    
    def test_add():
        assert add(2, 3) == 5
  Execution Results:
    Timeout: True
    Errors: 0 error(s)
```
**Diagnosis**: Code has infinite loop → timeout → reward=0

---

## How to Use This Information

### Step 1: Run Training
Run your training script and watch the console output.

### Step 2: Identify the Pattern

Look at the logged output for each step:

#### Pattern A: Empty Code
```
Generated Code (length=0):
    
```
→ **Problem**: Generator not producing output
→ **Fix**: Check generator model, increase max_tokens, check prompts

#### Pattern B: Empty Tests
```
Generated Tests (length=0):
    
```
→ **Problem**: Discriminator not producing output
→ **Fix**: Check discriminator model, increase max_tokens, check prompts

#### Pattern C: Very Short Output
```
Generated Code (length=15):
    def add(a, b
```
→ **Problem**: Generation cut off too early
→ **Fix**: Increase `max_new_tokens` in config

#### Pattern D: Syntax Errors
```
Errors: 1 error(s)
First error: SyntaxError: invalid syntax
```
→ **Problem**: Model generating invalid Python
→ **Fix**: Improve prompts, add examples, check model quality

#### Pattern E: Wrong Logic
```
gen_passed=0/5, val_passed=5/5
```
→ **Problem**: Code is syntactically valid but logically wrong
→ **Fix**: This is expected early in training, should improve over time

#### Pattern F: Invalid Tests
```
gen_passed=5/5, val_passed=0/5
```
→ **Problem**: Tests fail against ground truth
→ **Fix**: Improve discriminator prompts, check ground truth is correct

### Step 3: Take Action

Based on the pattern, apply the appropriate fix:

1. **Empty generation** → Check model loading, increase tokens
2. **Syntax errors** → Improve prompts, add examples
3. **Wrong logic** → Continue training, model needs to learn
4. **Invalid tests** → Fix discriminator prompts or ground truth

---

## Common Issues and Solutions

### Issue 1: Both Code and Tests Empty
```
Generated Code (length=0):
Generated Tests (length=0):
```

**Likely Causes**:
- Models not loaded properly
- LoRA adapters not active
- Temperature too low (0.0)
- Max tokens too low (< 50)

**Solutions**:
```python
# Check models are loaded
print(f"Generator params: {sum(p.numel() for p in generator.parameters())}")
print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters())}")

# Increase generation parameters
config.max_new_tokens = 512  # Was too low
config.temperature = 0.8     # Was too low
```

### Issue 2: Code Generated But Tests Empty
```
Generated Code (length=145): [valid code]
Generated Tests (length=0):
```

**Likely Causes**:
- Discriminator model issue
- Discriminator prompt template broken
- Discriminator max_tokens too low

**Solutions**:
```python
# Test discriminator manually
tests = discriminator.generate_tests(
    problem="Write a function to add two numbers",
    generator_code="def add(a, b): return a + b",
    num_tests=3,
    max_new_tokens=512
)
print(f"Manual test generation: '{tests}'")
```

### Issue 3: Tests Generated But All Invalid
```
Generated Tests (length=234): [tests with wrong assertions]
gen_passed=X/Y, val_passed=0/Y
```

**Likely Causes**:
- Discriminator generating wrong assertions
- Ground truth solution is wrong
- Test extraction logic broken

**Solutions**:
```python
# Check ground truth
print(f"Ground truth: {problem.reference_solution}")

# Manually validate a test
from sandbox.sandbox import Sandbox
sandbox = Sandbox()
result = sandbox.validate_tests_against_solution(
    tests="assert add(2, 3) == 5",
    solution="def add(a, b): return a + b"
)
print(f"Validation: {result.num_passed}/{result.num_total}")
```

### Issue 4: Syntax Errors in Generated Code
```
Errors: 1 error(s)
First error: SyntaxError: invalid syntax
```

**Likely Causes**:
- Model generating incomplete code
- Code extraction logic broken
- Model not trained for Python

**Solutions**:
```python
# Check code extraction
from models.generator import LLMGenerator
gen = LLMGenerator(model_name, device)
raw_output = gen._generate(prompt, max_new_tokens=512, temperature=0.7, top_p=0.9)
print(f"Raw output: {raw_output}")
extracted = gen._extract_code_from_markdown(raw_output)
print(f"Extracted: {extracted}")
```

---

## Monitoring During Training

Watch for these trends:

### Good Signs ✅
- Code length increasing over time
- Test length increasing over time
- Fewer syntax errors over time
- Rewards gradually increasing
- `gen_passed` and `val_passed` both > 0

### Bad Signs ❌
- Consistently empty generation
- Rewards stuck at 0.0 for many steps
- Same errors repeating
- `num_total` always 0
- Timeouts on every step

---

## Quick Diagnostic Commands

### Check Model Loading
```python
# In Python console
from models.generator import LLMGenerator
gen = LLMGenerator("meta-llama/Llama-3.1-8B-Instruct", "cuda")
print(f"Model loaded: {gen.model is not None}")
print(f"Has LoRA: {hasattr(gen.model, 'peft_config')}")
```

### Test Generation
```python
output = gen.generate_stage_output(
    problem="Write a function to add two numbers",
    previous_stages=[],
    stage_id=1,
    prompt_template="Problem: {problem}\n\nSolution:",
    max_new_tokens=200,
    temperature=0.7
)
print(f"Generated ({len(output)} chars): {output}")
```

### Test Execution
```python
from sandbox.sandbox import Sandbox
sandbox = Sandbox()
result = sandbox.execute_tests(
    code="def add(a, b): return a + b",
    tests="assert add(2, 3) == 5"
)
print(f"Result: {result.num_passed}/{result.num_total}")
```

---

## Expected Output Patterns

### Early Training (Steps 1-5)
- Code: 50-200 characters (simple functions)
- Tests: 100-300 characters (basic assertions)
- Rewards: 0.1-0.4 (low but non-zero)
- Errors: Common (model learning)

### Mid Training (Steps 10-20)
- Code: 100-400 characters (more complex)
- Tests: 200-500 characters (multiple tests)
- Rewards: 0.3-0.6 (improving)
- Errors: Less frequent

### Late Training (Steps 30+)
- Code: 200-600 characters (full solutions)
- Tests: 300-800 characters (comprehensive)
- Rewards: 0.5-0.8 (good performance)
- Errors: Rare

---

## Files Modified

- `training/adversarial_trainer.py`: Added detailed logging after each step

## Next Steps

1. Run training with enhanced logging
2. Examine the console output for each step
3. Identify which pattern matches your issue
4. Apply the appropriate solution
5. Re-run and verify improvement

The detailed logging will make it immediately obvious what's wrong!
