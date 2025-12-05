# All Cases Where Rewards Can Be Zero

## Generator Reward Cases

The generator reward formula is:
```python
reward = num_passed / num_total  # Pass rate
```

### Case 1: No Tests Generated (num_total = 0)
**Condition**: `execution_result.num_total == 0`
**Reward**: `0.0`
**Why**: Discriminator failed to generate any tests
**Example**:
```
Generator code: "def add(a, b): return a + b"
Tests: "" (empty)
Result: num_passed=0, num_total=0
Reward: 0.0
```

### Case 2: Execution Timeout
**Condition**: `execution_result.timed_out == True`
**Reward**: `0.0`
**Why**: Code took too long to execute (infinite loop, etc.)
**Example**:
```python
def add(a, b):
    while True:  # Infinite loop
        pass
    return a + b
```

### Case 3: All Tests Failed (num_passed = 0)
**Condition**: `num_passed == 0` and `num_total > 0`
**Reward**: `0.0 / num_total = 0.0`
**Why**: Generated code is incorrect and fails all tests
**Example**:
```
Generator code: "def add(a, b): return a - b"  # Wrong!
Tests: "assert add(2, 3) == 5"
Result: num_passed=0, num_total=1
Reward: 0.0
```

---

## Discriminator Reward Cases

The discriminator reward formula is:
```python
adversarial_score = 1.0 - generator_pass_rate
reward = adversarial_score * test_validity
```

### Case 1: No Tests Generated
**Condition**: `generator_result.num_total == 0` AND `validation_result.num_total == 0`
**Reward**: `(1.0 - 0.0) * 0.0 = 0.0`
**Why**: Discriminator generated no tests
**Example**:
```
Tests: "" (empty)
Generator result: num_passed=0, num_total=0
Validation result: num_passed=0, num_total=0
Reward: 0.0
```

### Case 2: All Tests Invalid (test_validity = 0)
**Condition**: `validation_result.num_passed == 0` and `validation_result.num_total > 0`
**Reward**: `adversarial_score * 0.0 = 0.0`
**Why**: All tests fail against ground truth (invalid tests)
**Example**:
```python
# Ground truth
def add(a, b):
    return a + b

# Invalid test (wrong assertion)
assert add(2, 3) == 6  # Should be 5!

# Result
validation_result: num_passed=0, num_total=1
test_validity = 0.0
Reward: 0.0 (regardless of adversarial_score)
```

### Case 3: Generator Passes All Tests (generator_pass_rate = 1.0)
**Condition**: `generator_result.num_passed == generator_result.num_total`
**Reward**: `(1.0 - 1.0) * test_validity = 0.0`
**Why**: Tests are too easy, generator passes everything
**Example**:
```python
# Generator code
def add(a, b):
    return a + b

# Easy test
assert add(2, 3) == 5

# Result
generator_result: num_passed=1, num_total=1
adversarial_score = 1.0 - 1.0 = 0.0
Reward: 0.0 (even if tests are valid)
```

### Case 4: Combination - Generator Passes All AND Tests Invalid
**Condition**: Both above conditions
**Reward**: `0.0 * 0.0 = 0.0`
**Why**: Worst case - tests are both easy and invalid

---

## Summary Table

| Scenario | Generator Reward | Discriminator Reward | Common Cause |
|----------|-----------------|---------------------|--------------|
| No tests generated | 0.0 | 0.0 | Discriminator failure |
| Code timeout | 0.0 | varies | Infinite loop in code |
| All tests failed | 0.0 | varies | Wrong code |
| All tests invalid | varies | 0.0 | Bad test generation |
| Generator passes all | varies | 0.0 | Tests too easy |
| Code syntax error | 0.0 | varies | Malformed code |
| Test syntax error | 0.0 | 0.0 | Malformed tests |

---

## Detailed Scenarios

### Scenario A: Empty Generation (Most Common)
```
Step 1/10 - Discriminator Reward: 0.0000 (gen_passed=0/0, val_passed=0/0)
Step 2/10 - Generator Reward: 0.0000 (passed=0/0)
```
**Diagnosis**: Models are generating empty strings
**Causes**:
- Models not loaded properly
- LoRA adapters not active
- Temperature too low (model too conservative)
- Max tokens too low (generation cut off)

### Scenario B: Invalid Tests
```
Step 1/10 - Discriminator Reward: 0.0000 (gen_passed=3/5, val_passed=0/5)
```
**Diagnosis**: Tests fail against ground truth
**Causes**:
- Discriminator generates wrong assertions
- Tests have syntax errors
- Tests don't match problem requirements

### Scenario C: Tests Too Easy
```
Step 1/10 - Discriminator Reward: 0.0000 (gen_passed=5/5, val_passed=5/5)
```
**Diagnosis**: Generator passes all tests (adversarial_score = 0)
**Causes**:
- Tests are trivial (e.g., `assert True`)
- Tests don't cover edge cases
- Generator code is actually correct

### Scenario D: Code Always Fails
```
Step 1/10 - Generator Reward: 0.0000 (passed=0/10)
Step 2/10 - Generator Reward: 0.0000 (passed=0/8)
```
**Diagnosis**: Generator never passes any tests
**Causes**:
- Generator produces syntactically invalid code
- Generator produces semantically wrong code
- Tests are too hard (unrealistic)

---

## How to Diagnose Your Zero Rewards

### Step 1: Check the Detailed Logs

Look at the test counts in parentheses:

**Pattern 1**: `(gen_passed=0/0, val_passed=0/0)`
→ **No tests generated** - discriminator issue

**Pattern 2**: `(gen_passed=X/Y, val_passed=0/Y)` where Y > 0
→ **Invalid tests** - discriminator generates wrong tests

**Pattern 3**: `(gen_passed=Y/Y, val_passed=Y/Y)` where Y > 0
→ **Tests too easy** - discriminator not adversarial enough

**Pattern 4**: `(passed=0/Y)` where Y > 0
→ **Code always fails** - generator issue

### Step 2: Check for Skipped Steps

```
⚠ Skipping discriminator step 1/10: empty code generated
```
→ Generation is failing completely

### Step 3: Manual Testing

Test generation manually:
```python
# Test generator
code = generator.generate_code(
    problem="Write a function to add two numbers",
    reasoning_chain=[],
    prompt_template="...",
    max_new_tokens=200
)
print(f"Generated code: '{code}'")
print(f"Length: {len(code)}")

# Test discriminator
tests = discriminator.generate_tests(
    problem="Write a function to add two numbers",
    generator_code="def add(a, b): return a + b",
    num_tests=3
)
print(f"Generated tests: '{tests}'")
print(f"Length: {len(tests)}")
```

---

## Solutions by Case

### For "No Tests Generated" (0/0)
1. Check discriminator model is loaded
2. Increase `max_new_tokens` (try 512+)
3. Increase `temperature` (try 0.8-1.0)
4. Check prompt templates are correct
5. Verify LoRA adapters are active

### For "Invalid Tests" (val_passed=0/Y)
1. Check ground truth solutions are correct
2. Improve discriminator prompt template
3. Add examples of valid tests to prompt
4. Increase discriminator training steps
5. Check test parsing/extraction logic

### For "Tests Too Easy" (gen_passed=Y/Y)
1. Improve discriminator prompt to be more adversarial
2. Add "edge case" instructions to prompt
3. Increase test difficulty in prompt
4. Train discriminator longer
5. Add reward shaping to penalize easy tests

### For "Code Always Fails" (passed=0/Y)
1. Check generator model is loaded
2. Improve generator prompt template
3. Add code examples to prompt
4. Check if tests are realistic
5. Verify code extraction logic works

---

## Expected Reward Ranges

### Early Training (Stage 1, Steps 1-10)
- **Generator**: 0.1 - 0.4 (10-40% pass rate)
- **Discriminator**: 0.2 - 0.6 (some adversarial success)

### Mid Training (Stage 2-3)
- **Generator**: 0.3 - 0.6 (improving)
- **Discriminator**: 0.3 - 0.7 (balanced competition)

### Late Training (Stage 4-5)
- **Generator**: 0.5 - 0.8 (good code quality)
- **Discriminator**: 0.2 - 0.5 (harder to fool generator)

### Red Flags
- **All zeros**: Something is broken
- **All ones**: Tests are trivial or code is perfect (unlikely)
- **No variation**: Model not learning

---

## Quick Diagnostic Checklist

- [ ] Models loaded successfully (check console output)
- [ ] LoRA adapters active (check model.peft_config exists)
- [ ] Generation produces non-empty output (manual test)
- [ ] Tests are syntactically valid (manual inspection)
- [ ] Ground truth solutions are correct (verify)
- [ ] Sandbox execution works (test manually)
- [ ] Prompt templates are reasonable (review)
- [ ] Config parameters are sensible (max_tokens > 100, temp > 0.5)

If all checks pass but rewards are still zero, the issue is likely in the training dynamics (models need more training or better prompts).
