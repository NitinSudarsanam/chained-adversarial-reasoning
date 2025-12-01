# Test Format Comparison

## The Question
"show_model_output registers that there are tests, is run_training and run_unified_training parsing the same way?"

## Answer: NO, they use different formats!

### show_model_output.py
```python
# Uses: baseline_tests from problem (raw asserts)
test_code = "\n".join(problem.baseline_tests)
# Example:
# assert compress_string('aaabbc') == 'a3bbc'
# assert compress_string('abc') == 'abc'

# Uses: sandbox_simple.execute_tests_simple()
result = execute_tests_simple(code, test_code, timeout=5)
```

### Training Scripts
```python
# Uses: discriminator-generated tests (pytest functions)
tests = discriminator.generate_tests(...)
# Example:
# import pytest
# 
# def test_basic():
#     assert compress_string('aaabbc') == 'a3bbc'
# 
# def test_no_compression():
#     assert compress_string('abc') == 'abc'

# Uses: sandbox.Sandbox.execute_tests()
result = sandbox.execute_tests(code, tests)
```

## Key Differences

| Aspect | show_model_output | Training |
|--------|------------------|----------|
| Test Source | `problem.baseline_tests` | Discriminator generates |
| Test Format | Raw asserts | Pytest functions |
| Sandbox | `sandbox_simple` | `sandbox.Sandbox` |
| Parsing | Counts assert lines | Parses pytest output |

## Why This Matters

### Baseline Tests (show_model_output)
```python
# Simple, direct
assert compress_string('aaabbc') == 'a3bbc'
assert compress_string('abc') == 'abc'
```
- Easy to count: 2 tests
- Easy to execute: run each line
- Always works if code is valid

### Generated Tests (training)
```python
# More complex
import pytest

def test_basic():
    assert compress_string('aaabbc') == 'a3bbc'

def test_no_compression():
    assert compress_string('abc') == 'abc'
```
- Harder to count: need to parse pytest output
- Harder to execute: need pytest
- Can fail if:
  - Model generates invalid pytest syntax
  - Model doesn't close functions properly
  - Model generates comments instead of code

## The Problem with Zero Rewards

When you see:
```
Generator Reward: 0.0000
Discriminator Reward: 0.0000
```

It could mean:
1. **Generator generates broken code** → fails all tests
2. **Discriminator generates invalid tests** → `num_total = 0` or tests don't run
3. **Sandbox can't parse the tests** → thinks there are 0 tests

## Debugging

Run this to see what's actually happening:
```bash
python debug_test_execution.py
```

This will show you:
1. Do baseline tests work? (should be YES)
2. Do pytest tests work? (should be YES)
3. Is broken code detected? (should be YES)
4. Are empty tests handled? (should be YES)

If any of these fail, the sandbox has a bug.

## Most Likely Issue

The **discriminator is generating invalid test code**, so:
- Sandbox tries to run it
- Pytest fails to parse it
- `num_total = 0` (no tests found)
- Reward = 0.0

To verify, add this to your training script:
```python
print(f"Generated tests:\n{tests}\n")
print(f"Test result: {result.num_passed}/{result.num_total}")
```

You'll probably see something like:
```
Generated tests:
# Test case 1
# Test case 2

Test result: 0/0
```

The model is generating comments, not actual test code!

## Solution

1. **Improve discriminator prompt** (already done)
2. **Use larger model** (0.5B is too small)
3. **Add validation** - check if generated tests are valid before using them
4. **Fall back to baseline tests** - if discriminator fails, use problem.baseline_tests
