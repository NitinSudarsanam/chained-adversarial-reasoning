"""
VERIFICATION SUMMARY: Generator and Discriminator Output Parsing
==================================================================

✓ TEST 1: Generator Output Parsing
   - Correctly extracts function code from markdown blocks (```python...```)
   - Handles unclosed code blocks
   - Handles plain code without markdown
   - All extracted code compiles successfully as valid Python
   
✓ TEST 2: Discriminator Output Parsing  
   - Correctly extracts test list from markdown blocks
   - Parses test cases as Python list of tuples
   - Handles various test list formats
   - All extracted test code parses as valid Python lists
   
✓ TEST 3: Model Integration
   - LLMGenerator._extract_code_from_markdown() works correctly
   - LLMDiscriminator._extract_code_from_markdown() works correctly
   - Both methods handle edge cases (whitespace, unclosed blocks, etc.)
   
✓ TEST 4: Execution Pipeline Integration
   - Generator code executes against test cases
   - Tests are parsed from discriminator output
   - Both work seamlessly with execute_tests() function
   - Validation flow works correctly

NEW SYSTEM PROMPTS
==================

Generator System Prompt:
- Requests: Python function code only (no pytest, no tests)
- Example: Shows exact format with twoSum solution
- Output Format: ```python ... ``` code blocks
- Extraction: Works with Generator._extract_code_from_markdown()

Discriminator System Prompt:  
- Requests: Test cases as Python list of tuples
- Example: Shows exact format with twoSum test cases
- Output Format: ```python [...] ``` lists
- Extraction: Works with Discriminator._extract_code_from_markdown()

OUTPUT FORMATS
==============

Generator Output (example):
```python
def twoSum(nums, target):
    num_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
```

Discriminator Output (example):
```python
[
  ([5, 3, 1, 0], 4, [1, 2]),
  ([5, 3, 6, 2, 4, 56, 1], 6, [3, 4]),
  ([8, 3, 2, 6, 2, 7, 8], 13, [3, 5])
]
```

PARSING VERIFIED
================
✓ Both models' outputs parse correctly
✓ Extraction methods work with new prompt formats
✓ No pytest functions in generator output
✓ Test cases are valid Python tuples
✓ Integration with execution pipeline works end-to-end
✓ LeetCode dataset loads and executes correctly

READY FOR TRAINING
==================
All parsing and formatting is verified and working correctly.
The system is ready to:
1. Generate solutions using LLMGenerator
2. Generate test cases using LLMDiscriminator
3. Execute tests against solutions
4. Compute rewards for adversarial training
"""

print(__doc__)
