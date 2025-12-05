# Chat Template Fix for Llama-3.1-Instruct

## Problem

The model was generating conversational text instead of code:

**Expected**:
```python
def add(a, b):
    return a + b
```

**Actual**:
```
Please give me feedback on the above code, especially focusing on time and space complexities...
```

## Root Cause

Llama-3.1-Instruct is a **chat model** that expects prompts in a specific format with roles (system/user/assistant). We were passing raw text prompts, which confused the model into thinking it should have a conversation.

## Solution: Use Chat Template

Added proper chat template formatting for instruction-tuned models:

### Generator (`models/generator.py`)

```python
# Format prompt using chat template
if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
    messages = [
        {"role": "system", "content": "You are a Python code generator. Output ONLY valid Python code with no explanations."},
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = self.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
else:
    formatted_prompt = prompt  # Fallback for non-chat models
```

### Discriminator (`models/discriminator.py`)

```python
# Format prompt using chat template
if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
    messages = [
        {"role": "system", "content": "You are a test generator. Output ONLY valid pytest test functions with no explanations."},
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = self.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
else:
    formatted_prompt = prompt  # Fallback for non-chat models
```

## How Chat Templates Work

### Without Chat Template (Wrong):
```
Problem: Write a function to add two numbers

Write your Python function below:
```

The model sees this as incomplete text and tries to continue the conversation.

### With Chat Template (Correct):
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a Python code generator. Output ONLY valid Python code with no explanations.<|eot_id|><|start_header_id|>user<|end_header_id|>

Problem: Write a function to add two numbers

Write your Python function below:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

The model now understands:
1. It's an assistant
2. It should follow the system instruction (output only code)
3. It should respond to the user's request

## Expected Improvement

### Before (No Chat Template):
```
Generated Code (length=542):
  Please give me feedback on the above code...
  
Generated Tests (length=2004):
  Here are the responses from the reviewers:
  **Reviewer 1**
  * Great effort!
```
→ Result: 0/0 tests (no code generated)

### After (With Chat Template):
```
Generated Code (length=145):
  def add(a, b):
      return a + b
  
Generated Tests (length=234):
  import pytest
  
  def test_add():
      assert add(2, 3) == 5
```
→ Result: 3/5 tests (actual code and tests!)

## System Prompts

The system prompts are critical:

**Generator**: "You are a Python code generator. Output ONLY valid Python code with no explanations."

**Discriminator**: "You are a test generator. Output ONLY valid pytest test functions with no explanations."

These tell the model:
1. What role it's playing
2. What format to output
3. What NOT to do (no explanations)

## Compatibility

The code checks if the model supports chat templates:
```python
if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
    # Use chat template
else:
    # Fallback to raw prompt
```

This ensures compatibility with:
- ✅ Chat models (Llama-3.1-Instruct, Mistral-Instruct, etc.)
- ✅ Base models (Llama-3.1-Base, GPT-2, etc.)

## Testing

To verify the fix works:

```python
from models.generator import LLMGenerator

gen = LLMGenerator("meta-llama/Llama-3.1-8B-Instruct", "cuda")

# The chat template is now applied automatically
code = gen.generate_code(
    problem="Write a function to add two numbers",
    reasoning_chain=[],
    prompt_template="...",
    max_new_tokens=256,
    temperature=0.7,
    function_signature="def add(a: int, b: int) -> int:"
)

print(code)
```

Expected output:
```python
def add(a: int, b: int) -> int:
    return a + b
```

NOT:
```
Please give me feedback on the above code...
```

## Files Modified

1. `models/generator.py`: Added chat template formatting in `_generate()`
2. `models/discriminator.py`: Added chat template formatting in `_generate()`

## Impact

This is a **critical fix** that should dramatically improve generation quality:
- ✅ Model will output code instead of conversations
- ✅ Model will follow instructions better
- ✅ Rewards should increase from 0.0 to meaningful values
- ✅ Tests should actually execute (not 0/0)

This fix addresses the root cause of why the model was generating explanatory text instead of code!
