"""Multi-stage reasoning pipeline definitions."""

from dataclasses import dataclass
from typing import List


@dataclass
class ReasoningStage:
    """Represents one stage in the reasoning pipeline."""
    id: int
    name: str
    description: str
    generator_prompt_template: str
    discriminator_prompt_template: str


# Define the five reasoning stages
REASONING_STAGES = [
    ReasoningStage(
        id=1,
        name="Informal Reasoning",
        description="High-level intuitive understanding of the problem",
        generator_prompt_template="""You are solving a coding problem. First, provide informal reasoning about the problem.

Problem: {problem}

Provide your informal reasoning - explain what the problem is asking, what approach you might take, and any initial thoughts. Be conversational and intuitive.

Informal Reasoning:""",
        discriminator_prompt_template="""You are generating test cases for a coding problem.

Problem: {problem}

Informal Reasoning:
{stage_output}

Generate {num_tests} basic test cases as pytest functions. Focus on: happy path, empty input, single element.

CRITICAL REQUIREMENTS:
1. Write ONLY pytest test functions - no explanations
2. Each test must start with "def test_"
3. Use assert statements to check correctness
4. Tests must be syntactically valid Python
5. Import any needed modules at the top

Example format:
```python
import pytest

def test_basic():
    assert function_name(input) == expected_output

def test_empty():
    assert function_name([]) == expected_empty_result
```

Write your test functions below:

```python
import pytest

"""
    ),
    
    ReasoningStage(
        id=2,
        name="Structured Reasoning",
        description="Organized breakdown of the problem with clear steps",
        generator_prompt_template="""You are solving a coding problem. You've done informal reasoning. Now provide structured reasoning.

Problem: {problem}

Previous Reasoning:
{previous_stages}

Provide structured reasoning with:
1. Problem breakdown
2. Key observations
3. Approach steps
4. Edge cases to consider

Structured Reasoning:""",
        discriminator_prompt_template="""You are generating edge case test cases for a coding problem.

Problem: {problem}

Structured Reasoning:
{stage_output}

Generate {num_tests} edge case test cases as pytest functions. Focus on: boundary conditions, edge cases.

CRITICAL REQUIREMENTS:
1. Write ONLY pytest test functions - no explanations
2. Each test must start with "def test_"
3. Use assert statements to check correctness
4. Tests must be syntactically valid Python
5. Test edge cases and boundaries

Write your test functions below:

```python
import pytest

"""
    ),
    
    ReasoningStage(
        id=3,
        name="Pseudocode",
        description="Algorithm expressed in pseudocode notation",
        generator_prompt_template="""You are solving a coding problem. You've done informal and structured reasoning. Now write pseudocode.

Problem: {problem}

Previous Reasoning:
{previous_stages}

Write clear pseudocode for the solution. Use indentation and clear variable names.

Pseudocode:""",
        discriminator_prompt_template="""You are generating algorithmic test cases for a coding problem.

Problem: {problem}

Pseudocode:
{stage_output}

Generate {num_tests} test cases as pytest functions that test algorithmic correctness. Focus on: loop boundaries, off-by-one errors, corner cases.

CRITICAL REQUIREMENTS:
1. Write ONLY pytest test functions - no explanations
2. Each test must start with "def test_"
3. Use assert statements to check correctness
4. Tests must be syntactically valid Python
5. Try to find algorithmic bugs

Write your test functions below:

```python
import pytest

"""
    ),
    
    ReasoningStage(
        id=4,
        name="Constraints and Invariants",
        description="Explicit constraints, invariants, and correctness conditions",
        generator_prompt_template="""You are solving a coding problem. You've developed reasoning and pseudocode. Now specify constraints and invariants.

Problem: {problem}

Previous Reasoning and Pseudocode:
{previous_stages}

List:
1. Input constraints
2. Output constraints
3. Loop invariants
4. Pre/post conditions
5. Time/space complexity

Constraints and Invariants:""",
        discriminator_prompt_template="""You are generating constraint-testing test cases for a coding problem.

Problem: {problem}

Constraints and Invariants:
{stage_output}

Generate {num_tests} test cases as pytest functions that verify constraints. Focus on: constraint violations, stress tests.

CRITICAL REQUIREMENTS:
1. Write ONLY pytest test functions - no explanations
2. Each test must start with "def test_"
3. Use assert statements to check correctness
4. Tests must be syntactically valid Python
5. Test constraint boundaries

Write your test functions below:

```python
import pytest

"""
    ),
    
    ReasoningStage(
        id=5,
        name="Executable Code",
        description="Final Python implementation",
        generator_prompt_template="""You are an expert Python programmer solving coding problems.

Problem: {problem}

Function Signature: {function_signature}

CRITICAL REQUIREMENTS:
1. Use the EXACT function signature provided above
2. Write ONLY the function implementation - no explanations, no comments outside the function
3. The function must be complete and executable
4. Return the computed result
5. Do NOT write multiple functions - only ONE function
6. Do NOT include test code or example usage

Previous reasoning:
{previous_stages}

Write your Python function below:

```python
"""
        
#         """You are a Python coding assistant. Write a complete, working Python function.

# IMPORTANT: 
# - Keep the EXACT function signature provided
# - Write actual working code, not comments or placeholders
# - Return the result

# Function to implement:
# {function_signature}

# Problem description:
# {problem}

# Write the complete function below (with actual code, not 'pass'):

# ```python
# """
,
        discriminator_prompt_template="""You are generating adversarial test cases for a coding problem.

Problem: {problem}

Generated Code:
{stage_output}

Generate {num_tests} challenging test cases as pytest functions. Make them adversarial - try to find bugs.

CRITICAL REQUIREMENTS:
1. Write ONLY pytest test functions - no explanations
2. Each test must start with "def test_"
3. Use assert statements to check correctness
4. Tests must be syntactically valid Python
5. Be adversarial - try to break the code
6. Test edge cases, corner cases, and tricky inputs

Write your test functions below:

```python
import pytest

"""
    )
]


def get_stage(stage_id: int) -> ReasoningStage:
    """Get reasoning stage by ID.
    
    Args:
        stage_id: Stage ID (1-5)
        
    Returns:
        ReasoningStage object
        
    Raises:
        ValueError: If stage_id is invalid
    """
    if stage_id < 1 or stage_id > 5:
        raise ValueError(f"Invalid stage_id: {stage_id}. Must be 1-5.")
    return REASONING_STAGES[stage_id - 1]


def get_all_stages() -> List[ReasoningStage]:
    """Get all reasoning stages in order.
    
    Returns:
        List of all ReasoningStage objects
    """
    return REASONING_STAGES.copy()
