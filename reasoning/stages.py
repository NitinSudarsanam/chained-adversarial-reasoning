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
        discriminator_prompt_template="""Generate test cases for this problem. Output ONLY a Python list of tuples.

PROBLEM: {problem}

INFORMAL REASONING:
{stage_output}

Generate {num_tests} basic test cases (happy path, empty input, single element).

Format: [(input_args, expected_output), ...]
DO NOT write pytest functions. DO NOT write solution code. ONLY the list.

```python
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
        discriminator_prompt_template="""Generate test cases for this problem. Output ONLY a Python list of tuples.

PROBLEM: {problem}

STRUCTURED REASONING:
{stage_output}

Generate {num_tests} edge case test cases (boundary conditions, edge cases).

Format: [(input_args, expected_output), ...]
DO NOT write pytest functions. DO NOT write solution code. ONLY the list.

```python
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
        discriminator_prompt_template="""Generate test cases for this problem. Output ONLY a Python list of tuples.

PROBLEM: {problem}

PSEUDOCODE:
{stage_output}

Generate {num_tests} algorithmic test cases (loop boundaries, off-by-one errors, corner cases).

Format: [(input_args, expected_output), ...]
DO NOT write pytest functions. DO NOT write solution code. ONLY the list.

```python
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
        discriminator_prompt_template="""Generate test cases for this problem. Output ONLY a Python list of tuples.

PROBLEM: {problem}

CONSTRAINTS AND INVARIANTS:
{stage_output}

Generate {num_tests} constraint-testing test cases (constraint violations, stress tests).

Format: [(input_args, expected_output), ...]
DO NOT write pytest functions. DO NOT write solution code. ONLY the list.

```python
"""
    ),
    
    ReasoningStage(
        id=5,
        name="Executable Code",
        description="Final Python implementation",
        generator_prompt_template="""PROBLEM DESCRIPTION: {problem}

FUNCTION SIGNATURE: {function_signature}

Previous reasoning:
{previous_stages}

YOUR RESPONSE:
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
        discriminator_prompt_template="""Generate test cases for this problem. Output ONLY a Python list of tuples, nothing else.

PROBLEM: {problem}

CODE TO TEST:
{stage_output}

Generate {num_tests} test cases in this exact format:
[(input1, input2, ..., expected_output), ...]

DO NOT write solution code. DO NOT write imports. ONLY the test list.

```python
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
