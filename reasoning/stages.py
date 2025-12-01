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
        discriminator_prompt_template="""You are generating basic test cases based on informal reasoning for a coding problem.

Problem: {problem}

Informal Reasoning:
{stage_output}

Generate 2-3 basic test cases as pytest functions that check if the core idea works. Focus on: happy path, empty input, single element cases.

Test Cases:
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
        discriminator_prompt_template="""You are generating edge case test cases based on structured reasoning for a coding problem.

Problem: {problem}

Structured Reasoning:
{stage_output}

Generate 2-3 test cases as pytest functions for edge cases mentioned in the structure. Focus on: boundary conditions, edge cases identified in the reasoning.

Test Cases:
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
        discriminator_prompt_template="""You are generating algorithmic test cases based on pseudocode for a coding problem.

Problem: {problem}

Pseudocode:
{stage_output}

Generate 2-3 test cases as pytest functions that could break this algorithm. Focus on: loop boundaries, off-by-one errors, algorithmic corner cases.

Test Cases:
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
        discriminator_prompt_template="""You are generating constraint-testing test cases based on stated constraints for a coding problem.

Problem: {problem}

Constraints and Invariants:
{stage_output}

Generate 2-3 test cases as pytest functions that verify the stated constraints are met. Focus on: constraint violations, complexity stress tests, stated assumptions.

Test Cases:
```python
import pytest

"""
    ),
    
    ReasoningStage(
        id=5,
        name="Executable Code",
        description="Final Python implementation",
        generator_prompt_template="""You are a Python expert. Complete this function with working code.

Example of a complete function:
def add_numbers(a: int, b: int) -> int:
    result = a + b
    return result

Now implement this function:

{function_signature}
    pass  # Replace this with your implementation

Problem: {problem}

Previous reasoning: {previous_stages}

Write the complete function with the implementation (replace 'pass' with actual code):
""",
        discriminator_prompt_template="""You are generating test cases for a coding problem and solution.

Problem: {problem}

Generated Code:
{stage_output}

Generate {num_tests} challenging test cases as pytest functions. Make them adversarial - try to break the code.

Test Cases:
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
