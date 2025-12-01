"""Simple script to see what the model generates."""

from models.generator import LLMGenerator
from reasoning.stages import get_stage
from data.problem_dataset import load_problems

# Load model
print("Loading model (takes ~30 seconds)...")
generator = LLMGenerator(model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct", device="cuda")

# Load first problem (use function-based problems)
problems = load_problems("data/function_problems.json")
problem = problems[0]

print(f"\n{'='*80}")
print(f"PROBLEM: {problem.id}")
print(f"{'='*80}")
print(f"\nDescription:\n{problem.description}")
print(f"\nRequired Signature:\n{problem.function_signature}")

# Get the stage 5 prompt
stage_5 = get_stage(5)

# Show what prompt the model sees
prompt = stage_5.generator_prompt_template.format(
    problem=problem.description,
    function_signature=problem.function_signature,
    previous_stages="None"
)

print(f"\n{'='*80}")
print("PROMPT SENT TO MODEL:")
print(f"{'='*80}")
print(prompt)
print(f"{'='*80}")

# Generate code
print("\nGenerating code (takes ~30-60 seconds on CPU)...\n")
code = generator.generate_code(
    problem=problem.description,
    reasoning_chain=[],
    prompt_template=stage_5.generator_prompt_template,
    max_new_tokens=512,  # Increased from 256
    temperature=0.7,
    function_signature=problem.function_signature
)

print(f"{'='*80}")
print("MODEL OUTPUT (after cleaning):")
print(f"{'='*80}")
print(code)
print(f"{'='*80}")

# Test it
from sandbox.sandbox_simple import execute_tests_simple
test_code = "\n".join(problem.baseline_tests)
result = execute_tests_simple(code, test_code, timeout=5)

print(f"\nTest Results: {result.num_passed}/{result.num_total} passed")
if not result.passed:
    print(f"Errors: {result.errors[:3]}")
