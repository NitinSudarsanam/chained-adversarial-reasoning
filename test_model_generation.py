"""Test what the model actually generates."""

from models.unified_model import UnifiedModel
from data.problem_dataset import load_problems
from reasoning.stages import get_stage

# Load model
print("Loading model...")
model = UnifiedModel("Qwen/Qwen2.5-Coder-0.5B-Instruct", device="cuda")
print()

# Load a problem
problems = load_problems("data/example_problems.json")
problem = problems[0]

print("="*80)
print(f"Problem: {problem.description[:100]}...")
print(f"Signature: {problem.function_signature}")
print("="*80)

# Test stage 5 generation
stage = get_stage(5)
prompt = stage.generator_prompt_template.format(
    problem=problem.description,
    function_signature=problem.function_signature,
    previous_stages="None"
)

print("\nPROMPT:")
print("-"*80)
print(prompt)
print("-"*80)

print("\nGenerating code...")
code = model.generate_code(
    problem=problem.description,
    reasoning_chain=[],
    prompt_template=stage.generator_prompt_template,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    function_signature=problem.function_signature
)

print("\nGENERATED CODE:")
print("-"*80)
print(code)
print("-"*80)

# Check if it's valid
if not code or len(code) < 20:
    print("\n❌ Code is too short or empty!")
elif "pass" in code and code.count("\n") < 3:
    print("\n❌ Code only contains 'pass'!")
elif problem.function_signature.split("(")[0].split()[-1] not in code:
    print(f"\n❌ Function name not found in code!")
else:
    print("\n✓ Code looks reasonable")

# Test discriminator generation
print("\n" + "="*80)
print("Testing discriminator (test generation)...")
print("="*80)

tests = model.generate_tests(
    problem=problem.description,
    generator_code=code,
    num_tests=3,
    prompt_template=stage.discriminator_prompt_template,
    max_new_tokens=256,
    temperature=0.8
)

print("\nGENERATED TESTS:")
print("-"*80)
print(tests)
print("-"*80)

if not tests or len(tests) < 20:
    print("\n❌ Tests are too short or empty!")
elif "def test_" not in tests:
    print("\n❌ No test functions found!")
else:
    print("\n✓ Tests look reasonable")
