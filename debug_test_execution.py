"""Debug script to see what's happening with test execution."""

from sandbox.sandbox import Sandbox
from data.problem_dataset import load_problems

# Load a problem
problems = load_problems("data/function_problems.json")
problem = problems[0]

print("="*80)
print("PROBLEM:", problem.id)
print("="*80)

# Test with baseline tests (raw asserts)
print("\n1. Testing with BASELINE TESTS (raw asserts):")
print("-"*80)
baseline_tests_str = "\n".join(problem.baseline_tests)
print("Tests:")
print(baseline_tests_str)
print("-"*80)

sandbox = Sandbox(timeout=5)
result = sandbox.execute_tests(problem.reference_solution, baseline_tests_str)

print(f"\nResult:")
print(f"  Passed: {result.passed}")
print(f"  num_passed: {result.num_passed}")
print(f"  num_total: {result.num_total}")
print(f"  Errors: {result.errors[:3] if result.errors else 'None'}")

# Test with pytest-style tests
print("\n" + "="*80)
print("2. Testing with PYTEST-STYLE TESTS:")
print("-"*80)
pytest_tests = """import pytest

def test_basic():
    assert compress_string('aaabbc') == 'a3bbc'

def test_no_compression():
    assert compress_string('abc') == 'abc'

def test_partial():
    assert compress_string('aabbcc') == 'aabbcc'
"""
print("Tests:")
print(pytest_tests)
print("-"*80)

result2 = sandbox.execute_tests(problem.reference_solution, pytest_tests)

print(f"\nResult:")
print(f"  Passed: {result2.passed}")
print(f"  num_passed: {result2.num_passed}")
print(f"  num_total: {result2.num_total}")
print(f"  Errors: {result2.errors[:3] if result2.errors else 'None'}")

# Test with BROKEN code
print("\n" + "="*80)
print("3. Testing BROKEN CODE with baseline tests:")
print("-"*80)
broken_code = """def compress_string(s: str) -> str:
    return "wrong"
"""
print("Code:")
print(broken_code)
print("-"*80)

result3 = sandbox.execute_tests(broken_code, baseline_tests_str)

print(f"\nResult:")
print(f"  Passed: {result3.passed}")
print(f"  num_passed: {result3.num_passed}")
print(f"  num_total: {result3.num_total}")
print(f"  Errors: {result3.errors[:3] if result3.errors else 'None'}")

# Test with EMPTY tests
print("\n" + "="*80)
print("4. Testing with EMPTY tests:")
print("-"*80)
result4 = sandbox.execute_tests(problem.reference_solution, "")

print(f"\nResult:")
print(f"  Passed: {result4.passed}")
print(f"  num_passed: {result4.num_passed}")
print(f"  num_total: {result4.num_total}")
print(f"  Errors: {result4.errors[:3] if result4.errors else 'None'}")

print("\n" + "="*80)
print("SUMMARY:")
print("="*80)
print(f"Baseline tests work: {result.num_total > 0}")
print(f"Pytest tests work: {result2.num_total > 0}")
print(f"Broken code detected: {result3.num_passed == 0 and result3.num_total > 0}")
print(f"Empty tests handled: {result4.num_total == 0}")
