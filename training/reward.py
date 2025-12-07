"""Reward computation for adversarial RL training."""

import ast
from dataclasses import dataclass
from execution.direct_executor import execute_tests, ExecutionResult


@dataclass
class Rewards:
    """Rewards for both generator and discriminator."""
    generator_reward: float
    discriminator_reward: float
    gen_result: ExecutionResult = None  # Result on discriminator-generated tests
    val_result: ExecutionResult = None  # Validation result (ground truth on disc tests)
    gen_result_valid_only: ExecutionResult = None  # Only valid disc tests counted
    gen_result_combined: ExecutionResult = None  # Result on disc + baseline tests
    gen_result_baseline_only: ExecutionResult = None  # Result on baseline tests only


def _safe_parse_tests(tests: str):
    """Parse test list string into a Python object; return [] on failure."""
    try:
        return ast.literal_eval(tests)
    except Exception:
        return []


def _merge_baseline_tests(tests_list, baseline_tests):
    """Append baseline tests (list/tuple form) to generated tests list."""
    if not baseline_tests:
        return tests_list
    merged = list(tests_list)
    for t in baseline_tests:
        if isinstance(t, (list, tuple)) and len(t) >= 2:
            inputs = t[:-1]
            expected = t[-1]
            inputs_tuple = tuple(inputs)  # enforce tuple for input args even if single
            merged.append((inputs_tuple, expected))
    return merged


def _pass_rate_minus_one(execution_result: ExecutionResult) -> float:
    """Pass rate shifted so any failure yields a negative reward.

    Returns pass_rate - 1.0; perfect pass -> 0.0, any failure -> negative.
    """
    if execution_result.num_total == 0:
        return 0.0
    return (execution_result.num_passed / execution_result.num_total) - 1.0


def run_code_tests(code: str, tests: str, ground_truth: str, baseline_tests=None) -> Rewards:
    """Compute rewards by executing code and tests.
    
    Args:
        code: Generated code to test
        tests: Test cases as string representation of list of tuples (discriminator-generated)
        ground_truth: Reference solution
        baseline_tests: Optional baseline test cases
        
    Returns:
        Rewards object with generator and discriminator rewards plus execution results
    """
    # Execute ground truth first to validate discriminator-generated tests
    val_result = execute_tests(ground_truth, tests)
    
    # Execute generated code with validity info from validation (for discriminator reward)
    gen_result = execute_tests(code, tests, val_result)
    
    # Execute baseline tests separately
    gen_result_baseline_only = None
    if baseline_tests:
        baseline_tests_str = str(baseline_tests)
        gen_result_baseline_only = execute_tests(code, baseline_tests_str)
    
    # Prepare combined test set for generator reward (add baseline tests)
    parsed_tests = _safe_parse_tests(tests)
    combined_tests = _merge_baseline_tests(parsed_tests, baseline_tests)
    combined_tests_str = str(combined_tests) if combined_tests else tests
    gen_result_combined = execute_tests(code, combined_tests_str)
    
    if gen_result.num_total == 0:
        # If no discriminator tests, use baseline-only result for valid_only
        gen_result_valid_only = gen_result_baseline_only if gen_result_baseline_only else gen_result
        return Rewards(0, -1, gen_result_combined, val_result, gen_result_valid_only, gen_result_combined, gen_result_baseline_only)
    
    # Create filtered result with only valid discriminator tests
    valid_indices = [i for i, v in enumerate(gen_result.is_valid) if v]
    valid_passed = [i for i in valid_indices if i in gen_result.passed_tests]
    valid_failed = [i for i in valid_indices if i in gen_result.failed_tests]
    
    gen_result_valid_only = ExecutionResult(
        num_passed=len(valid_passed),
        num_total=len(valid_indices),
        passed_tests=valid_passed,
        failed_tests=valid_failed,
        is_valid=[True] * len(valid_indices)
    )
    
    # Reward constants (normalized by test count)
    n = gen_result.num_total
    CORRECT_TEST = 0.01 / n
    WRONG_TEST = -1 / n
    CAUGHT_BUG = 1 / n
    
    # Compute rewards directly from execution indices
    invalid_indices = [i for i, v in enumerate(gen_result.is_valid) if not v]
    
    # Discriminator reward: valid tests are good, invalid are bad, catching bugs is good
    valid_passed_count = len(valid_passed)
    valid_failed_count = len(valid_failed)
    disc_reward = (len(valid_indices) * CORRECT_TEST) + (valid_passed_count * 0) + (valid_failed_count * CAUGHT_BUG) + (len(invalid_indices) * WRONG_TEST)
    
    # Generator reward: simple pass rate on combined tests (no -1 shift)
    gen_reward = gen_result_combined.num_passed / gen_result_combined.num_total if gen_result_combined.num_total > 0 else 0.0
    
    print(f"num tests: {n}")
    print(f"valid tests: {len(valid_indices)}")
    print(f"passed valid tests: {valid_passed_count}")
    print(f"baseline tests: {gen_result_baseline_only.num_total if gen_result_baseline_only else 0}")
    print(f"combined tests: {gen_result_combined.num_total}")
    
    return Rewards(gen_reward, disc_reward, gen_result_combined, val_result, gen_result_valid_only, gen_result_combined, gen_result_baseline_only)


def compute_generator_reward(execution_result: ExecutionResult) -> float:
    """Generator reward: pass_rate - 1 so any failure is negative."""
    return _pass_rate_minus_one(execution_result)


def compute_discriminator_reward(
    generator_result: ExecutionResult,
    validation_result: ExecutionResult
) -> float:
    """Discriminator reward = (1 - gen_pass_rate) * test_validity."""
    gen_pass = (generator_result.num_passed / generator_result.num_total 
                if generator_result.num_total > 0 else 0.0)
    val_pass = (validation_result.num_passed / validation_result.num_total 
                if validation_result.num_total > 0 else 0.0)
    return (1.0 - gen_pass) * val_pass
