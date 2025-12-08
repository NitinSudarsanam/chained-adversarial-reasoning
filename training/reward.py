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
        # If no discriminator tests, still penalize discriminator for lack of test generation
        # Generator should be evaluated on baseline tests if available
        gen_result_valid_only = gen_result_baseline_only if gen_result_baseline_only else gen_result
        
        # Use baseline tests for combined result if available
        if gen_result_baseline_only and gen_result_baseline_only.num_total > 0:
            gen_result_combined_to_use = gen_result_baseline_only
        else:
            gen_result_combined_to_use = gen_result_combined
        
        # Compute generator reward based on baseline tests if available
        if gen_result_combined_to_use.num_total > 0:
            raw_gen_reward = gen_result_combined_to_use.num_passed / gen_result_combined_to_use.num_total
            gen_reward = raw_gen_reward - 0.5  # Shift to ensure non-zero loss
        else:
            gen_reward = 0.0  # No tests at all (no valid disc tests, no baseline), neutral reward
        
        # Discriminator gets -0.5 for not generating valid tests
        # Print debug info
        print(f"num tests: 0 (no valid discriminator tests)")
        print(f"valid tests: 0")
        print(f"invalid tests: {val_result.num_total if val_result else 0}")
        print(f"passed valid tests: 0")
        print(f"failed valid tests (bugs caught): 0")
        print(f"baseline tests: {gen_result_baseline_only.num_total if gen_result_baseline_only else 0}")
        print(f"combined tests: {gen_result_combined_to_use.num_total}")
        
        # Debug: Print test execution details with clearer labels
        if gen_result_combined_to_use.num_total > 0:
            print(f"\nTest execution breakdown:")
            print(f"  Baseline tests only: {gen_result_combined_to_use.num_total} tests")
            print(f"    Passed: {gen_result_combined_to_use.passed_tests} ({gen_result_combined_to_use.num_passed} total)")
            print(f"    Failed: {gen_result_combined_to_use.failed_tests} ({len(gen_result_combined_to_use.failed_tests)} total)")
        
        print(f"\ndisc_reward: -0.5000 (no valid tests generated)")
        if gen_result_combined_to_use.num_total > 0:
            print(f"gen_reward: {raw_gen_reward:.4f} (raw pass_rate on baseline) -> {gen_reward:.4f} (shifted by -0.5)")
        else:
            print(f"gen_reward: 0.0000 (no tests available)")
        
        return Rewards(gen_reward, -0.5, gen_result_combined_to_use, val_result, gen_result_valid_only, gen_result_combined_to_use, gen_result_baseline_only)
    
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
    
    # Reward constants for discriminator
    VALID_REWARD = 0.4      # Max reward for generating valid tests
    INVALID_PENALTY = 0.6   # Max penalty for invalid tests (scales with invalid%)
    BUG_CATCH_BONUS = 0.6   # Max bonus for catching bugs
    
    # Compute rewards directly from execution indices
    invalid_indices = [i for i, v in enumerate(gen_result.is_valid) if not v]
    
    # Calculate percentages
    valid_passed_count = len(valid_passed)
    valid_failed_count = len(valid_failed)
    num_valid = len(valid_indices)
    num_invalid = len(invalid_indices)
    
    invalid_pct = num_invalid / n if n > 0 else 0.0  # Percentage of tests that are invalid
    valid_pct = num_valid / n if n > 0 else 0.0      # Percentage of tests that are valid
    bug_catch_rate = valid_failed_count / n if n > 0 else 0.0  # Percentage of tests catching bugs
    
    # Discriminator reward structure:
    # Base: reward for generating valid tests = +VALID_REWARD * valid_pct
    # Penalty: penalize for invalid tests magnitude = -INVALID_PENALTY * invalid_pct (magnitude varies with invalid percentage)
    # Bonus: reward for catching bugs = +BUG_CATCH_BONUS * bug_catch_rate
    
    disc_reward = 0.0
    # Reward for valid tests (up to +VALID_REWARD)
    disc_reward += valid_pct * VALID_REWARD
    # Penalty for invalid tests - magnitude proportional to invalid_pct (up to -INVALID_PENALTY)
    disc_reward -= invalid_pct * INVALID_PENALTY
    # Bonus for catching bugs (up to +BUG_CATCH_BONUS)
    disc_reward += bug_catch_rate * BUG_CATCH_BONUS
    
    # Generator reward: simple pass rate on combined tests (no -1 shift)
    gen_reward = gen_result_combined.num_passed / gen_result_combined.num_total if gen_result_combined.num_total > 0 else 0.0
    
    print(f"num tests: {n}")
    print(f"valid tests: {num_valid} ({valid_pct*100:.1f}%)")
    print(f"invalid tests: {num_invalid} ({invalid_pct*100:.1f}%)")
    print(f"passed valid tests: {valid_passed_count}")
    print(f"failed valid tests (bugs caught): {valid_failed_count} ({bug_catch_rate*100:.1f}%)")
    print(f"baseline tests: {gen_result_baseline_only.num_total if gen_result_baseline_only else 0}")
    print(f"combined tests: {gen_result_combined.num_total}")
    
    # Debug: Print test execution details with clearer labels
    if gen_result_combined.num_total > 0:
        print(f"\nTest execution breakdown:")
        print(f"  Combined tests (disc valid + disc invalid + baseline): {gen_result_combined.num_total} tests")
        print(f"    Passed: {gen_result_combined.passed_tests} ({gen_result_combined.num_passed} total)")
        print(f"    Failed: {gen_result_combined.failed_tests} ({len(gen_result_combined.failed_tests)} total)")
        
        if gen_result_baseline_only and gen_result_baseline_only.num_total > 0:
            print(f"  Baseline tests only: {gen_result_baseline_only.num_total} tests")
            print(f"    Passed: {gen_result_baseline_only.passed_tests} ({gen_result_baseline_only.num_passed} total)")
            print(f"    Failed: {gen_result_baseline_only.failed_tests} ({len(gen_result_baseline_only.failed_tests)} total)")
        
        if num_valid > 0:
            print(f"  Valid discriminator tests only: {num_valid} tests")
            print(f"    Passed: {valid_passed} ({len(valid_passed)} total)")
            print(f"    Failed: {valid_failed} ({len(valid_failed)} total)")
    
    print(f"\ndisc_reward components: valid({valid_pct*VALID_REWARD:.3f}) - invalid({invalid_pct*INVALID_PENALTY:.3f}) + bugs({bug_catch_rate*BUG_CATCH_BONUS:.3f}) = {disc_reward:.4f}")
    print(f"gen_reward: {raw_gen_reward:.4f} (raw pass_rate) -> {gen_reward:.4f} (shifted by -0.5)")
    
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
