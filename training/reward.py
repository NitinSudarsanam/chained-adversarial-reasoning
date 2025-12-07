"""Reward computation for adversarial RL training."""

from dataclasses import dataclass
from execution.direct_executor import execute_tests, ExecutionResult


@dataclass
class Rewards:
    """Rewards for both generator and discriminator."""
    generator_reward: float
    discriminator_reward: float
    gen_result: ExecutionResult = None
    val_result: ExecutionResult = None


def run_code_tests(code: str, tests: str, ground_truth: str) -> Rewards:
    """Compute rewards by executing code and tests.
    
    Args:
        code: Generated code to test
        tests: Test cases as string representation of list of tuples
        ground_truth: Reference solution
        
    Returns:
        Rewards object with generator and discriminator rewards plus execution results
    """
    # Execute ground truth first to validate tests
    val_result = execute_tests(ground_truth, tests)
    
    # Execute generated code with validity info from validation
    gen_result = execute_tests(code, tests, val_result)
    
    if gen_result.num_total == 0:
        return Rewards(0, -1, gen_result, val_result)
    
    # Reward constants (normalized by test count)
    n = gen_result.num_total
    CORRECT_TEST = 0.01 / n
    WRONG_TEST = -1 / n
    PASSED_TEST = 1 / n
    FAILED_TEST = -1 / n
    CAUGHT_BUG = 1 / n
    
    # Compute rewards directly from execution indices
    valid_indices = [i for i, v in enumerate(gen_result.is_valid) if v]
    invalid_indices = [i for i, v in enumerate(gen_result.is_valid) if not v]
    
    # Rewards for valid tests
    valid_passed = len(set(valid_indices) & set(gen_result.passed_tests))
    valid_failed = len(valid_indices) - valid_passed
    
    gen_reward = (valid_passed * PASSED_TEST) + (valid_failed * FAILED_TEST)
    disc_reward = (len(valid_indices) * CORRECT_TEST) + (valid_passed * 0) + (valid_failed * CAUGHT_BUG) + (len(invalid_indices) * WRONG_TEST)
    
    print(f"num tests: {n}")
    print(f"valid tests: {len(valid_indices)}")
    print(f"passed valid tests: {valid_passed}")
    
    return Rewards(gen_reward, disc_reward, gen_result, val_result)


def compute_generator_reward(execution_result: ExecutionResult) -> float:
    """Generator reward = pass rate."""
    if execution_result.num_total == 0:
        return 0.0
    return execution_result.num_passed / execution_result.num_total


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
