"""Reward computation for adversarial RL training."""

from sandbox.sandbox import ExecutionResult


def compute_generator_reward(execution_result: ExecutionResult) -> float:
    """Compute reward for generator based on test pass rate.
    
    The generator is rewarded for passing more tests. Reward is the
    percentage of tests passed.
    
    Args:
        execution_result: Result from executing generator code against tests
        
    Returns:
        Reward in range [0.0, 1.0]
    """
    if execution_result.num_total == 0:
        # No tests to pass - neutral reward
        return 0.5
    
    if execution_result.timed_out:
        # Timeout is a failure
        return 0.0
    
    # Reward = pass rate
    pass_rate = execution_result.num_passed / execution_result.num_total
    
    return float(pass_rate)


def compute_discriminator_reward(
    generator_result: ExecutionResult,
    validation_result: ExecutionResult
) -> float:
    """Compute reward for discriminator based on adversarial effectiveness.
    
    The discriminator is rewarded for:
    1. Making the generator fail tests (1.0 - generator_pass_rate)
    2. Generating valid tests that pass against ground truth (test_validity)
    
    Final reward = (1.0 - generator_pass_rate) * test_validity_score
    
    This creates adversarial competition while penalizing invalid tests.
    
    Args:
        generator_result: Result from running tests against generator code
        validation_result: Result from running tests against ground truth
        
    Returns:
        Reward in range [0.0, 1.0]
    """
    # Compute generator pass rate
    if generator_result.num_total == 0:
        generator_pass_rate = 0.0
    else:
        generator_pass_rate = generator_result.num_passed / generator_result.num_total
    
    # Compute test validity score
    if validation_result.num_total == 0:
        test_validity = 0.0
    else:
        test_validity = validation_result.num_passed / validation_result.num_total
    
    # Discriminator wants generator to fail (low pass rate)
    # But only gets credit if tests are valid (pass ground truth)
    adversarial_score = 1.0 - generator_pass_rate
    reward = adversarial_score * test_validity
    
    return float(reward)


def normalize_reward(reward: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize reward to specified range.
    
    Args:
        reward: Raw reward value
        min_val: Minimum value for normalization
        max_val: Maximum value for normalization
        
    Returns:
        Normalized reward
    """
    # Clip to range
    reward = max(min_val, min(max_val, reward))
    
    return reward


def compute_pass_rate(execution_result: ExecutionResult) -> float:
    """Compute test pass rate from execution result.
    
    Args:
        execution_result: Execution result
        
    Returns:
        Pass rate in range [0.0, 1.0]
    """
    if execution_result.num_total == 0:
        return 0.0
    
    return execution_result.num_passed / execution_result.num_total
