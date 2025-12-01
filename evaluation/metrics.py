"""Evaluation metrics for adversarial RL training."""

from typing import List, Dict
from sandbox.sandbox import ExecutionResult


def compute_pass_rate(results: List[ExecutionResult]) -> float:
    """Compute average test pass rate across multiple results.
    
    Args:
        results: List of execution results
        
    Returns:
        Average pass rate in range [0.0, 1.0]
    """
    if not results:
        return 0.0
    
    total_passed = sum(r.num_passed for r in results)
    total_tests = sum(r.num_total for r in results)
    
    if total_tests == 0:
        return 0.0
    
    return total_passed / total_tests


def compute_failure_rate(results: List[ExecutionResult]) -> float:
    """Compute average test failure rate across multiple results.
    
    Args:
        results: List of execution results
        
    Returns:
        Average failure rate in range [0.0, 1.0]
    """
    return 1.0 - compute_pass_rate(results)


def compute_test_diversity(test_suites: List[str]) -> float:
    """Measure uniqueness of generated tests.
    
    Computes diversity based on unique test function names and patterns.
    
    Args:
        test_suites: List of test suite strings
        
    Returns:
        Diversity score in range [0.0, 1.0]
    """
    if not test_suites:
        return 0.0
    
    # Extract test function names
    all_test_names = []
    for suite in test_suites:
        lines = suite.split('\n')
        for line in lines:
            if 'def test_' in line:
                # Extract function name
                start = line.find('def test_')
                if start != -1:
                    end = line.find('(', start)
                    if end != -1:
                        test_name = line[start+4:end]
                        all_test_names.append(test_name)
    
    if not all_test_names:
        return 0.0
    
    # Compute uniqueness ratio
    unique_tests = len(set(all_test_names))
    total_tests = len(all_test_names)
    
    diversity = unique_tests / total_tests
    
    return diversity


def compute_reasoning_coherence(reasoning_chain: List[str]) -> float:
    """Measure consistency across reasoning stages.
    
    Computes coherence based on:
    - Length consistency (stages should have reasonable length)
    - Content overlap (later stages should reference earlier concepts)
    
    Args:
        reasoning_chain: List of reasoning outputs for each stage
        
    Returns:
        Coherence score in range [0.0, 1.0]
    """
    if not reasoning_chain or len(reasoning_chain) < 2:
        return 0.0
    
    # Check length consistency
    lengths = [len(stage.split()) for stage in reasoning_chain]
    avg_length = sum(lengths) / len(lengths)
    
    # Penalize if any stage is too short or too long
    length_score = 1.0
    for length in lengths:
        if length < 5:  # Too short
            length_score *= 0.8
        elif length > avg_length * 3:  # Too long
            length_score *= 0.9
    
    # Check content overlap (simple word-based)
    overlap_scores = []
    for i in range(1, len(reasoning_chain)):
        prev_words = set(reasoning_chain[i-1].lower().split())
        curr_words = set(reasoning_chain[i].lower().split())
        
        if len(curr_words) == 0:
            overlap_scores.append(0.0)
        else:
            overlap = len(prev_words & curr_words) / len(curr_words)
            overlap_scores.append(min(overlap, 1.0))
    
    avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0.0
    
    # Combine scores
    coherence = (length_score + avg_overlap) / 2.0
    
    return min(coherence, 1.0)


def aggregate_metrics(
    execution_results: List[ExecutionResult],
    test_suites: List[str],
    reasoning_chains: List[List[str]]
) -> Dict[str, float]:
    """Aggregate all metrics into a single dictionary.
    
    Args:
        execution_results: List of execution results
        test_suites: List of generated test suites
        reasoning_chains: List of reasoning chains
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {
        'pass_rate': compute_pass_rate(execution_results),
        'failure_rate': compute_failure_rate(execution_results),
        'test_diversity': compute_test_diversity(test_suites),
        'reasoning_coherence': 0.0
    }
    
    # Compute average reasoning coherence
    if reasoning_chains:
        coherence_scores = [
            compute_reasoning_coherence(chain) 
            for chain in reasoning_chains
        ]
        metrics['reasoning_coherence'] = sum(coherence_scores) / len(coherence_scores)
    
    return metrics
