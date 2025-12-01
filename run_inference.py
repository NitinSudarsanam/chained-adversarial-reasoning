"""Main entry point for running inference with trained models."""

import argparse
import json
from pathlib import Path

from inference.inference_engine import InferenceEngine
from data.problem_dataset import load_problems


def format_result(result, show_reasoning=True):
    """Format inference result for display.
    
    Args:
        result: InferenceResult object
        show_reasoning: Whether to show full reasoning chain
        
    Returns:
        Formatted string
    """
    output = []
    output.append("=" * 60)
    output.append("INFERENCE RESULT")
    output.append("=" * 60)
    output.append(f"\nProblem: {result.problem_description[:100]}...")
    output.append(f"Inference Time: {result.inference_time:.2f}s")
    
    if show_reasoning:
        output.append("\n" + "-" * 60)
        output.append("REASONING CHAIN")
        output.append("-" * 60)
        
        for i, stage_output in enumerate(result.reasoning_chain, 1):
            output.append(f"\nStage {i} ({result.stage_times[i-1]:.2f}s):")
            output.append(stage_output[:200] + "..." if len(stage_output) > 200 else stage_output)
    
    output.append("\n" + "-" * 60)
    output.append("GENERATED CODE")
    output.append("-" * 60)
    output.append(result.generated_code)
    
    if result.execution_result:
        output.append("\n" + "-" * 60)
        output.append("EXECUTION RESULT")
        output.append("-" * 60)
        output.append(f"Passed: {result.execution_result.passed}")
        output.append(f"Tests: {result.execution_result.num_passed}/{result.execution_result.num_total}")
        
        if result.execution_result.errors:
            output.append("\nErrors:")
            for error in result.execution_result.errors[:3]:  # Show first 3 errors
                output.append(f"  - {error}")
    
    output.append("=" * 60)
    
    return "\n".join(output)


def run_single_problem(args):
    """Run inference on a single problem."""
    print("\n" + "=" * 60)
    print("SINGLE PROBLEM INFERENCE")
    print("=" * 60 + "\n")
    
    # Load inference engine
    print(f"Loading model from {args.checkpoint}...")
    engine = InferenceEngine.from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    print()
    
    # Get problem description
    if args.problem_file:
        with open(args.problem_file, 'r', encoding='utf-8') as f:
            problem_description = f.read()
    else:
        problem_description = args.problem
    
    # Run inference
    print("Running inference...")
    result = engine.solve_problem(
        problem_description=problem_description,
        function_signature=args.signature or "",
        execute_tests=args.execute_tests,
        test_cases=args.tests.split(';') if args.tests else None,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    # Display result
    print("\n" + format_result(result, show_reasoning=not args.no_reasoning))
    
    # Save result if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result_dict = {
            'problem': result.problem_description,
            'reasoning_chain': result.reasoning_chain,
            'generated_code': result.generated_code,
            'inference_time': result.inference_time,
            'stage_times': result.stage_times
        }
        
        if result.execution_result:
            result_dict['execution'] = {
                'passed': result.execution_result.passed,
                'num_passed': result.execution_result.num_passed,
                'num_total': result.execution_result.num_total,
                'errors': result.execution_result.errors
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"\nResult saved to {output_path}")


def run_batch_inference(args):
    """Run inference on multiple problems from a file."""
    print("\n" + "=" * 60)
    print("BATCH INFERENCE")
    print("=" * 60 + "\n")
    
    # Load inference engine
    print(f"Loading model from {args.checkpoint}...")
    engine = InferenceEngine.from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    print()
    
    # Load problems
    print(f"Loading problems from {args.problems_file}...")
    problems = load_problems(args.problems_file)
    print(f"Loaded {len(problems)} problems\n")
    
    # Run batch inference
    print("Running batch inference...")
    results = engine.solve_batch(
        problems=problems,
        execute_tests=args.execute_tests,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    # Display summary
    print("\n" + "=" * 60)
    print("BATCH INFERENCE SUMMARY")
    print("=" * 60)
    
    total_time = sum(r.inference_time for r in results)
    avg_time = total_time / len(results) if results else 0
    
    print(f"\nTotal Problems: {len(results)}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Time: {avg_time:.2f}s per problem")
    
    if args.execute_tests:
        passed_count = sum(
            1 for r in results
            if r.execution_result and r.execution_result.passed
        )
        print(f"Tests Passed: {passed_count}/{len(results)}")
    
    print("\nPer-Problem Results:")
    for i, result in enumerate(results, 1):
        status = "✓" if result.execution_result and result.execution_result.passed else "✗"
        print(f"  {i}. {result.problem_description[:50]}... ({result.inference_time:.2f}s) {status}")
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_list = []
        for result in results:
            result_dict = {
                'problem': result.problem_description,
                'reasoning_chain': result.reasoning_chain,
                'generated_code': result.generated_code,
                'inference_time': result.inference_time,
                'stage_times': result.stage_times
            }
            
            if result.execution_result:
                result_dict['execution'] = {
                    'passed': result.execution_result.passed,
                    'num_passed': result.execution_result.num_passed,
                    'num_total': result.execution_result.num_total,
                    'errors': result.execution_result.errors
                }
            
            results_list.append(result_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, indent=2)
        
        print(f"\nResults saved to {output_path}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Run inference with trained adversarial RL models"
    )
    
    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help="Path to checkpoint file"
    )
    
    # Mode selection
    parser.add_argument(
        '--batch',
        action='store_true',
        help="Run batch inference on multiple problems"
    )
    
    # Single problem arguments
    parser.add_argument(
        '--problem',
        type=str,
        help="Problem description (for single problem mode)"
    )
    parser.add_argument(
        '--problem-file',
        type=str,
        help="Path to file containing problem description"
    )
    parser.add_argument(
        '--signature',
        type=str,
        help="Function signature"
    )
    parser.add_argument(
        '--tests',
        type=str,
        help="Test cases separated by semicolons"
    )
    
    # Batch mode arguments
    parser.add_argument(
        '--problems-file',
        type=str,
        help="Path to problems JSON file (for batch mode)"
    )
    
    # Common arguments
    parser.add_argument(
        '--device',
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on"
    )
    parser.add_argument(
        '--execute-tests',
        action='store_true',
        help="Execute generated code against test cases"
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help="Maximum tokens to generate per stage"
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.9,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        '--output',
        type=str,
        help="Path to save results JSON"
    )
    parser.add_argument(
        '--no-reasoning',
        action='store_true',
        help="Don't display reasoning chain (only code)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.batch:
            if not args.problems_file:
                print("Error: --problems-file required for batch mode")
                return 1
            run_batch_inference(args)
        else:
            if not args.problem and not args.problem_file:
                print("Error: --problem or --problem-file required for single problem mode")
                return 1
            run_single_problem(args)
        
        print("\n" + "=" * 60)
        print("Inference completed successfully!")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
