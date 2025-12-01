"""Multi-attempt generation with feedback support."""

from typing import List, Tuple
from models.generator import LLMGenerator
from models.discriminator import LLMDiscriminator
from sandbox.sandbox import Sandbox, ExecutionResult
from data.problem_dataset import Problem


class MultiAttemptManager:
    """Manages multiple generation attempts with feedback."""
    
    def __init__(
        self,
        generator: LLMGenerator,
        discriminator: LLMDiscriminator,
        sandbox: Sandbox,
        max_attempts: int = 3
    ):
        """Initialize multi-attempt manager.
        
        Args:
            generator: Generator LLM
            discriminator: Discriminator LLM
            sandbox: Sandbox for execution
            max_attempts: Maximum number of attempts
        """
        self.generator = generator
        self.discriminator = discriminator
        self.sandbox = sandbox
        self.max_attempts = max_attempts
    
    def generate_with_feedback(
        self,
        problem: Problem,
        stage_id: int,
        prompt_template: str,
        max_new_tokens: int = 512
    ) -> Tuple[str, List[ExecutionResult]]:
        """Generate output with multiple attempts and feedback.
        
        Args:
            problem: Problem to solve
            stage_id: Current stage ID
            prompt_template: Prompt template
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (best_output, results_history)
        """
        attempts = []
        results = []
        feedback_history = []
        
        for attempt in range(self.max_attempts):
            # Build feedback prompt
            if attempt == 0:
                feedback = ""
            else:
                feedback = self._build_feedback(
                    attempts[-1],
                    results[-1],
                    feedback_history[-1]
                )
            
            # Generate with feedback
            if stage_id == 5:
                # Code generation
                output = self.generator.generate_code(
                    problem=problem.description,
                    reasoning_chain=[],
                    prompt_template=prompt_template + feedback,
                    max_new_tokens=max_new_tokens
                )
            else:
                # Reasoning stage
                output = self.generator.generate_stage_output(
                    problem=problem.description,
                    previous_stages=[],
                    stage_id=stage_id,
                    prompt_template=prompt_template + feedback,
                    max_new_tokens=max_new_tokens
                )
            
            attempts.append(output)
            
            # Test the output
            if stage_id == 5:
                tests = self.discriminator.generate_tests(
                    problem=problem.description,
                    generator_code=output,
                    num_tests=5
                )
                result = self.sandbox.execute_tests(output, tests)
                results.append(result)
                
                # Generate feedback
                critique = self.discriminator.generate_critique(
                    problem=problem.description,
                    stage_output=output,
                    stage_id=stage_id,
                    prompt_template="Critique this code:\n{stage_output}\n\nCritique:"
                )
                feedback_history.append(critique)
                
                # Check if we achieved good results
                if result.passed:
                    break
            else:
                # For reasoning stages, just store empty result
                results.append(ExecutionResult(
                    passed=True,
                    num_passed=1,
                    num_total=1,
                    errors=[],
                    stdout="",
                    stderr="",
                    timed_out=False
                ))
                feedback_history.append("")
        
        # Return best attempt
        if results and any(r.passed for r in results):
            # Find first passing attempt
            for i, result in enumerate(results):
                if result.passed:
                    return attempts[i], results
        
        # Return last attempt if none passed
        return attempts[-1] if attempts else "", results
    
    def _build_feedback(
        self,
        previous_output: str,
        previous_result: ExecutionResult,
        critique: str
    ) -> str:
        """Build feedback prompt from previous attempt.
        
        Args:
            previous_output: Previous generation output
            previous_result: Execution result
            critique: Discriminator critique
            
        Returns:
            Feedback string to append to prompt
        """
        feedback = "\n\nPrevious Attempt:\n"
        feedback += previous_output + "\n\n"
        
        if not previous_result.passed:
            feedback += f"This attempt failed {previous_result.num_total - previous_result.num_passed} tests.\n"
            if previous_result.errors:
                feedback += "Errors:\n"
                for error in previous_result.errors[:3]:  # Limit to 3 errors
                    feedback += f"- {error}\n"
        
        if critique:
            feedback += f"\nCritique: {critique}\n"
        
        feedback += "\nPlease improve your solution based on this feedback.\n\n"
        
        return feedback
