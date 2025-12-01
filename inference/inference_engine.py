"""Inference engine for using trained models to solve new problems."""

import time
import torch
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

from models.generator import LLMGenerator
from models.discriminator import LLMDiscriminator
from sandbox.sandbox import Sandbox, ExecutionResult
from data.problem_dataset import Problem
from reasoning.stages import get_stage


@dataclass
class InferenceResult:
    """Result from inference on a single problem."""
    problem_description: str
    reasoning_chain: List[str]  # Outputs from stages 1-4
    generated_code: str  # Output from stage 5
    execution_result: Optional[ExecutionResult]
    inference_time: float
    stage_times: List[float]  # Time per stage


class InferenceEngine:
    """Engine for running inference with trained models."""
    
    def __init__(
        self,
        generator: LLMGenerator,
        discriminator: LLMDiscriminator,
        sandbox: Sandbox,
        device: str = "cpu"
    ):
        """Initialize inference engine.
        
        Args:
            generator: Trained generator model
            discriminator: Trained discriminator model
            sandbox: Sandbox for code execution
            device: Device to run on (cpu/cuda)
        """
        self.generator = generator
        self.discriminator = discriminator
        self.sandbox = sandbox
        self.device = device
        
        # Set models to eval mode
        self.generator.model.eval()
        self.discriminator.model.eval()
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        generator_model_name: str = None,
        discriminator_model_name: str = None,
        device: str = "cpu"
    ) -> "InferenceEngine":
        """Load inference engine from saved checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            generator_model_name: Optional model name override
            discriminator_model_name: Optional model name override
            device: Device to run on
            
        Returns:
            Initialized InferenceEngine
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Load checkpoint data
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        
        # Get model names from checkpoint or use provided
        gen_model_name = generator_model_name or checkpoint_data.get(
            'config', {}
        ).get('generator_model', 'Qwen/Qwen2.5-Coder-1.5B-Instruct')
        
        disc_model_name = discriminator_model_name or checkpoint_data.get(
            'config', {}
        ).get('discriminator_model', 'Qwen/Qwen2.5-Coder-1.5B-Instruct')
        
        # Initialize models
        print(f"Loading generator: {gen_model_name}")
        generator = LLMGenerator(model_name=gen_model_name, device=device)
        generator.model.load_state_dict(checkpoint_data['generator_state_dict'])
        generator.model.eval()
        
        print(f"Loading discriminator: {disc_model_name}")
        discriminator = LLMDiscriminator(model_name=disc_model_name, device=device)
        discriminator.model.load_state_dict(checkpoint_data['discriminator_state_dict'])
        discriminator.model.eval()
        
        # Initialize sandbox
        sandbox = Sandbox(timeout=5)
        
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
        
        return cls(generator, discriminator, sandbox, device)
    
    def solve_problem(
        self,
        problem_description: str,
        function_signature: str = "",
        execute_tests: bool = False,
        test_cases: Optional[List[str]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> InferenceResult:
        """Execute multi-stage reasoning to solve a problem.
        
        Args:
            problem_description: Description of the problem to solve
            function_signature: Optional function signature
            execute_tests: Whether to execute generated code against tests
            test_cases: Optional list of test cases to run
            max_new_tokens: Max tokens to generate per stage
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            InferenceResult with reasoning chain and code
        """
        start_time = time.time()
        stage_times = []
        reasoning_chain = []
        
        # Combine problem description with function signature if provided
        full_problem = problem_description
        if function_signature:
            full_problem = f"{problem_description}\n\nFunction signature:\n{function_signature}"
        
        # Generate reasoning through all 5 stages
        with torch.no_grad():
            for stage_id in range(1, 6):
                stage_start = time.time()
                stage = get_stage(stage_id)
                
                if stage_id == 5:
                    # Final code generation
                    output = self.generator.generate_code(
                        problem=full_problem,
                        reasoning_chain=reasoning_chain,
                        prompt_template=stage.generator_prompt_template,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        function_signature=function_signature
                    )
                else:
                    # Intermediate reasoning stage
                    output = self.generator.generate_stage_output(
                        problem=full_problem,
                        previous_stages=reasoning_chain,
                        stage_id=stage_id,
                        prompt_template=stage.generator_prompt_template,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )
                
                reasoning_chain.append(output)
                stage_times.append(time.time() - stage_start)
        
        # Extract final code (last stage output)
        generated_code = reasoning_chain[-1] if reasoning_chain else ""
        
        # Optionally execute tests
        execution_result = None
        if execute_tests and test_cases and generated_code:
            # Combine test cases into test code
            test_code = "\n".join(test_cases)
            execution_result = self.sandbox.execute_tests(generated_code, test_code)
        
        total_time = time.time() - start_time
        
        return InferenceResult(
            problem_description=problem_description,
            reasoning_chain=reasoning_chain[:-1],  # Exclude final code from chain
            generated_code=generated_code,
            execution_result=execution_result,
            inference_time=total_time,
            stage_times=stage_times
        )
    
    def solve_batch(
        self,
        problems: List[Problem],
        execute_tests: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> List[InferenceResult]:
        """Solve multiple problems efficiently.
        
        Args:
            problems: List of problems to solve
            execute_tests: Whether to execute generated code
            max_new_tokens: Max tokens per stage
            temperature: Sampling temperature
            top_p: Top-p sampling
            
        Returns:
            List of InferenceResults
        """
        results = []
        
        for problem in problems:
            # Use baseline tests if available
            test_cases = problem.baseline_tests if execute_tests else None
            
            result = self.solve_problem(
                problem_description=problem.description,
                function_signature=problem.function_signature,
                execute_tests=execute_tests,
                test_cases=test_cases,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            results.append(result)
        
        return results
    
    def get_reasoning_chain(
        self,
        problem_description: str,
        function_signature: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> List[str]:
        """Get only the reasoning chain without code execution.
        
        Args:
            problem_description: Problem to solve
            function_signature: Optional function signature
            max_new_tokens: Max tokens per stage
            temperature: Sampling temperature
            top_p: Top-p sampling
            
        Returns:
            List of reasoning outputs from stages 1-5
        """
        result = self.solve_problem(
            problem_description=problem_description,
            function_signature=function_signature,
            execute_tests=False,
            test_cases=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Return full chain including final code
        return result.reasoning_chain + [result.generated_code]
