"""Simplified trainer using a single unified model for both generator and discriminator."""

import torch
from typing import List, Dict, Any
from tqdm import tqdm

from models.unified_model import UnifiedModel
from sandbox.sandbox import Sandbox
from data.problem_dataset import Problem
from reasoning.stages import get_stage
from training.reward import compute_generator_reward, compute_discriminator_reward
from training.rl_loop import train_step, create_optimizer, freeze_model, unfreeze_model
from training.config import TrainingConfig
from training.checkpoint_manager import CheckpointManager


class UnifiedTrainer:
    """Trainer using a single model for both generation and discrimination."""
    
    def __init__(
        self,
        model: UnifiedModel,
        sandbox: Sandbox,
        config: TrainingConfig,
        checkpoint_manager: CheckpointManager = None
    ):
        """Initialize unified trainer.
        
        Args:
            model: Unified model (acts as both generator and discriminator)
            sandbox: Sandbox for code execution
            config: Training configuration
            checkpoint_manager: Optional checkpoint manager
        """
        self.model = model
        self.sandbox = sandbox
        self.config = config
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        
        # Create single optimizer
        self.optimizer = create_optimizer(model, config.learning_rate)
        
        # Training state
        self.current_stage = 1
        self.metrics_history = []
        
        # Caches for efficiency
        self._generation_cache = {}
        self._test_cache = {}
    
    def clear_caches(self):
        """Clear generation and test caches to free memory."""
        self._generation_cache.clear()
        self._test_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def train_stage(
        self,
        stage_id: int,
        problems: List[Problem],
        n_steps: int
    ) -> Dict[str, Any]:
        """Train model at a specific reasoning stage.
        
        Alternates between:
        - Training as discriminator (generate tests)
        - Training as generator (solve problems)
        
        Args:
            stage_id: Stage ID (1-5)
            problems: List of problems to train on
            n_steps: Number of training steps
            
        Returns:
            Dictionary of stage metrics
        """
        print(f"\n{'='*60}")
        print(f"Training Stage {stage_id}: {get_stage(stage_id).name}")
        print(f"{'='*60}\n")
        
        total_gen_reward = 0.0
        total_disc_reward = 0.0
        total_loss = 0.0
        num_updates = 0
        num_skipped = 0
        
        for step in tqdm(range(n_steps), desc=f"Stage {stage_id}"):
            problem = problems[step % len(problems)]
            
            # Clear CUDA cache periodically (not every step)
            if step > 0 and step % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Alternate between discriminator and generator training
            if step % 2 == 0:
                # Train as discriminator
                metrics = self._train_discriminator_step(problem, stage_id)
                if metrics:
                    total_disc_reward += metrics.get('reward', 0.0)
                    if 'policy_loss' in metrics:
                        total_loss += metrics['policy_loss']
                        num_updates += 1
                else:
                    num_skipped += 1
            else:
                # Train as generator
                metrics = self._train_generator_step(problem, stage_id)
                if metrics:
                    total_gen_reward += metrics.get('reward', 0.0)
                    if 'policy_loss' in metrics:
                        total_loss += metrics['policy_loss']
                        num_updates += 1
                else:
                    num_skipped += 1
        
        # Compute averages
        n_gen_steps = (n_steps + 1) // 2
        n_disc_steps = n_steps // 2
        
        avg_gen_reward = total_gen_reward / n_gen_steps if n_gen_steps > 0 else 0.0
        avg_disc_reward = total_disc_reward / n_disc_steps if n_disc_steps > 0 else 0.0
        avg_loss = total_loss / num_updates if num_updates > 0 else 0.0
        
        if num_skipped > 0:
            print(f"\n  Warning: Skipped {num_skipped}/{n_steps} steps due to empty generation")
        
        # Clear caches after each stage to free memory
        print(f"  Clearing caches (saved {len(self._generation_cache)} generations, {len(self._test_cache)} tests)")
        self.clear_caches()
        
        stage_metrics = {
            'stage_id': stage_id,
            'stage_name': get_stage(stage_id).name,
            'avg_gen_reward': avg_gen_reward,
            'avg_disc_reward': avg_disc_reward,
            'avg_loss': avg_loss,
            'num_updates': num_updates,
            'num_skipped': num_skipped
        }
        
        # Save checkpoint
        checkpoint_metrics = {
            'generator_reward': avg_gen_reward,
            'discriminator_reward': avg_disc_reward,
            'test_validity': 0.9
        }
        is_best = self.checkpoint_manager.should_save_as_best(checkpoint_metrics)
        
        # Note: checkpoint saving would need to be adapted for unified model
        # self.checkpoint_manager.save_checkpoint(...)
        
        return stage_metrics
    
    def _train_discriminator_step(self, problem: Problem, stage_id: int) -> Dict:
        """Single discriminator training step."""
        # Generate only up to target stage (not all 5!)
        reasoning_chain, final_code, accumulated_tests = self._generate_up_to_stage(problem, stage_id)
        
        if not final_code or not final_code.strip() or not accumulated_tests or not accumulated_tests.strip():
            return None
        
        # Generate tests for this stage
        stage_output = reasoning_chain[stage_id - 1] if stage_id <= len(reasoning_chain) else final_code
        stage = get_stage(stage_id)
        
        prompt = stage.discriminator_prompt_template.format(
            problem=problem.description,
            stage_output=stage_output,
            num_tests=self.config.num_tests_per_problem
        )
        
        # Switch to discriminator mode
        self.model.set_discriminator_mode()
        
        self.model.eval()
        with torch.no_grad():
            stage_tests = self.model.generate_tests(
                problem=problem.description,
                generator_code=stage_output,
                num_tests=self.config.num_tests_per_problem,
                prompt_template=stage.discriminator_prompt_template,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature
            )
        
        if not stage_tests or not stage_tests.strip():
            return None
        
        # Get log probs with discriminator adapter active
        old_log_probs = self.model.get_log_probs(prompt, stage_tests)
        
        # Execute and validate tests
        gen_result = self.sandbox.execute_tests(final_code, accumulated_tests)
        val_result = self.sandbox.validate_tests_against_solution(accumulated_tests, problem.reference_solution)
        
        reward = compute_discriminator_reward(gen_result, val_result)
        
        # Train
        self.model.train()
        metrics = train_step(
            model=self.model,
            optimizer=self.optimizer,
            prompts=[prompt],
            outputs=[stage_tests],
            rewards=[reward],
            old_log_probs_list=[old_log_probs],
            clip_epsilon=self.config.clip_epsilon
        )
        self.model.eval()
        
        metrics['reward'] = reward
        return metrics
    
    def _train_generator_step(self, problem: Problem, stage_id: int) -> Dict:
        """Single generator training step."""
        # Generate only up to target stage (not all 5!)
        reasoning_chain, final_code, accumulated_tests = self._generate_up_to_stage(problem, stage_id)
        
        if not final_code or not final_code.strip() or not accumulated_tests or not accumulated_tests.strip():
            return None
        
        # Switch to generator mode
        self.model.set_generator_mode()
        
        # Get log probs for this stage
        stage = get_stage(stage_id)
        stage_output = reasoning_chain[stage_id - 1] if stage_id <= len(reasoning_chain) else final_code
        
        previous_text = "\n\n".join([
            f"Stage {i+1}:\n{s}" 
            for i, s in enumerate(reasoning_chain[:stage_id-1])
        ])
        prompt = stage.generator_prompt_template.format(
            problem=problem.description,
            previous_stages=previous_text if stage_id > 1 else "None"
        )
        
        # Get log probs with generator adapter active
        old_log_probs = self.model.get_log_probs(prompt, stage_output)
        
        # Execute tests
        result = self.sandbox.execute_tests(final_code, accumulated_tests)
        reward = compute_generator_reward(result)
        
        # Train
        self.model.train()
        metrics = train_step(
            model=self.model,
            optimizer=self.optimizer,
            prompts=[prompt],
            outputs=[stage_output],
            rewards=[reward],
            old_log_probs_list=[old_log_probs],
            clip_epsilon=self.config.clip_epsilon
        )
        self.model.eval()
        
        metrics['reward'] = reward
        return metrics
    
    def _generate_up_to_stage(self, problem: Problem, target_stage: int) -> tuple:
        """Generate reasoning chain up to target stage only (not all 5).
        
        This is much more efficient than generating all 5 stages when we only need
        stages 1-N for training stage N.
        """
        # Check cache first
        cache_key = (problem.id, target_stage)
        if hasattr(self, '_generation_cache') and cache_key in self._generation_cache:
            return self._generation_cache[cache_key]
        
        reasoning_chain = []
        accumulated_tests = []
        
        self.model.eval()
        with torch.no_grad():
            for stage_id in range(1, target_stage + 1):
                stage = get_stage(stage_id)
                
                # Generate reasoning/code
                if stage_id == 5:
                    output = self.model.generate_code(
                        problem=problem.description,
                        reasoning_chain=reasoning_chain,
                        prompt_template=stage.generator_prompt_template,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        function_signature=problem.function_signature
                    )
                    if not output or not output.strip():
                        print(f"    Warning: Empty code generated at stage {stage_id}")
                    elif len(output) < 20:
                        print(f"    Warning: Very short code at stage {stage_id}: {output[:50]}")
                else:
                    output = self.model.generate_stage_output(
                        problem=problem.description,
                        previous_stages=reasoning_chain,
                        stage_id=stage_id,
                        prompt_template=stage.generator_prompt_template,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p
                    )
                    if not output or not output.strip():
                        print(f"    Warning: Empty output at stage {stage_id}")
                
                reasoning_chain.append(output)
                
                # Generate tests (with caching)
                tests = self._generate_tests_cached(problem, output, stage_id, stage)
                
                if tests and tests.strip():
                    accumulated_tests.append(tests)
                else:
                    print(f"    Warning: Empty tests at stage {stage_id}")
        
        final_code = reasoning_chain[-1] if reasoning_chain else ""
        all_tests = "\n\n".join(accumulated_tests)
        
        result = (reasoning_chain, final_code, all_tests)
        
        # Cache result
        if not hasattr(self, '_generation_cache'):
            self._generation_cache = {}
        self._generation_cache[cache_key] = result
        
        return result
    
    def _generate_tests_cached(self, problem: Problem, output: str, stage_id: int, stage) -> str:
        """Generate tests with caching to avoid redundant generation."""
        # Create cache key from problem, output, and stage
        import hashlib
        output_hash = hashlib.md5(output.encode()).hexdigest()[:8]
        cache_key = (problem.id, stage_id, output_hash)
        
        if not hasattr(self, '_test_cache'):
            self._test_cache = {}
        
        if cache_key in self._test_cache:
            return self._test_cache[cache_key]
        
        # Generate tests
        tests = self.model.generate_tests(
            problem=problem.description,
            generator_code=output,
            num_tests=self.config.num_tests_per_problem,
            prompt_template=stage.discriminator_prompt_template,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature
        )
        
        # Cache for reuse
        self._test_cache[cache_key] = tests
        
        return tests
    
    def train_full_pipeline(self, problems: List[Problem]) -> Dict[str, Any]:
        """Train all stages sequentially."""
        print("\n" + "="*60)
        print("STARTING FULL PIPELINE TRAINING (UNIFIED MODEL)")
        print("="*60 + "\n")
        
        all_stage_metrics = []
        
        for stage_id in range(1, 6):
            self.current_stage = stage_id
            stage_metrics = self.train_stage(stage_id, problems, self.config.n_generator_steps)
            all_stage_metrics.append(stage_metrics)
            
            print(f"\nStage {stage_id} Summary:")
            print(f"  Generator Reward: {stage_metrics['avg_gen_reward']:.4f}")
            print(f"  Discriminator Reward: {stage_metrics['avg_disc_reward']:.4f}")
            print()
        
        final_metrics = {
            'stages': all_stage_metrics,
            'total_stages_trained': 5
        }
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60 + "\n")
        
        return final_metrics
