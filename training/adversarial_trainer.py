"""Adversarial trainer for multi-stage reasoning system."""

import torch
import random
from typing import List, Dict, Any
from tqdm import tqdm

from models.generator import LLMGenerator
from models.discriminator import LLMDiscriminator
from data.problem_dataset import Problem
from reasoning.stages import get_stage
from training.reward import compute_generator_reward, compute_discriminator_reward, run_code_tests
from training.rl_loop import (
    train_step,
    create_optimizer,
    freeze_model,
    unfreeze_model
)
from training.config import TrainingConfig
from training.checkpoint_manager import CheckpointManager


class AdversarialTrainer:
    """Orchestrates adversarial training between generator and discriminator."""
    
    def __init__(
        self,
        generator: LLMGenerator,
        discriminator: LLMDiscriminator,
        config: TrainingConfig,
        checkpoint_manager: CheckpointManager = None
    ):
        """Initialize adversarial trainer.
        
        Args:
            generator: Generator LLM
            discriminator: Discriminator LLM
            config: Training configuration
            checkpoint_manager: Optional checkpoint manager for saving/loading
        """
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        
        # Create optimizers
        self.gen_optimizer = create_optimizer(
            generator,
            config.learning_rate
        )
        self.disc_optimizer = create_optimizer(
            discriminator,
            config.learning_rate
        )
        
        # Training state
        self.current_stage = 1
        self.metrics_history = {
            'generator': [],
            'discriminator': []
        }
        
        # Problem sampling state (for random sampling without replacement)
        self.problem_indices = []
        self.problem_index_ptr = 0
        
        # Generation cache for efficiency (avoid regenerating same content)
        self._generation_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _sample_problem(self, problems: List[Problem]) -> Problem:
        """Sample problem with random shuffling (without replacement per epoch).
        
        Args:
            problems: List of available problems
            
        Returns:
            Sampled problem
        """
        # Initialize or reshuffle when needed
        if not self.problem_indices or self.problem_index_ptr >= len(problems):
            # Reshuffle when we've used all problems or first time
            self.problem_indices = list(range(len(problems)))
            random.shuffle(self.problem_indices)
            self.problem_index_ptr = 0
            print(f"  Shuffled {len(problems)} problems for new epoch")
        
        idx = self.problem_indices[self.problem_index_ptr]
        self.problem_index_ptr += 1
        return problems[idx]
    
    def train_discriminator_epoch(
        self,
        stage_id: int,
        problems: List[Problem],
        n_steps: int
    ) -> Dict[str, float]:
        """Train discriminator with frozen generator.
        
        Args:
            stage_id: Current reasoning stage (1-5)
            problems: List of problems to train on
            n_steps: Number of training steps
            
        Returns:
            Dictionary of metrics
        """
        print(f"Training discriminator at stage {stage_id} for {n_steps} steps...")
        
        # Freeze generator
        freeze_model(self.generator)
        unfreeze_model(self.discriminator)
        
        total_reward = 0.0
        total_loss = 0.0
        num_updates = 0
        num_skipped = 0
        
        for step in tqdm(range(n_steps), desc="Discriminator"):
            # Sample a problem (random without replacement)
            problem = self._sample_problem(problems)
            
            # Generate full reasoning chain and accumulate tests from all stages
            reasoning_chain, final_code, accumulated_tests = self._generate_full_chain_with_tests(
                problem, stage_id
            )
            
            if not final_code or not final_code.strip():
                num_skipped += 1
                print(f"  ⚠ Skipping discriminator step {step+1}/{n_steps}: empty code generated")
                print(f"     Reasoning chain length: {len(reasoning_chain)}, Problem: {problem.id}")
                continue
            
            if not accumulated_tests or not accumulated_tests.strip():
                num_skipped += 1
                print(f"  ⚠ Skipping discriminator step {step+1}/{n_steps}: empty tests generated")
                print(f"     Final code length: {len(final_code)}, Problem: {problem.id}")
                continue
            
            # Get the tests generated at THIS stage for log probs
            stage_output = reasoning_chain[stage_id - 1] if stage_id <= len(reasoning_chain) else final_code
            stage = get_stage(stage_id)
            
            # Generate tests for this stage to get log probs
            prompt = stage.discriminator_prompt_template.format(
                problem=problem.description,
                stage_output=stage_output,
                num_tests=self.config.num_tests_per_problem
            )
            
            stage_tests = self.discriminator.generate_tests(
                problem=problem.description,
                generator_code=stage_output,
                num_tests=self.config.num_tests_per_problem,
                prompt_template=stage.discriminator_prompt_template,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature
            )
            
            if not stage_tests or not stage_tests.strip():
                num_skipped += 1
                print(f"  ⚠ Skipping discriminator step {step+1}/{n_steps}: empty stage tests generated")
                print(f"     Stage output length: {len(stage_output)}, Problem: {problem.id}")
                continue
            
            old_log_probs = self.discriminator.get_log_probs(prompt, stage_tests)
            
            # Compute reward using run_code_tests
            rewards = run_code_tests(final_code, accumulated_tests, problem.reference_solution)
            reward = rewards.discriminator_reward
            gen_result = rewards.gen_result
            val_result = rewards.val_result
            total_reward += reward
            
            # Calculate pass percentages
            gen_pass_pct = (gen_result.num_passed / gen_result.num_total * 100) if gen_result.num_total > 0 else 0.0
            val_pass_pct = (val_result.num_passed / val_result.num_total * 100) if val_result.num_total > 0 else 0.0
            
            # Log reward for this step with details
            print(f"\n  Step {step+1}/{n_steps} - Discriminator Reward: {reward:.4f}")
            print(f"    Generator Pass Rate: {gen_result.num_passed}/{gen_result.num_total} ({gen_pass_pct:.1f}%)")
            print(f"    Validation Pass Rate: {val_result.num_passed}/{val_result.num_total} ({val_pass_pct:.1f}%)")
            
            # Log generated code and tests for debugging
            print(f"  Generated Code (length={len(final_code)}):")
            code_preview = final_code[:300].replace('\n', '\n    ')
            print(f"    {code_preview}{'...' if len(final_code) > 300 else ''}")
            
            print(f"  Generated Tests (length={len(accumulated_tests)}):")
            tests_preview = accumulated_tests[:300].replace('\n', '\n    ')
            print(f"    {tests_preview}{'...' if len(accumulated_tests) > 300 else ''}")
            
            # Train step
            metrics = train_step(
                model=self.discriminator,
                optimizer=self.disc_optimizer,
                prompts=[prompt],
                outputs=[stage_tests],
                rewards=[reward],
                old_log_probs_list=[old_log_probs],
                clip_epsilon=self.config.clip_epsilon
            )
            
            if 'policy_loss' in metrics and not torch.isnan(torch.tensor(metrics['policy_loss'])):
                total_loss += metrics['policy_loss']
                num_updates += 1
        
        # Calculate average over successful steps only
        num_successful = n_steps - num_skipped
        avg_reward = total_reward / num_successful if num_successful > 0 else 0.0
        avg_loss = total_loss / num_updates if num_updates > 0 else 0.0
        
        # Log summary
        print(f"\n  Discriminator Training Summary:")
        print(f"    Total steps: {n_steps}")
        print(f"    Successful steps: {num_successful}")
        print(f"    Skipped steps: {num_skipped}")
        print(f"    Updates applied: {num_updates}")
        print(f"    Average reward: {avg_reward:.4f}")
        print(f"    Average loss: {avg_loss:.4f}")
        
        # Log cache efficiency
        total_cache_attempts = self._cache_hits + self._cache_misses
        if total_cache_attempts > 0:
            cache_hit_rate = (self._cache_hits / total_cache_attempts) * 100
            print(f"    Cache: {self._cache_hits} hits, {self._cache_misses} misses ({cache_hit_rate:.1f}% hit rate)")
        
        if num_skipped > 0:
            print(f"  ⚠ Warning: {num_skipped}/{n_steps} steps skipped due to empty generation")
        
        if num_successful == 0:
            print(f"  ❌ ERROR: All steps were skipped! Check model generation.")
        
        return {
            'avg_reward': avg_reward,
            'avg_loss': avg_loss,
            'num_updates': num_updates,
            'num_skipped': num_skipped
        }

    
    def train_generator_epoch(
        self,
        stage_id: int,
        problems: List[Problem],
        n_steps: int
    ) -> Dict[str, float]:
        """Train generator with frozen discriminator.
        
        Args:
            stage_id: Current reasoning stage (1-5)
            problems: List of problems to train on
            n_steps: Number of training steps
            
        Returns:
            Dictionary of metrics
        """
        print(f"Training generator at stage {stage_id} for {n_steps} steps...")
        
        # Freeze discriminator
        freeze_model(self.discriminator)
        unfreeze_model(self.generator)
        
        total_reward = 0.0
        total_loss = 0.0
        num_updates = 0
        num_skipped = 0
        
        for step in tqdm(range(n_steps), desc="Generator"):
            # Sample a problem (random without replacement)
            problem = self._sample_problem(problems)
            
            # Generate full reasoning chain and accumulate tests from all stages
            reasoning_chain, final_code, accumulated_tests = self._generate_full_chain_with_tests(
                problem, stage_id
            )
            
            if not final_code or not final_code.strip():
                num_skipped += 1
                print(f"  ⚠ Skipping generator step {step+1}/{n_steps}: empty code generated")
                print(f"     Reasoning chain length: {len(reasoning_chain)}, Problem: {problem.id}")
                continue
            
            if not accumulated_tests or not accumulated_tests.strip():
                num_skipped += 1
                print(f"  ⚠ Skipping generator step {step+1}/{n_steps}: empty tests generated")
                print(f"     Final code length: {len(final_code)}, Problem: {problem.id}")
                continue
            
            # Get old log probs for THIS stage's generation
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
            old_log_probs = self.generator.get_log_probs(prompt, stage_output)
            
            # Execute ALL accumulated tests against final code
            rewards = run_code_tests(final_code, accumulated_tests, problem.reference_solution)
            reward = rewards.generator_reward
            result = rewards.gen_result
            total_reward += reward
            
            # Calculate pass percentage
            pass_pct = (result.num_passed / result.num_total * 100) if result.num_total > 0 else 0.0
            
            # Log reward for this step with details
            print(f"\n  Step {step+1}/{n_steps} - Generator Reward: {reward:.4f}")
            print(f"    Test Pass Rate: {result.num_passed}/{result.num_total} ({pass_pct:.1f}%)")
            
            # Log generated code and tests for debugging
            print(f"  Generated Code (length={len(final_code)}):")
            code_preview = final_code[:300].replace('\n', '\n    ')
            print(f"    {code_preview}{'...' if len(final_code) > 300 else ''}")
            
            print(f"  Generated Tests (length={len(accumulated_tests)}):")
            tests_preview = accumulated_tests[:300].replace('\n', '\n    ')
            print(f"    {tests_preview}{'...' if len(accumulated_tests) > 300 else ''}")
            
            # Train step
            metrics = train_step(
                model=self.generator,
                optimizer=self.gen_optimizer,
                prompts=[prompt],
                outputs=[stage_output],
                rewards=[reward],
                old_log_probs_list=[old_log_probs],
                clip_epsilon=self.config.clip_epsilon
            )
            
            if 'policy_loss' in metrics and not torch.isnan(torch.tensor(metrics['policy_loss'])):
                total_loss += metrics['policy_loss']
                num_updates += 1
        
        # Calculate average over successful steps only
        num_successful = n_steps - num_skipped
        avg_reward = total_reward / num_successful if num_successful > 0 else 0.0
        avg_loss = total_loss / num_updates if num_updates > 0 else 0.0
        
        # Log summary
        print(f"\n  Generator Training Summary:")
        print(f"    Total steps: {n_steps}")
        print(f"    Successful steps: {num_successful}")
        print(f"    Skipped steps: {num_skipped}")
        print(f"    Updates applied: {num_updates}")
        print(f"    Average reward: {avg_reward:.4f}")
        print(f"    Average loss: {avg_loss:.4f}")
        
        # Log cache efficiency
        total_cache_attempts = self._cache_hits + self._cache_misses
        if total_cache_attempts > 0:
            cache_hit_rate = (self._cache_hits / total_cache_attempts) * 100
            print(f"    Cache: {self._cache_hits} hits, {self._cache_misses} misses ({cache_hit_rate:.1f}% hit rate)")
        
        if num_skipped > 0:
            print(f"  ⚠ Warning: {num_skipped}/{n_steps} steps skipped due to empty generation")
        
        if num_successful == 0:
            print(f"  ❌ ERROR: All steps were skipped! Check model generation.")
        
        return {
            'avg_reward': avg_reward,
            'avg_loss': avg_loss,
            'num_updates': num_updates,
            'num_skipped': num_skipped
        }
    
    def train_alternating(
        self,
        stage_id: int,
        problems: List[Problem],
        k_steps: int
    ) -> Dict[str, float]:
        """Alternate between generator and discriminator training.
        
        Args:
            stage_id: Current reasoning stage (1-5)
            problems: List of problems to train on
            k_steps: Number of alternating steps
            
        Returns:
            Dictionary of metrics
        """
        print(f"Alternating training at stage {stage_id} for {k_steps} steps...")
        
        gen_rewards = []
        disc_rewards = []
        
        for step in tqdm(range(k_steps), desc="Alternating"):
            if step % 2 == 0:
                # Train generator
                metrics = self.train_generator_epoch(stage_id, problems, 1)
                gen_rewards.append(metrics['avg_reward'])
            else:
                # Train discriminator
                metrics = self.train_discriminator_epoch(stage_id, problems, 1)
                disc_rewards.append(metrics['avg_reward'])
        
        return {
            'gen_avg_reward': sum(gen_rewards) / len(gen_rewards) if gen_rewards else 0.0,
            'disc_avg_reward': sum(disc_rewards) / len(disc_rewards) if disc_rewards else 0.0
        }

    
    def _generate_reasoning_chain(
        self,
        problem: Problem,
        target_stage: int
    ) -> tuple:
        """Generate reasoning chain up to target stage.
        
        Args:
            problem: Problem to solve
            target_stage: Target stage (1-5)
            
        Returns:
            Tuple of (reasoning_chain, final_output)
        """
        reasoning_chain = []
        
        for stage_id in range(1, target_stage + 1):
            stage = get_stage(stage_id)
            
            if stage_id == 5:
                # Final code generation
                output = self.generator.generate_code(
                    problem=problem.description,
                    reasoning_chain=reasoning_chain,
                    prompt_template=stage.generator_prompt_template,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    function_signature=problem.function_signature
                )
            else:
                # Intermediate reasoning stage
                output = self.generator.generate_stage_output(
                    problem=problem.description,
                    previous_stages=reasoning_chain,
                    stage_id=stage_id,
                    prompt_template=stage.generator_prompt_template,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p
                )
            
            reasoning_chain.append(output)
        
        final_output = reasoning_chain[-1] if reasoning_chain else ""
        
        return reasoning_chain, final_output
    
    def _generate_full_chain_with_tests(
        self,
        problem: Problem,
        training_stage: int
    ) -> tuple:
        """Generate full reasoning chain (1-5) and accumulate tests from all stages.
        
        NOTE: We MUST generate all 5 stages because:
        - Tests are executed against final_code (stage 5)
        - We need executable code to validate tests
        - Even when training stage 1, we need stage 5 code for execution
        
        OPTIMIZATION: Cache results to avoid regenerating for same problem+stage
        
        Args:
            problem: Problem to solve
            training_stage: Which stage is being trained (1-5)
            
        Returns:
            Tuple of (reasoning_chain, final_code, accumulated_tests)
        """
        # OPTIMIZATION: Check cache first
        cache_key = (problem.id, training_stage)
        if cache_key in self._generation_cache:
            self._cache_hits += 1
            return self._generation_cache[cache_key]
        
        self._cache_misses += 1
        
        reasoning_chain = []
        accumulated_tests = []
        
        # Generate all 5 stages (needed for final executable code)
        for stage_id in range(1, 6):
            stage = get_stage(stage_id)
            
            # Use no_grad for stages before training_stage (frozen)
            use_grad = stage_id >= training_stage
            
            # Always use eval mode and no_grad for generation
            # We only need gradients for get_log_probs, not generation
            self.generator.eval()
            with torch.no_grad():
                if stage_id == 5:
                    output = self.generator.generate_code(
                        problem=problem.description,
                        reasoning_chain=reasoning_chain,
                        prompt_template=stage.generator_prompt_template,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        function_signature=problem.function_signature
                    )
                else:
                    output = self.generator.generate_stage_output(
                        problem=problem.description,
                        previous_stages=reasoning_chain,
                        stage_id=stage_id,
                        prompt_template=stage.generator_prompt_template,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p
                    )
            
            # Restore training mode if this stage is trainable
            if use_grad:
                self.generator.train()
            
            reasoning_chain.append(output)
            
            # Generate tests for this stage
            # Always use eval mode and no_grad for generation (even if trainable)
            # We only need gradients for get_log_probs, not generation
            self.discriminator.eval()
            with torch.no_grad():
                tests = self.discriminator.generate_tests(
                    problem=problem.description,
                    generator_code=output,
                    num_tests=self.config.num_tests_per_problem,
                    prompt_template=stage.discriminator_prompt_template,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature
                )
            
            # Restore training mode if this stage is trainable
            if use_grad:
                self.discriminator.train()
            
            # Old code kept for reference but not used:
            if False and not use_grad:
                # Frozen test generation
                with torch.no_grad():
                    tests = self.discriminator.generate_tests(
                        problem=problem.description,
                        generator_code=output,
                        num_tests=self.config.num_tests_per_problem,
                        prompt_template=stage.discriminator_prompt_template,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature
                    )
            
            if tests and tests.strip():
                accumulated_tests.append(tests)
        
        final_code = reasoning_chain[-1] if reasoning_chain else ""
        all_tests = "\n\n".join(accumulated_tests)
        
        # Cache the result for reuse
        result = (reasoning_chain, final_code, all_tests)
        self._generation_cache[cache_key] = result
        
        return result
    
    def clear_cache(self):
        """Clear generation cache to free memory."""
        cache_size = len(self._generation_cache)
        self._generation_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        if cache_size > 0:
            print(f"  Cleared generation cache ({cache_size} entries)")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def train_stage(
        self,
        stage_id: int,
        problems: List[Problem]
    ) -> Dict[str, Any]:
        """Train both models at a specific reasoning stage.
        
        Implements the N+N+K training pattern:
        - N steps training discriminator (frozen generator)
        - N steps training generator (frozen discriminator)
        - K steps alternating between both
        
        Args:
            stage_id: Stage ID (1-5)
            problems: List of problems to train on
            
        Returns:
            Dictionary of stage metrics
        """
        print(f"\n{'='*60}")
        print(f"Training Stage {stage_id}: {get_stage(stage_id).name}")
        print(f"{'='*60}\n")
        
        # Clear cache at start of each stage to free memory
        self.clear_cache()
        
        # Phase 1: Train discriminator
        disc_metrics = self.train_discriminator_epoch(
            stage_id=stage_id,
            problems=problems,
            n_steps=self.config.n_discriminator_steps
        )
        
        # Phase 2: Train generator
        gen_metrics = self.train_generator_epoch(
            stage_id=stage_id,
            problems=problems,
            n_steps=self.config.n_generator_steps
        )
        
        # Phase 3: Alternating training
        alt_metrics = self.train_alternating(
            stage_id=stage_id,
            problems=problems,
            k_steps=self.config.k_alternating_steps
        )
        
        # Aggregate metrics
        stage_metrics = {
            'stage_id': stage_id,
            'stage_name': get_stage(stage_id).name,
            'discriminator': disc_metrics,
            'generator': gen_metrics,
            'alternating': alt_metrics
        }
        
        # Store in history
        self.metrics_history['generator'].append(gen_metrics)
        self.metrics_history['discriminator'].append(disc_metrics)
        
        # Prepare metrics for checkpoint
        checkpoint_metrics = {
            'generator_reward': gen_metrics['avg_reward'],
            'discriminator_reward': disc_metrics['avg_reward'],
            'test_validity': 0.9  # Placeholder - would need actual validation
        }
        
        # Check if this is the best checkpoint
        is_best = self.checkpoint_manager.should_save_as_best(checkpoint_metrics)
        
        # Save checkpoint after stage
        self.checkpoint_manager.save_checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            stage=stage_id,
            epoch=self.config.n_generator_steps,  # Using n_steps as epoch
            metrics=checkpoint_metrics,
            is_best=is_best
        )
        
        return stage_metrics

    
    def train_full_pipeline(
        self,
        problems: List[Problem]
    ) -> Dict[str, Any]:
        """Train all stages sequentially from bottom to top.
        
        Trains stages 1-5 in order, fully completing each stage before
        moving to the next. This ensures proper dependency ordering.
        
        Args:
            problems: List of problems to train on
            
        Returns:
            Dictionary of all training metrics
        """
        print("\n" + "="*60)
        print("STARTING FULL PIPELINE TRAINING")
        print("="*60 + "\n")
        
        all_stage_metrics = []
        
        # Train each stage sequentially
        for stage_id in range(1, 6):
            self.current_stage = stage_id
            
            stage_metrics = self.train_stage(stage_id, problems)
            all_stage_metrics.append(stage_metrics)
            
            # Print stage summary
            print(f"\nStage {stage_id} Summary:")
            print(f"  Generator Reward: {stage_metrics['generator']['avg_reward']:.4f}")
            print(f"  Discriminator Reward: {stage_metrics['discriminator']['avg_reward']:.4f}")
            print()
        
        # Aggregate final metrics
        final_metrics = {
            'stages': all_stage_metrics,
            'total_stages_trained': 5,
            'final_generator_reward': all_stage_metrics[-1]['generator']['avg_reward'],
            'final_discriminator_reward': all_stage_metrics[-1]['discriminator']['avg_reward']
        }
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print(f"Best checkpoint: {self.checkpoint_manager.get_best_checkpoint()}")
        print("="*60 + "\n")
        
        return final_metrics


    def resume_from_checkpoint(
        self,
        checkpoint_path: str = None
    ) -> int:
        """Resume training from a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file. If None, uses latest checkpoint.
            
        Returns:
            Stage number to resume from (next stage after checkpoint)
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_manager.get_latest_checkpoint()
            if checkpoint_path is None:
                print("No checkpoint found to resume from")
                return 1
        
        # Load checkpoint
        metadata = self.checkpoint_manager.load_checkpoint(
            checkpoint_path,
            self.generator,
            self.discriminator
        )
        
        # Return next stage to train
        completed_stage = metadata['stage']
        next_stage = completed_stage + 1 if completed_stage < 5 else 5
        
        print(f"Resuming training from stage {next_stage}")
        return next_stage
    
    def train_full_pipeline_with_resume(
        self,
        problems: List[Problem],
        start_stage: int = 1
    ) -> Dict[str, Any]:
        """Train all stages with support for resuming from a specific stage.
        
        Args:
            problems: List of problems to train on
            start_stage: Stage to start training from (1-5)
            
        Returns:
            Dictionary of all training metrics
        """
        print("\n" + "="*60)
        print(f"STARTING TRAINING FROM STAGE {start_stage}")
        print("="*60 + "\n")
        
        all_stage_metrics = []
        
        # Train from start_stage to stage 5
        for stage_id in range(start_stage, 6):
            self.current_stage = stage_id
            
            stage_metrics = self.train_stage(stage_id, problems)
            all_stage_metrics.append(stage_metrics)
            
            # Print stage summary
            print(f"\nStage {stage_id} Summary:")
            print(f"  Generator Reward: {stage_metrics['generator']['avg_reward']:.4f}")
            print(f"  Discriminator Reward: {stage_metrics['discriminator']['avg_reward']:.4f}")
            print()
        
        # Aggregate final metrics
        final_metrics = {
            'stages': all_stage_metrics,
            'total_stages_trained': len(all_stage_metrics),
            'final_generator_reward': all_stage_metrics[-1]['generator']['avg_reward'],
            'final_discriminator_reward': all_stage_metrics[-1]['discriminator']['avg_reward']
        }
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print(f"Best checkpoint: {self.checkpoint_manager.get_best_checkpoint()}")
        print("="*60 + "\n")
        
        return final_metrics
