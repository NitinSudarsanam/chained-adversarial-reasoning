"""Simplified trainer using a single unified model for both generator and discriminator."""

import torch
import random
from typing import List, Dict, Any
from tqdm import tqdm

from models.unified_model import UnifiedModel
from data.problem_dataset import Problem
from reasoning.stages import get_stage
from training.reward import compute_generator_reward, compute_discriminator_reward, run_code_tests
from training.rl_loop import train_step, create_optimizer, freeze_model, unfreeze_model
from training.config import TrainingConfig
from training.checkpoint_manager import CheckpointManager


class UnifiedTrainer:
    """Trainer using a single model for both generation and discrimination."""
    
    def __init__(
        self,
        model: UnifiedModel,
        config: TrainingConfig,
        checkpoint_manager: CheckpointManager = None
    ):
        """Initialize unified trainer.
        
        Args:
            model: Unified model (acts as both generator and discriminator)
            config: Training configuration
            checkpoint_manager: Optional checkpoint manager
        """
        self.model = model
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
        
        # Problem sampling state (for random sampling without replacement)
        self.problem_indices = []
        self.problem_index_ptr = 0
    
    def clear_caches(self):
        """Clear generation and test caches to free memory."""
        self._generation_cache.clear()
        self._test_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
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
            # Sample problem (random without replacement)
            problem = self._sample_problem(problems)
            
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
        
        # Save checkpoint after each stage
        checkpoint_metrics = {
            'generator_reward': avg_gen_reward,
            'discriminator_reward': avg_disc_reward,
            'test_validity': 0.9
        }
        is_best = self.checkpoint_manager.should_save_as_best(checkpoint_metrics)
        
        # Save unified model checkpoint
        self._save_unified_checkpoint(stage_id, n_steps, checkpoint_metrics, is_best)
        
        return stage_metrics
    
    def _save_unified_checkpoint(self, stage: int, epoch: int, metrics: Dict, is_best: bool = False):
        """Save checkpoint for unified model.
        
        Args:
            stage: Current training stage (1-5)
            epoch: Current epoch/step number
            metrics: Training metrics
            is_best: Whether this is the best checkpoint
        """
        from datetime import datetime
        import torch
        
        timestamp = datetime.now().isoformat()
        
        # Create checkpoint filename
        checkpoint_name = f"unified_checkpoint_stage_{stage}_epoch_{epoch}.pt"
        checkpoint_path = self.checkpoint_manager.checkpoint_dir / checkpoint_name
        
        # Move state dicts to CPU to prevent GPU memory spike
        # Save only LoRA adapters if using LoRA (much smaller)
        print("  Preparing checkpoint data (moving to CPU)...")
        
        if self.model.use_lora:
            # Save only LoRA adapter weights (both generator and discriminator)
            model_state = self.model.model.state_dict()
            model_state_cpu = {k: v.cpu().clone() for k, v in model_state.items()
                              if 'lora_' in k}
            del model_state
            print(f"    Saving LoRA adapters only ({len(model_state_cpu)} parameters)")
        else:
            # Save full model state (fallback)
            model_state = self.model.model.state_dict()
            model_state_cpu = {k: v.cpu().clone() for k, v in model_state.items()}
            del model_state
            print(f"    Saving full model state ({len(model_state_cpu)} parameters)")
        
        optimizer_state = self.optimizer.state_dict()
        optimizer_state_cpu = {}
        for k, v in optimizer_state.items():
            if isinstance(v, torch.Tensor):
                optimizer_state_cpu[k] = v.cpu()
            else:
                optimizer_state_cpu[k] = v
        del optimizer_state  # Free GPU memory immediately
        
        # Prepare checkpoint data (all on CPU now)
        checkpoint_data = {
            "stage": stage,
            "epoch": epoch,
            "timestamp": timestamp,
            "model_state_dict": model_state_cpu,
            "optimizer_state_dict": optimizer_state_cpu,
            "metrics": metrics,
            "is_lora": self.model.use_lora,
            "config": {
                "model_name": self.model.model_name,
                "device": self.model.device,
                "use_lora": self.model.use_lora,
                "current_adapter": self.model.current_adapter
            }
        }
        
        # Save checkpoint
        try:
            torch.save(checkpoint_data, checkpoint_path)
            print(f"  ✓ Saved checkpoint: {checkpoint_path.name}")
        except Exception as e:
            print(f"  ✗ Failed to save checkpoint: {e}")
            del checkpoint_data, model_state_cpu, optimizer_state_cpu
            return
        
        # Clean up CPU memory
        del checkpoint_data, model_state_cpu, optimizer_state_cpu
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Save metadata
        metadata_path = checkpoint_path.with_suffix('.json')
        import json
        metadata = {
            "stage": stage,
            "epoch": epoch,
            "timestamp": timestamp,
            "metrics": metrics,
            "checkpoint_path": str(checkpoint_path),
            "is_best": is_best
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update best checkpoint if needed
        if is_best:
            best_path = self.checkpoint_manager.checkpoint_dir / "unified_checkpoint_best.pt"
            best_metadata_path = self.checkpoint_manager.checkpoint_dir / "unified_checkpoint_best.json"
            
            import shutil
            shutil.copy2(checkpoint_path, best_path)
            shutil.copy2(metadata_path, best_metadata_path)
            print(f"  ✓ Updated best checkpoint")
    
    def _train_discriminator_step(self, problem: Problem, stage_id: int) -> Dict:
        """Single discriminator training step."""
        import time
        step_start = time.time()
        
        # Generate only up to target stage (not all 5!)
        gen_start = time.time()
        reasoning_chain, final_code, accumulated_tests = self._generate_up_to_stage(problem, stage_id)
        gen_time = time.time() - gen_start
        
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
        
        # Generate tests
        test_start = time.time()
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
        test_time = time.time() - test_start
        
        if not stage_tests or not stage_tests.strip():
            return None
        
        # Get log probs with discriminator adapter active
        logprob_start = time.time()
        old_log_probs = self.model.get_log_probs(prompt, stage_tests)
        logprob_time = time.time() - logprob_start
        
        # Compute rewards using run_code_tests
        rewards = run_code_tests(final_code, accumulated_tests, problem.reference_solution)
        reward = rewards.discriminator_reward
        
        # Calculate pass percentages
        gen_pass_pct = (gen_result.num_passed / gen_result.num_total * 100) if gen_result.num_total > 0 else 0.0
        val_pass_pct = (val_result.num_passed / val_result.num_total * 100) if val_result.num_total > 0 else 0.0
        
        # Log reward with percentages
        print(f"    Discriminator Reward: {reward:.4f}")
        print(f"      Gen Pass: {gen_result.num_passed}/{gen_result.num_total} ({gen_pass_pct:.1f}%)")
        print(f"      Val Pass: {val_result.num_passed}/{val_result.num_total} ({val_pass_pct:.1f}%)")
        
        # Train
        train_start = time.time()
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
        train_time = time.time() - train_start
        
        total_time = time.time() - step_start
        
        # Add timing info
        metrics['reward'] = reward
        metrics['timing'] = {
            'generation': gen_time,
            'test_gen': test_time,
            'log_probs': logprob_time,
            'execution': exec_time,
            'training': train_time,
            'total': total_time
        }
        
        # Print timing breakdown if slow
        if total_time > 60:
            print(f"\n  ⚠ Slow step ({total_time:.1f}s):")
            print(f"    Generation: {gen_time:.1f}s")
            print(f"    Test gen: {test_time:.1f}s")
            print(f"    Log probs: {logprob_time:.1f}s")
            print(f"    Execution: {exec_time:.1f}s")
            print(f"    Training: {train_time:.1f}s")
        
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
        
        # Compute rewards using run_code_tests
        rewards = run_code_tests(final_code, accumulated_tests, problem.reference_solution)
        reward = rewards.generator_reward
        
        # Calculate pass percentage
        pass_pct = (result.num_passed / result.num_total * 100) if result.num_total > 0 else 0.0
        
        # Log reward with percentage
        print(f"    Generator Reward: {reward:.4f}")
        print(f"      Test Pass: {result.num_passed}/{result.num_total} ({pass_pct:.1f}%)")
        
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
        """Generate full reasoning chain (all 5 stages) for a problem.
        
        NOTE: We MUST generate all 5 stages because:
        - Tests are executed against final_code (stage 5)
        - We need executable code to validate tests
        - Even when training stage 1, we need stage 5 code for execution
        
        OPTIMIZATION: Cache results to avoid regenerating for same problem+stage
        """
        # Check cache first - AGGRESSIVE CACHING
        cache_key = (problem.id, target_stage)
        if hasattr(self, '_generation_cache') and cache_key in self._generation_cache:
            cached = self._generation_cache[cache_key]
            # Return cached result immediately
            return cached
        
        reasoning_chain = []
        accumulated_tests = []
        
        self.model.eval()
        with torch.no_grad():
            # Generate all 5 stages (needed for final executable code)
            for stage_id in range(1, 6):
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
