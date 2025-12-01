"""Main entry point for adversarial RL training."""

import argparse
import json
from pathlib import Path

from models.generator import LLMGenerator
from models.discriminator import LLMDiscriminator
from sandbox.sandbox import Sandbox
from data.problem_dataset import load_problems
from training.adversarial_trainer import AdversarialTrainer
from training.config import TrainingConfig
from training.checkpoint_manager import CheckpointManager


def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train adversarial RL multi-stage reasoning system"
    )
    parser.add_argument(
        '--generator-model',
        type=str,
        default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        help="HuggingFace model for generator"
    )
    parser.add_argument(
        '--discriminator-model',
        type=str,
        default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        help="HuggingFace model for discriminator"
    )
    parser.add_argument(
        '--device',
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on"
    )
    parser.add_argument(
        '--problems-file',
        type=str,
        default="data/function_problems.json",
        help="Path to problems JSON file"
    )
    parser.add_argument(
        '--n-discriminator-steps',
        type=int,
        default=2,
        help="Number of discriminator training steps per stage"
    )
    parser.add_argument(
        '--n-generator-steps',
        type=int,
        default=2,
        help="Number of generator training steps per stage"
    )
    parser.add_argument(
        '--k-alternating-steps',
        type=int,
        default=2,
        help="Number of alternating training steps per stage"
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-5,
        help="Learning rate for training"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="output",
        help="Directory to save results"
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default="checkpoints",
        help="Directory to save/load checkpoints"
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help="Resume training from latest checkpoint"
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help="Resume training from specific checkpoint file"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("ADVERSARIAL RL MULTI-STAGE REASONING SYSTEM")
    print("="*60 + "\n")
    
    # Create configuration
    config = TrainingConfig(
        generator_model=args.generator_model,
        discriminator_model=args.discriminator_model,
        device=args.device,
        n_discriminator_steps=args.n_discriminator_steps,
        n_generator_steps=args.n_generator_steps,
        k_alternating_steps=args.k_alternating_steps,
        learning_rate=args.learning_rate
    )
    
    print("Configuration:")
    print(f"  Generator Model: {config.generator_model}")
    print(f"  Discriminator Model: {config.discriminator_model}")
    print(f"  Device: {config.device}")
    print(f"  N Discriminator Steps: {config.n_discriminator_steps}")
    print(f"  N Generator Steps: {config.n_generator_steps}")
    print(f"  K Alternating Steps: {config.k_alternating_steps}")
    print(f"  Learning Rate: {config.learning_rate}")
    print()
    
    try:
        # Load problems
        print(f"Loading problems from {args.problems_file}...")
        problems = load_problems(args.problems_file)
        print(f"Loaded {len(problems)} problems\n")
        
        # Initialize models
        print("Initializing models...")
        generator = LLMGenerator(
            model_name=config.generator_model,
            device=config.device
        )
        print()
        
        discriminator = LLMDiscriminator(
            model_name=config.discriminator_model,
            device=config.device
        )
        print()
        
        # Initialize sandbox
        sandbox = Sandbox(timeout=config.sandbox_timeout)
        
        # Initialize checkpoint manager
        checkpoint_manager = CheckpointManager(checkpoint_dir=args.checkpoint_dir)
        
        # Create trainer
        print("Creating adversarial trainer...")
        trainer = AdversarialTrainer(
            generator=generator,
            discriminator=discriminator,
            sandbox=sandbox,
            config=config,
            checkpoint_manager=checkpoint_manager
        )
        print()
        
        # Handle resume logic
        start_stage = 1
        if args.resume or args.resume_from:
            print("Attempting to resume from checkpoint...")
            start_stage = trainer.resume_from_checkpoint(args.resume_from)
            print()
        
        # Run training
        print("Starting training...\n")
        if start_stage > 1:
            results = trainer.train_full_pipeline_with_resume(problems, start_stage)
        else:
            results = trainer.train_full_pipeline(problems)
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results_file = output_dir / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Total Stages Trained: {results['total_stages_trained']}")
        print(f"Final Generator Reward: {results['final_generator_reward']:.4f}")
        print(f"Final Discriminator Reward: {results['final_discriminator_reward']:.4f}")
        print()
        
        print("Stage-by-Stage Results:")
        for stage_metrics in results['stages']:
            print(f"\n  Stage {stage_metrics['stage_id']}: {stage_metrics['stage_name']}")
            print(f"    Generator Reward: {stage_metrics['generator']['avg_reward']:.4f}")
            print(f"    Discriminator Reward: {stage_metrics['discriminator']['avg_reward']:.4f}")
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
