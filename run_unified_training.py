"""Run training with unified model (single model for both generator and discriminator)."""

import argparse
import json
from pathlib import Path
import torch
import gc

# Clear GPU memory before starting
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print("Cleared GPU cache")

from models.unified_model import UnifiedModel
from data.problem_dataset import load_problems
from training.unified_trainer import UnifiedTrainer
from training.config import TrainingConfig
from training.checkpoint_manager import CheckpointManager


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train unified model (single model for both tasks)"
    )
    parser.add_argument(
        '--model',
        type=str,
        default="Qwen/Qwen2.5-Coder-3B-Instruct",  # Small model that fits in 15GB GPU
        help="HuggingFace model"
    )
    parser.add_argument(
        '--device',
        type=str,
        default="cuda",
        help="Device (cuda or cpu)"
    )
    parser.add_argument(
        '--no-lora',
        action='store_true',
        help="Disable LoRA (use full model fine-tuning, requires more memory)"
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help="Fast mode: reduce tokens, tests, and timeout for 2-3x speedup"
    )
    parser.add_argument(
        '--problems',
        type=str,
        default="data/function_problems.json",
        help="Path to problems JSON file"
    )
    parser.add_argument(
        '--n-steps',
        type=int,
        default=2,
        help="Number of training steps per stage"
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-5,
        help="Learning rate"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="outputs",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("UNIFIED MODEL TRAINING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Device: {args.device}")
    print(f"  Problems: {args.problems}")
    print(f"  Steps per stage: {args.n_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    
    # Show GPU memory status
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"\nGPU Memory:")
        print(f"  Total: {total:.2f} GiB")
        print(f"  Allocated: {allocated:.2f} GiB")
        print(f"  Reserved: {reserved:.2f} GiB")
        print(f"  Free: {total - reserved:.2f} GiB")
    print()
    
    # Load problems
    print("Loading problems...")
    problems = load_problems(args.problems)
    print(f"Loaded {len(problems)} problems\n")
    
    # Initialize unified model
    print("Initializing unified model...")
    use_lora = not args.no_lora
    if use_lora:
        print("  Using LoRA for efficient training (93% less memory)")
    else:
        print("  Using full model fine-tuning (requires more memory)")
    model = UnifiedModel(args.model, device=args.device, use_lora=use_lora)
    print()
    
    # Create config
    # Apply fast mode settings if requested
    if args.fast:
        print("  ⚡ Fast mode enabled: 2-3x speedup")
        num_tests = 1
        max_tokens = 64
        timeout = 2
    else:
        num_tests = 2
        max_tokens = 128
        timeout = 3
    
    config = TrainingConfig(
        n_discriminator_steps=args.n_steps,
        n_generator_steps=args.n_steps,
        k_alternating_steps=0,  # Not used in unified trainer
        learning_rate=args.learning_rate,
        num_tests_per_problem=num_tests,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        clip_epsilon=0.2
    )
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_dir="checkpoints_unified")
    
    # Create trainer
    trainer = UnifiedTrainer(
        model=model,
        config=config,
        checkpoint_manager=checkpoint_manager
    )
    
    # Train
    try:
        results = trainer.train_full_pipeline(problems)
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "unified_training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {results_file}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
