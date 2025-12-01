"""Minimal training run with code logging for verification."""

import os
import json
from pathlib import Path
from datetime import datetime

from models.generator import LLMGenerator
from models.discriminator import LLMDiscriminator
from training.adversarial_trainer import AdversarialTrainer, TrainingConfig
from data.problem_dataset import load_problems
from sandbox.sandbox import Sandbox

# Create output directory for generated code
output_dir = Path("training_output")
output_dir.mkdir(exist_ok=True)
run_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir.mkdir(exist_ok=True)

print(f"Output directory: {run_dir}")
print("="*80)

# Minimal configuration - using larger model for better results
config = TrainingConfig(
    generator_model="Qwen/Qwen2.5-Coder-1.5B-Instruct",  # Larger model (was 350M)
    discriminator_model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    device="cuda",
    n_discriminator_steps=1,  # Minimal
    n_generator_steps=1,      # Minimal
    k_alternating_steps=1,    # 2 iterations (was 1)
    max_new_tokens=512,       # Good balance
    temperature=0.7,          # Higher for more creativity
    learning_rate=1e-5
)

# Load just 2 problems for speed
print("Loading problems...")
all_problems = load_problems("data/function_problems.json")
problems = all_problems[:2]  # Just first 2 problems
print(f"Using {len(problems)} problems: {[p.id for p in problems]}\n")

# Initialize models
print("Loading models...")
generator = LLMGenerator(config.generator_model, config.device)
discriminator = LLMDiscriminator(config.discriminator_model, config.device)
sandbox = Sandbox(timeout=5)

# Initialize trainer
trainer = AdversarialTrainer(
    generator=generator,
    discriminator=discriminator,
    sandbox=sandbox,
    config=config
)

# Save initial baseline code
print("\n" + "="*80)
print("BASELINE: Generating code before training")
print("="*80)

baseline_dir = run_dir / "baseline"
baseline_dir.mkdir(exist_ok=True)

for i, problem in enumerate(problems):
    print(f"\nProblem {i+1}: {problem.id}")
    
    # Generate baseline code
    from reasoning.stages import get_stage
    stage_5 = get_stage(5)
    
    code = generator.generate_code(
        problem=problem.description,
        reasoning_chain=[],
        prompt_template=stage_5.generator_prompt_template,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        function_signature=problem.function_signature
    )
    
    # Save to file
    code_file = baseline_dir / f"{problem.id}.py"
    with open(code_file, 'w') as f:
        f.write(f"# Problem: {problem.id}\n")
        f.write(f"# Description: {problem.description[:100]}...\n\n")
        f.write(code)
    
    print(f"  Saved to: {code_file}")
    print(f"  Code preview: {code[:100]}...")

# Run training
print("\n" + "="*80)
print("TRAINING: Running 1 iteration")
print("="*80)

metrics = trainer.train_full_pipeline(
    problems=problems
)

print("\nTraining complete!")
print(f"Metrics: {metrics}")

# Save trained code
print("\n" + "="*80)
print("AFTER TRAINING: Generating code after training")
print("="*80)

trained_dir = run_dir / "trained"
trained_dir.mkdir(exist_ok=True)

for i, problem in enumerate(problems):
    print(f"\nProblem {i+1}: {problem.id}")
    
    # Generate trained code
    code = generator.generate_code(
        problem=problem.description,
        reasoning_chain=[],
        prompt_template=stage_5.generator_prompt_template,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        function_signature=problem.function_signature
    )
    
    # Save to file
    code_file = trained_dir / f"{problem.id}.py"
    with open(code_file, 'w') as f:
        f.write(f"# Problem: {problem.id}\n")
        f.write(f"# Description: {problem.description[:100]}...\n\n")
        f.write(code)
    
    print(f"  Saved to: {code_file}")
    print(f"  Code preview: {code[:100]}...")

# Save summary
summary = {
    "config": {
        "model": config.generator_model,
        "problems": [p.id for p in problems],
        "n_discriminator_steps": config.n_discriminator_steps,
        "n_generator_steps": config.n_generator_steps,
        "k_alternating_steps": config.k_alternating_steps
    },
    "metrics": metrics,
    "output_dir": str(run_dir)
}

summary_file = run_dir / "summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print(f"\nAll outputs saved to: {run_dir}")
print(f"\nCheck these folders:")
print(f"  - {baseline_dir}  (code before training)")
print(f"  - {trained_dir}   (code after training)")
print(f"  - {run_dir / 'checkpoints'}  (model checkpoints)")
print(f"\nSummary: {summary_file}")
