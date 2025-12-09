# Chained Adversarial Reasoning for Code Generation

## What Is This?

Current language models can write code, but they often:
- Jump straight to implementation without showing their reasoning
- Miss edge cases and produce brittle solutions
- Struggle with complex problems requiring careful thought
- Generate code that looks right but fails on corner cases

This project addresses these issues through **adversarial multi-stage reasoning** - a reinforcement learning system where two AI models compete to make each other better at coding.

## Core Idea

### The Multi-Stage Reasoning Approach

Instead of asking an LLM to immediately write code, we force it through **5 explicit reasoning stages**:

1. **Informal Reasoning** - "What is this problem really asking? What's my intuition?"
2. **Structured Reasoning** - "Let me break this down systematically with clear steps"
3. **Pseudocode** - "Here's the algorithm in plain language"
4. **Constraints & Invariants** - "What assumptions must hold? What are the edge cases?"
5. **Executable Code** - "Now I'll implement it properly in Python"

This mirrors how expert programmers actually solve problems - they don't jump straight to code. By forcing the model to articulate its reasoning at each stage, we make the problem-solving process explicit, interpretable, and more robust.

### The Adversarial Training Loop

Here's where it gets interesting. We train **two models that compete**:

**Generator (The Problem Solver)**
- Produces solutions through the 5 reasoning stages
- Tries to write code that passes all test cases
- Gets rewarded for correct solutions

**Discriminator (The Bug Hunter)**
- Reads the generator's reasoning at each stage
- Generates adversarial test cases designed to break the code
- Gets rewarded for finding bugs the baseline tests miss

They improve through competition:
- Generator learns to handle edge cases (because discriminator keeps finding them)
- Discriminator learns to create better tests (because generator keeps passing simple ones)
- Both co-evolve, pushing each other to be more sophisticated

### Main Idea

**Better than just training on correct solutions** because:
- The discriminator actively searches for weaknesses, not just random testing
- The generator must defend against adversarial tests, making it more robust
- We get interpretable reasoning chains showing the model's thought process
- The system improves on dimensions beyond "getting the right answer"

**Better than single-stage code generation** because:
- Explicit reasoning stages prevent jumping to conclusions
- Each stage can be inspected for errors
- The model learns problem-solving methodology, not just code patterns
- Edge cases and invariants are considered before writing code

## Project Goal

Create LLMs that produce more robust, well-reasoned code by:
1. **Structured reasoning** - Breaking down problem-solving into explicit stages
2. **Adversarial pressure** - Training against a discriminator that tries to break the code
3. **Execution feedback** - Using PPO reinforcement learning with real test execution results

The result: Models that don't just memorize solutions, but actually learn to reason about problems systematically and defensively.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING LOOP                             │
│                                                                  │
│  ┌──────────────┐                    ┌──────────────┐          │
│  │  Generator   │──── solution ─────▶│ Discriminator│          │
│  │              │     code           │              │          │
│  │ (5 reasoning │                    │ (Generate    │          │
│  │  stages)     │◀──── tests ────────│  adversarial │          │
│  └──────────────┘                    │  tests)      │          │
│         │                             └──────────────┘          │
│         │                                    │                  │
│         └──────────┬───────────────────────┘                  │
│                    ▼                                            │
│         ┌─────────────────────────┐                            │
│         │  Sandbox (Execute)      │                            │
│         │  - Run tests on code    │                            │
│         │  - Validate tests       │                            │
│         └─────────────────────────┘                            │
│                    │                                            │
│         ┌──────────┴───────────┐                               │
│         ▼                      ▼                                │
│  ┌─────────────┐        ┌─────────────┐                       │
│  │ Gen Reward  │        │ Disc Reward │                       │
│  │ pass_rate-0.5│       │ validity +  │                       │
│  │             │        │ bug_catching│                       │
│  └─────────────┘        └─────────────┘                       │
│         │                      │                                │
│         └──────────┬───────────┘                               │
│                    ▼                                            │
│             ┌─────────────┐                                     │
│             │ PPO Update  │                                     │
│             │ (LoRA only) │                                     │
│             └─────────────┘                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Generator (5 Reasoning Stages)**:
   - Stage 1: Informal Reasoning - High-level intuition
   - Stage 2: Structured Reasoning - Organized breakdown
   - Stage 3: Pseudocode - Algorithm design
   - Stage 4: Constraints & Invariants - Correctness conditions
   - Stage 5: Executable Code - Final Python implementation

2. **Discriminator**: Generates adversarial test cases at each stage to expose bugs

3. **Sandbox**: Safely executes code and tests with timeout protection

4. **Reward System**:
   - **Generator**: `(pass_rate - 0.5)` - Encourages passing tests while ensuring non-zero gradients
   - **Discriminator**: `(valid_pct × 0.4) - (invalid_pct × 0.6) + (bug_catch_rate × 0.6)` - Rewards valid tests that catch bugs

5. **PPO Training**: Proximal Policy Optimization with clipped surrogate objective

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/NitinSudarsanam/chained-adversarial-reasoning.git
cd chained-adversarial-reasoning

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- 16GB+ RAM

Key dependencies:
- `transformers` - HuggingFace models
- `peft` - LoRA efficient fine-tuning
- `bitsandbytes` - 4-bit quantization
- `pytest` - Test execution

### Basic Training

```bash
# Train with default settings (Qwen 1.5B model)
python run_training.py

# Train with custom model
python run_training.py --generator-model "Qwen/Qwen2.5-Coder-7B-Instruct"

# Minimal training (1 step per stage, for testing)
python run_minimal_training.py
```

### Configuration

Edit `training/config.py` to customize:

```python
@dataclass
class TrainingConfig:
    # Models
    generator_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    discriminator_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    device: str = "cuda"  # or "cpu"
    
    # Training
    n_discriminator_steps: int = 10
    n_generator_steps: int = 10
    learning_rate: float = 1e-5
    clip_epsilon: float = 0.2
    
    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.7
```

## Project Structure

```
chained-adversarial-reasoning/
│
├── data/                          # Problem datasets
│   ├── problem_dataset.py         # Problem loading/validation
│   ├── function_problems.json     # Main problem set
│   ├── leetcode_formatted.json    # LeetCode problems
│   └── example_problems.json      # Simple examples
│
├── models/                        # Model implementations
│   ├── generator.py               # Generator LLM (5-stage reasoning)
│   ├── discriminator.py           # Discriminator LLM (test generation)
│   └── unified_model.py           # Single model for both tasks
│
├── reasoning/                     # Multi-stage reasoning
│   └── stages.py                  # 5 reasoning stage definitions
│
├── sandbox/                       # Safe code execution
│   ├── sandbox.py                 # Main sandbox (pytest-based)
│   └── sandbox_simple.py          # Simple sandbox (raw asserts)
│
├── training/                      # Training logic
│   ├── adversarial_trainer.py     # Dual-model adversarial trainer
│   ├── unified_trainer.py         # Single-model trainer
│   ├── rl_loop.py                 # PPO training step
│   ├── reward.py                  # Reward computation
│   ├── config.py                  # Training configuration
│   ├── checkpoint_manager.py      # Save/load checkpoints
│   └── training_logger.py         # Training metrics logging
│
├── inference/                     # Inference engine
│   └── inference_engine.py        # Run trained models
│
├── evaluation/                    # Evaluation metrics
│   └── metrics.py                 # Performance metrics
│
├── execution/                     # Code execution
│   └── direct_executor.py         # Direct Python execution
│
├── run_training.py                # Main training script
├── run_minimal_training.py        # Minimal test training
├── run_inference.py               # Inference script
├── show_model_output.py           # Debug model outputs
└── requirements.txt               # Python dependencies
```

## Technical Deep Dive

### Multi-Stage Reasoning

The generator produces solutions through 5 explicit stages, each building on the previous:

**Stage 1: Informal Reasoning** - High-level intuition and initial thoughts about the problem

**Stage 2: Structured Reasoning** - Organized breakdown with key observations and approach steps

**Stage 3: Pseudocode** - Algorithm design in plain language with clear logic flow

**Stage 4: Constraints & Invariants** - Explicit correctness conditions, edge cases, and complexity analysis

**Stage 5: Executable Code** - Final Python implementation incorporating all previous reasoning

This staged approach ensures the model considers the problem holistically before coding, reducing bugs from hasty implementation.

### Adversarial Test Generation

The discriminator reads the generator's reasoning at each stage and generates test cases designed to expose weaknesses:

- **Early stages** (1-2): Basic tests for happy paths and simple edge cases
- **Later stages** (3-4): Sophisticated tests targeting algorithmic edge cases, boundary conditions, and corner cases identified in the reasoning

**Test Format**: Each test is a tuple of `(inputs, expected_output)` where inputs match the function signature

The discriminator is rewarded for generating valid tests that catch bugs, creating evolutionary pressure toward better test coverage.

### Reward System

**Generator Reward**:
```python
raw_reward = pass_rate  # [0, 1]
final_reward = raw_reward - 0.5  # [-0.5, +0.5]
```
- Shifted to ensure non-zero loss even at 0% pass rate
- Negative reward for poor performance, positive for good

**Discriminator Reward**:
```python
reward = (valid_pct × 0.4) - (invalid_pct × 0.6) + (bug_catch_rate × 0.6)
reward = clamp(reward, -1, 1)
### Reward System

**Generator Reward**: Based on test pass rate, shifted to ensure learning signal even at 0% success
- Higher reward for passing more tests
- Negative reward when failing tests drives improvement
- Must handle both baseline tests and adversarial tests from discriminator
### LoRA Fine-Tuning

Uses **LoRA (Low-Rank Adaptation)** for efficient training by only training small adapter layers instead of the full model.

**Benefits**:
- Trains only ~0.5% of model parameters (vs 100% for full fine-tuning)
- Checkpoint sizes: ~10-50MB instead of 2-7GB
- Much faster training with less memory usage
- Base model stays frozen and 4-bit quantized

**How it works**: Small trainable matrices are injected into the model's attention layers. During forward pass, the frozen base model computation is combined with the LoRA adapters. Only the tiny LoRA weights get updated during training.
ratio = torch.exp(new_log_probs - old_log_probs)

# Clipped surrogate objective
clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
surrogate1 = ratio * advantage
surrogate2 = clipped_ratio * advantage
loss = -torch.min(surrogate1, surrogate2).mean()

# Update model
optimizer.zero_grad()
loss.backward()
optimizer.step()
### PPO Training

Uses **Proximal Policy Optimization** (PPO), the same algorithm behind ChatGPT's training.

**Key aspects**:
- Prevents large, destabilizing policy updates through clipping
- Learns from actual test execution results (not just labels)
- Separate optimizers for generator and discriminator
- Alternating updates: train discriminator → train generator → repeat

PPO is stable, sample-efficient, and proven effective for training language models with reinforcement learning from execution feedback.
```

### Inference

```python
from training.checkpoint_manager import CheckpointManager
from models.generator import LLMGenerator
from models.discriminator import LLMDiscriminator

# Initialize models
generator = LLMGenerator("Qwen/Qwen2.5-Coder-1.5B-Instruct", device="cuda")
discriminator = LLMDiscriminator("Qwen/Qwen2.5-Coder-1.5B-Instruct", device="cuda")

# Load trained checkpoint
checkpoint_manager = CheckpointManager("checkpoints")
best_checkpoint = checkpoint_manager.get_best_checkpoint()
checkpoint_manager.load_checkpoint(best_checkpoint, generator, discriminator)

# Generate solution
from reasoning.stages import get_stage
from data.problem_dataset import load_problems

problems = load_problems("data/function_problems.json")
problem = problems[0]

# Generate all 5 stages
reasoning_chain = []
for stage_id in range(1, 6):
    stage = get_stage(stage_id)
    output = generator.generate_stage_output(
        problem=problem.description,
        previous_stages=reasoning_chain,
        stage_id=stage_id,
        prompt_template=stage.generator_prompt_template,
### Inference

```bash
# Run inference with trained model
python run_inference.py
```

This will:
1. Load the best trained checkpoint
2. Generate solutions for problems through all 5 reasoning stages
3. Show the complete reasoning chain
4. Execute and validate the final codent(f"Latest checkpoint: {latest}")

# Load specific checkpoint
metadata = checkpoint_manager.load_checkpoint(
    checkpoint_path="checkpoints/checkpoint_stage_5_epoch_10.pt",
    generator=generator,
    discriminator=discriminator
)
print(f"Loaded checkpoint from stage {metadata['stage']}, epoch {metadata['epoch']}")
```

## Monitoring Training

### Debugging Model Outputs

```bash
# Show what models generate before training
python show_model_output.py
```

This displays the generator's reasoning chain, discriminator's test cases, execution results, and computed rewards.

### Checkpoints

Checkpoints are automatically saved during training to `checkpoints/`:
- `checkpoint_stage_X_epoch_Y.pt` - Regular checkpoints
- `checkpoint_best.pt` - Best performing model

Use `CheckpointManager` to load checkpoints for inference or resume training. - Valid disc tests: 2/4 (50.0%)
    - Combined with baseline: 3/6 (50.0%)
    - Baseline only: 1/2 (50.0%)
  Reward: 0.00
  Loss: 0.456

✓ Saved checkpoint: checkpoints/checkpoint_stage_1_epoch_1.pt
```

### Metrics Explained

- **Generator**:
  - `loss`: PPO loss (lower is better during updates)
  - `reward`: Pass rate - 0.5 (higher is better)
  - `test_pass_rate`: Percentage of tests passed

- **Discriminator**:
  - `loss`: PPO loss
  - `reward`: Validity + bug catching (higher is better)
  - `test_validity`: Percentage of syntactically valid tests
  - `bug_catch_rate`: Percentage of tests that fail on generator's code

## Troubleshooting

### Out of Memory (OOM)

```bash
# Use smaller model
python run_training.py --generator-model "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Reduce batch size (edit config.py)
n_discriminator_steps = 5
n_generator_steps = 5

# Clear GPU memory between runs
python clear_gpu_memory.py
```

### Empty Model Outputs

The system includes extensive logging for empty outputs:

```
⚠ WARNING: generate_stage_output produced EMPTY output
   Stage ID: 5
   Prompt preview: You are solving a coding problem...
```

**Common causes**:
- Model temperature too low → increase to 0.7-1.0
- Max tokens too small → increase to 512+
- Model not loaded correctly → check device placement

### Memory Usage

1. **Clear cache regularly**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

2. **Use gradient checkpointing** (for larger models):
   ```python
   model.gradient_checkpointing_enable()
   ```

3. **Reduce concurrent problems**:
   ```python
   config.n_discriminator_steps = 5  # vs 10
   ```

## Key Concepts

### Why Multi-Stage Reasoning?

Breaking problem-solving into stages:
1. **Forces explicit thinking** - Can't skip to code
2. **Improves debugging** - See where reasoning fails
3. **Better generalization** - Learn problem-solving process, not just patterns
4. **Interpretable** - Can inspect reasoning at each stage

### Why Adversarial Training?

1. **Robustness** - Generator must handle edge cases
2. **Better test coverage** - Discriminator finds creative bugs
3. **Co-evolution** - Both models improve together
4. **Beyond ground truth** - Discovers bugs even correct-looking code might have

### Why PPO?

1. **Stable** - Clipped updates prevent large policy changes
2. **Sample efficient** - Reuses old experiences with importance sampling
3. **Proven** - Successfully used in AlphaGo, ChatGPT, etc.
4. **On-policy** - Directly optimizes the policy we care about

## Contributing

Contributions welcome! Areas for improvement:

- [ ] Add more problem datasets (CodeContest, HumanEval, etc.)
- [ ] Implement curriculum learning (easy → hard problems)
- [ ] Add type checking for generated code
- [ ] Implement multi-file code generation
- [ ] Add support for other languages (Java, C++, etc.)
- [ ] Improve test salvaging from malformed discriminator output
- [ ] Add baseline comparisons (without reasoning stages)
- [ ] Implement value function for advantage estimation

## License

MIT License - see LICENSE file for details

## Acknowledgments

- **PEFT** library for LoRA implementation
- **HuggingFace** for model hosting and transformers library
- **Qwen** for the excellent Coder models
- **OpenAI** for PPO algorithm research

## Contact

For questions or issues, please open a GitHub issue or contact the maintainer.

---
