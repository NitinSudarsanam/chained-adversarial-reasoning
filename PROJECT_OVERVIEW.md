# Chained Adversarial Reasoning - Complete Project Overview

## Table of Contents
1. [Project Goal](#project-goal)
2. [Architecture Overview](#architecture-overview)
3. [Directory Structure](#directory-structure)
4. [Core Concepts](#core-concepts)
5. [File-by-File Breakdown](#file-by-file-breakdown)
6. [Training Flow](#training-flow)
7. [How to Use](#how-to-use)

---

## Project Goal

Train language models to solve coding problems through **adversarial multi-stage reasoning**:

1. **Generator** - Solves problems through 5 reasoning stages
2. **Discriminator** - Generates adversarial tests to find bugs
3. **Adversarial Training** - They compete: generator tries to pass tests, discriminator tries to break the code

The result: Models that produce more robust, well-reasoned code.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                             │
│                                                              │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │  Generator   │────────▶│ Discriminator│                 │
│  │              │  code   │              │                 │
│  │ (Solve       │         │ (Generate    │                 │
│  │  problems)   │◀────────│  tests)      │                 │
│  └──────────────┘  tests  └──────────────┘                 │
│         │                         │                         │
│         │                         │                         │
│         ▼                         ▼                         │
│  ┌──────────────────────────────────────┐                  │
│  │         Sandbox (Execute)            │                  │
│  │  - Run tests against code            │                  │
│  │  - Validate tests against ground     │                  │
│  │    truth                             │                  │
│  └──────────────────────────────────────┘                  │
│         │                         │                         │
│         ▼                         ▼                         │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │ Gen Reward   │         │ Disc Reward  │                 │
│  │ (pass rate)  │         │ (adversarial)│                 │
│  └──────────────┘         └──────────────┘                 │
│         │                         │                         │
│         └─────────┬───────────────┘                         │
│                   ▼                                         │
│            ┌─────────────┐                                  │
│            │ RL Training │                                  │
│            │    (PPO)    │                                  │
│            └─────────────┘                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
chained-adversarial-reasoning/
│
├── data/                          # Problem datasets
│   ├── problem_dataset.py         # Problem loading/validation
│   ├── function_problems.json     # Main problem set
│   └── example_problems.json      # Simple examples
│
├── models/                        # Model implementations
│   ├── generator.py               # Generator LLM wrapper
│   ├── discriminator.py           # Discriminator LLM wrapper
│   └── unified_model.py           # Single model for both (NEW)
│
├── reasoning/                     # Multi-stage reasoning
│   └── stages.py                  # 5 reasoning stages definition
│
├── sandbox/                       # Safe code execution
│   ├── sandbox.py                 # Main sandbox (pytest-based)
│   └── sandbox_simple.py          # Simple sandbox (raw asserts)
│
├── training/                      # Training logic
│   ├── adversarial_trainer.py     # Dual-model trainer
│   ├── unified_trainer.py         # Single-model trainer (NEW)
│   ├── rl_loop.py                 # PPO training step
│   ├── reward.py                  # Reward computation
│   ├── config.py                  # Training configuration
│   └── checkpoint_manager.py      # Save/load checkpoints
│
├── inference/                     # Inference engine
│   └── inference_engine.py        # Run trained models
│
├── evaluation/                    # Evaluation metrics
│   └── metrics.py                 # Compute performance metrics
│
├── run_training.py                # Main training script (dual model)
├── run_unified_training.py        # Main training script (unified)
├── run_inference.py               # Run inference on problems
├── show_model_output.py           # Debug: see model output
└── requirements.txt               # Python dependencies
```

---

## Core Concepts

### 1. Multi-Stage Reasoning

The generator solves problems through 5 stages:

1. **Informal Reasoning** - High-level intuition
2. **Structured Reasoning** - Organized breakdown
3. **Pseudocode** - Algorithm in pseudocode
4. **Constraints & Invariants** - Formal specifications
5. **Executable Code** - Final Python implementation

Each stage builds on previous stages, creating a chain of reasoning.

### 2. Adversarial Training

- **Generator Goal**: Pass all tests (maximize pass rate)
- **Discriminator Goal**: Generate tests that fail the generator (maximize failure rate)
- **Competition**: They push each other to improve

### 3. Reinforcement Learning (PPO)

Uses Proximal Policy Optimization:
- **Policy**: Model's generation strategy
- **Reward**: Test pass rate (generator) or adversarial effectiveness (discriminator)
- **Training**: Adjust model to maximize reward

### 4. Two Approaches

**Dual Model (Original)**:
- Separate generator and discriminator models
- More memory (2 models)
- More complex training

**Unified Model (New)**:
- Single model does both tasks via different prompts
- 50% less memory
- Simpler training
- **Recommended for limited GPU**

---

## File-by-File Breakdown

### Entry Points

#### `run_training.py`
**Purpose**: Main training script for dual-model approach

**What it does**:
1. Loads two separate models (generator + discriminator)
2. Loads problems from JSON
3. Creates `AdversarialTrainer`
4. Trains through all 5 stages sequentially
5. Saves checkpoints

**Usage**:
```bash
python run_training.py \
  --generator-model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --discriminator-model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --device cuda \
  --n-discriminator-steps 10 \
  --n-generator-steps 10
```

**Key Parameters**:
- `--generator-model`: HuggingFace model for generator
- `--discriminator-model`: HuggingFace model for discriminator
- `--device`: cuda or cpu
- `--n-discriminator-steps`: Training steps for discriminator per stage
- `--n-generator-steps`: Training steps for generator per stage
- `--k-alternating-steps`: Alternating training steps

---

#### `run_unified_training.py`
**Purpose**: Main training script for unified-model approach (RECOMMENDED)

**What it does**:
1. Loads ONE model that does both tasks
2. Loads problems from JSON
3. Creates `UnifiedTrainer`
4. Alternates between generator and discriminator training
5. Uses 50% less memory than dual model

**Usage**:
```bash
python run_unified_training.py \
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --device cuda \
  --n-steps 10
```

**Why use this**:
- ✅ 50% less GPU memory
- ✅ Simpler code
- ✅ Faster training
- ✅ Still adversarial

---

#### `run_inference.py`
**Purpose**: Run trained models on new problems

**What it does**:
1. Loads trained model(s)
2. Generates solution for a problem
3. Optionally executes tests
4. Shows results

**Usage**:
```bash
# Single problem
python run_inference.py \
  --problem "Write a function to reverse a string" \
  --signature "def reverse_string(s: str) -> str:"

# Batch mode
python run_inference.py \
  --batch \
  --problems-file data/function_problems.json \
  --output results.json
```

---

#### `show_model_output.py`
**Purpose**: Debug script to see raw model output

**What it does**:
1. Loads a model
2. Shows the exact prompt sent to model
3. Shows the raw generated code
4. Tests it against baseline tests
5. Shows pass/fail results

**Usage**:
```bash
python show_model_output.py
```

**When to use**:
- Model generating broken code
- Want to see exact prompts
- Debugging zero rewards
- Testing prompt changes

---

### Models

#### `models/generator.py`
**Purpose**: Wrapper for generator LLM

**Key Methods**:
- `generate_stage_output()` - Generate reasoning for stages 1-4
- `generate_code()` - Generate final code (stage 5)
- `get_log_probs()` - Get log probabilities for RL training
- `train()` / `eval()` - Set training mode
- `parameters()` - Get model parameters for optimizer

**What it wraps**:
- HuggingFace `AutoModelForCausalLM`
- Adds generation methods
- Adds RL training support
- Handles code extraction and cleaning

---

#### `models/discriminator.py`
**Purpose**: Wrapper for discriminator LLM

**Key Methods**:
- `generate_tests()` - Generate adversarial test cases
- `generate_critique()` - Generate critique of reasoning
- `get_log_probs()` - Get log probabilities for RL training
- `train()` / `eval()` - Set training mode
- `parameters()` - Get model parameters for optimizer

**What it wraps**:
- HuggingFace `AutoModelForCausalLM`
- Adds test generation methods
- Adds RL training support
- Handles test code sanitization

---

#### `models/unified_model.py`
**Purpose**: Single model that does both generator and discriminator tasks

**Key Methods**:
- **Generator methods**: `generate_stage_output()`, `generate_code()`
- **Discriminator methods**: `generate_tests()`, `generate_critique()`
- `get_log_probs()` - For RL training
- `train()` / `eval()` - Set training mode

**How it works**:
- Same model, different prompts
- "Solve this problem" → generator mode
- "Generate tests for this code" → discriminator mode
- Single optimizer, single memory footprint

---

### Reasoning

#### `reasoning/stages.py`
**Purpose**: Defines the 5 reasoning stages

**What it contains**:
```python
REASONING_STAGES = [
    ReasoningStage(
        id=1,
        name="Informal Reasoning",
        generator_prompt_template="...",
        discriminator_prompt_template="..."
    ),
    # ... stages 2-5
]
```

**Each stage has**:
- `id`: 1-5
- `name`: Human-readable name
- `description`: What this stage does
- `generator_prompt_template`: Prompt for generator
- `discriminator_prompt_template`: Prompt for discriminator

**Key Function**:
- `get_stage(stage_id)` - Get stage by ID

---

### Sandbox

#### `sandbox/sandbox.py`
**Purpose**: Safe execution environment for code and tests

**Key Methods**:
- `execute_tests(code, tests)` - Run tests against code
- `validate_tests_against_solution(tests, solution)` - Check if tests are valid

**How it works**:
1. Creates temporary directory
2. Writes code and tests to files
3. Runs pytest in subprocess with timeout
4. Parses pytest output
5. Returns `ExecutionResult` with pass/fail counts

**Safety features**:
- Timeout (default 5 seconds)
- Subprocess isolation
- Temporary files (auto-deleted)
- Error handling

**Test format support**:
- Raw assert statements → wraps in test functions
- Pytest functions → runs directly

---

#### `sandbox/sandbox_simple.py`
**Purpose**: Simpler sandbox for raw assert statements

**When to use**:
- Baseline tests (hand-written asserts)
- Quick testing
- No pytest needed

**Difference from main sandbox**:
- Doesn't use pytest
- Runs raw Python code
- Simpler parsing
- Used by `show_model_output.py`

---

### Training

#### `training/adversarial_trainer.py`
**Purpose**: Orchestrates dual-model adversarial training

**Key Methods**:
- `train_discriminator_epoch()` - Train discriminator with frozen generator
- `train_generator_epoch()` - Train generator with frozen discriminator
- `train_alternating()` - Alternate between both
- `train_stage()` - Train one reasoning stage (N+N+K pattern)
- `train_full_pipeline()` - Train all 5 stages sequentially

**Training pattern (N+N+K)**:
1. N steps: Train discriminator (generator frozen)
2. N steps: Train generator (discriminator frozen)
3. K steps: Alternate between both

**What it manages**:
- Two models (generator + discriminator)
- Two optimizers
- Freezing/unfreezing models
- Checkpoint saving
- Metrics tracking

---

#### `training/unified_trainer.py`
**Purpose**: Simplified trainer for unified model

**Key Methods**:
- `train_stage()` - Train one stage (alternates gen/disc)
- `_train_discriminator_step()` - Single discriminator step
- `_train_generator_step()` - Single generator step
- `_generate_full_chain()` - Generate reasoning + tests
- `train_full_pipeline()` - Train all 5 stages

**Differences from adversarial_trainer**:
- ✅ One model instead of two
- ✅ One optimizer instead of two
- ✅ No freezing/unfreezing needed
- ✅ Simpler code
- ✅ 50% less memory

**Training pattern**:
- Alternates: discriminator step, generator step, discriminator step, ...

---

#### `training/rl_loop.py`
**Purpose**: Core RL training logic (PPO)

**Key Functions**:
- `train_step()` - Execute one RL training step
- `compute_policy_loss()` - Compute PPO loss
- `create_optimizer()` - Create AdamW optimizer
- `freeze_model()` / `unfreeze_model()` - Freeze/unfreeze parameters

**What `train_step()` does**:
1. Get current log probabilities from model
2. Compare with old log probabilities
3. Compute PPO clipped loss
4. Backpropagate
5. Clip gradients
6. Update weights
7. Return metrics

**PPO formula**:
```
ratio = exp(log_prob_new - log_prob_old)
clipped_ratio = clip(ratio, 1-ε, 1+ε)
loss = -min(ratio * advantage, clipped_ratio * advantage)
```

---

#### `training/reward.py`
**Purpose**: Compute rewards for RL training

**Key Functions**:

**`compute_generator_reward(execution_result)`**:
```python
# Generator wants to pass tests
if num_total == 0:
    return 0.0  # No tests = no reward
reward = num_passed / num_total  # Pass rate
return reward
```

**`compute_discriminator_reward(gen_result, val_result)`**:
```python
# Discriminator wants generator to fail, but tests must be valid
adversarial_score = 1.0 - generator_pass_rate
test_validity = validation_pass_rate
reward = adversarial_score * test_validity
return reward
```

**Why this works**:
- Generator: Maximize pass rate → better code
- Discriminator: Maximize failure rate → harder tests
- But: Tests must be valid (pass ground truth)
- Result: Adversarial competition with quality control

---

#### `training/config.py`
**Purpose**: Training configuration dataclass

**Key Parameters**:
```python
@dataclass
class TrainingConfig:
    n_discriminator_steps: int = 10
    n_generator_steps: int = 10
    k_alternating_steps: int = 5
    learning_rate: float = 1e-5
    num_tests_per_problem: int = 5
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    clip_epsilon: float = 0.2
```

---

#### `training/checkpoint_manager.py`
**Purpose**: Save and load model checkpoints

**Key Methods**:
- `save_checkpoint()` - Save models + metadata
- `load_checkpoint()` - Load models from checkpoint
- `get_best_checkpoint()` - Get path to best checkpoint
- `get_latest_checkpoint()` - Get path to latest checkpoint
- `should_save_as_best()` - Check if current is best

**What it saves**:
- Model state dicts
- Training metrics
- Stage/epoch info
- Timestamp
- Config

**Checkpoint format**:
```
checkpoints/
├── checkpoint_stage_1_epoch_10.pt
├── checkpoint_stage_1_epoch_10.json  (metadata)
├── checkpoint_stage_2_epoch_10.pt
└── checkpoint_best.pt  (best model)
```

---

### Data

#### `data/problem_dataset.py`
**Purpose**: Load and validate coding problems

**Key Classes**:
```python
@dataclass
class Problem:
    id: str
    description: str
    function_signature: str
    baseline_tests: List[str]
    reference_solution: str
    difficulty: str
    tags: List[str]
```

**Key Functions**:
- `load_problems(filepath)` - Load from JSON
- `validate_problem(problem)` - Check if valid

**Validation checks**:
- All fields non-empty
- Reference solution is valid Python
- Baseline tests are valid Python
- Function signature is present

---

#### `data/function_problems.json`
**Purpose**: Main problem dataset

**Format**:
```json
{
  "problems": [
    {
      "id": "string_compression",
      "description": "Implement string compression...",
      "function_signature": "def compress_string(s: str) -> str:",
      "baseline_tests": [
        "assert compress_string('aaabbc') == 'a3bbc'",
        "assert compress_string('abc') == 'abc'"
      ],
      "reference_solution": "def compress_string(s: str) -> str:\n    ...",
      "difficulty": "easy",
      "tags": ["string", "compression"]
    }
  ]
}
```

**What it contains**:
- Problem descriptions
- Function signatures
- Hand-written baseline tests
- Ground truth solutions
- Metadata (difficulty, tags)

---

### Inference

#### `inference/inference_engine.py`
**Purpose**: Run trained models on new problems

**Key Class**: `InferenceEngine`

**Key Methods**:
- `solve_problem()` - Generate solution for a problem
- `solve_with_reasoning()` - Generate with full reasoning chain
- `batch_solve()` - Solve multiple problems

**What it does**:
1. Takes problem description
2. Generates reasoning through 5 stages
3. Generates final code
4. Optionally executes tests
5. Returns solution + metrics

---

### Evaluation

#### `evaluation/metrics.py`
**Purpose**: Compute evaluation metrics

**Key Functions**:
- `compute_pass_at_k()` - Pass@k metric
- `compute_test_coverage()` - Test coverage
- `compute_code_quality()` - Code quality metrics

---

## Training Flow

### Dual Model Training

```
1. Load generator and discriminator models
2. For each stage (1-5):
   a. Train discriminator (N steps):
      - Generate full reasoning chain
      - Generate tests for this stage
      - Execute tests against code
      - Validate tests against ground truth
      - Compute discriminator reward
      - Update discriminator weights
   
   b. Train generator (N steps):
      - Generate full reasoning chain
      - Get tests from discriminator
      - Execute tests against code
      - Compute generator reward
      - Update generator weights
   
   c. Alternating training (K steps):
      - Alternate between discriminator and generator
   
   d. Save checkpoint
3. Training complete
```

### Unified Model Training

```
1. Load single unified model
2. For each stage (1-5):
   For N steps:
      If step is even:
         - Train as discriminator (generate tests)
      Else:
         - Train as generator (solve problem)
   Save checkpoint
3. Training complete
```

---

## How to Use

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. Quick Test

```bash
# See what model generates
python show_model_output.py
```

### 3. Training (Recommended: Unified Model)

```bash
# Small model (fits in 15GB GPU)
python run_unified_training.py \
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --device cuda \
  --n-steps 10 \
  --learning-rate 1e-5

# Larger model (if you have 40GB+ GPU)
python run_unified_training.py \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --device cuda \
  --n-steps 10
```

### 4. Training (Dual Model)

```bash
# Only if you have lots of memory
python run_training.py \
  --generator-model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --discriminator-model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --device cuda \
  --n-discriminator-steps 10 \
  --n-generator-steps 10
```

### 5. Inference

```bash
# Single problem
python run_inference.py \
  --problem "Write a function to check if a string is a palindrome" \
  --signature "def is_palindrome(s: str) -> bool:"

# Batch
python run_inference.py \
  --batch \
  --problems-file data/function_problems.json \
  --output results.json
```

### 6. Debug

```bash
# Test sandbox
python debug_test_execution.py

# Test model generation
python test_model_generation.py

# Check memory
python configure_memory.py
```

---

## Common Issues

### Zero Rewards
**Symptom**: `Generator Reward: 0.0000, Discriminator Reward: 0.0000`

**Causes**:
1. Model too small (0.5B struggles)
2. Model generating invalid code
3. Model generating invalid tests

**Solutions**:
- Run `python test_model_generation.py` to see what's generated
- Use larger model (1.5B or 3B)
- Increase `max_new_tokens` to 512
- Increase `temperature` to 0.9

### Out of Memory
**Symptom**: `CUDA out of memory`

**Solutions**:
- Use unified model (50% less memory)
- Use smaller model (0.5B instead of 7B)
- Reduce `max_new_tokens`
- Use CPU (slow but works)
- Restart runtime to clear memory

### Model Not Following Instructions
**Symptom**: Generated code has wrong function name, no implementation

**Solutions**:
- Use larger model
- Improve prompts (already done)
- Increase temperature for more creativity
- Check `show_model_output.py` to see exact output

---

## Key Takeaways

1. **Use unified model** for limited GPU memory
2. **0.5B model is small** - expect lower quality, but it fits in memory
3. **1.5B or 3B is better** - if you have the memory
4. **Zero rewards = broken generation** - debug with test scripts
5. **Training is slow** - each step takes 1-2 minutes
6. **Adversarial training works** - models improve through competition

---

## Next Steps

1. **Run `show_model_output.py`** - See what model generates
2. **Run `test_model_generation.py`** - Test both generator and discriminator
3. **Run `debug_test_execution.py`** - Verify sandbox works
4. **Start training** - Use unified model with 0.5B
5. **Monitor rewards** - Should be 0.3-0.7, not 0.0
6. **Upgrade model** - If rewards are good, try 1.5B or 3B

Good luck! 🚀
