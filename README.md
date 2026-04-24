# Chained Adversarial Reasoning

A self-play RL system where two LLMs compete, one writing code and one writing tests, to make code generation more robust.

Final project for CSCI 2470 (Deep Learning) at Brown. Written end-to-end: reasoning pipeline, PPO loop, isolated sandbox, reward design, training harness, eval suite.

## Overview

Code-generation LLMs tend to jump straight to implementation and produce solutions that look right but break on corner cases. We wanted to try something simpler: can two small LLMs teach each other to write better code with nothing but a Python interpreter in the loop?

Turns out yes. The system has three pieces:

1. Generator: an LLM that writes a Python solution to a coding problem.
2. Discriminator: a separate LLM that reads the generator's output and writes adversarial test cases meant to break it.
3. Sandbox: a hardened subprocess executor that runs generated code and tests against a ground-truth reference and returns rewards to both models.

Both models are fine-tuned with PPO (Proximal Policy Optimization) using LoRA adapters on top of a frozen 4-bit quantized base model. Rewards come entirely from execution outcomes against a reference implementation. No human feedback, no labeled reasoning traces, no reward model. The whole learning signal is a pass/fail vector from a Python subprocess.

## Repository Layout

```
chained-adversarial-reasoning/
├── 1-stage-reasoning/      # Final system used for reported runs
│   ├── csci2470_1stageFinal.ipynb    # End-to-end training notebook
│   ├── generator.ipynb / discriminator.ipynb / sandbox.ipynb
│   ├── rl_loop.py          # PPO update step
│   ├── sandbox.py          # Subprocess-isolated code executor
│   ├── log1.jsonl          # Raw training telemetry (100 steps)
│   └── analysis.ipynb      # Post-hoc analysis of training logs
│
├── 5-stage-reasoning/      # Initial prototype with full reasoning pipeline
│   ├── reasoning/stages.py # 5-stage prompt templates
│   ├── models/             # Generator, discriminator, unified-model wrappers
│   ├── training/           # Adversarial trainer, PPO loop, reward, checkpoints
│   ├── sandbox/            # pytest-based and assert-based executors
│   ├── inference/          # Trained-model inference engine
│   ├── evaluation/         # Metric computation
│   ├── run_training.py
│   ├── run_inference.py
│   └── requirements.txt
│
├── data/                   # Problem datasets
│   ├── leetcode_formatted.json       # LeetCode problem bank (primary)
│   ├── function_problems.json        # Curated function-signature problems
│   ├── custom_problems*.json         # Hand-written problems
│   └── problem_dataset.py            # Loader + validation
│
└── evals/                  # Evaluation scripts and results
    ├── eval_generator_{trained,untrained}.py
    ├── eval_discriminator_{trained,untrained}.py
    ├── colab.py            # Shared model / PPO harness for Colab runs
    └── results/            # Per-problem pass counts (CSV)
```

## The Two Systems

### 5-Stage Reasoning (initial design)

The generator produces a solution through five prompts, each conditioned on the previous outputs:

1. Informal reasoning: intuitive read of the problem
2. Structured reasoning: numbered breakdown, observations, edge cases
3. Pseudocode: language-agnostic algorithm
4. Constraints and invariants: pre/post-conditions, complexity, loop invariants
5. Executable code: final Python implementation

The discriminator runs at every stage and generates tests aimed at whatever the generator has committed to so far. Happy-path tests early, constraint-violating stress tests late.

This design lives in `5-stage-reasoning/` and is fully implemented, but training both models through five stages per problem was more compute than we had.

### 1-Stage Reasoning (final system)

We cut the pipeline to a single forward pass. The generator produces code directly, and the discriminator writes a batch of adversarial test cases from the problem statement alone. Trimming the pipeline made training fit inside a Colab budget while keeping the adversarial dynamic. This is the configuration behind all reported training runs and evaluation results. Supporting code is in `1-stage-reasoning/`.

## Reward Shaping

Rewards come from execution outcomes against a reference solution. For each generated test `(inputs, expected)`:

- If the reference solution produces `expected` on `inputs`, the test is valid. The discriminator gets credit and the test is then run against the generator's code.
- If the test is valid and the generator's code fails it, the discriminator gets an additional bug-catch reward and the generator is penalized.
- If the test is invalid (disagrees with the reference, malformed, or unparseable), the discriminator is penalized.

This keeps incentives honest both ways. The discriminator cannot farm rewards by emitting garbage or contradictory tests, and the generator is only penalized by tests that the ground truth actually passes. You also get a curriculum for free: as the generator improves, the only tests that still earn bug-catch reward are the genuinely adversarial ones. Reward constants are in `1-stage-reasoning/sandbox.py`.

## Training Details

| | |
|---|---|
| Base model (1-stage) | Llama 3.1 8B Instruct |
| Base model (5-stage default) | Qwen2.5-Coder-1.5B-Instruct |
| Fine-tuning | LoRA (r=16, α=32, dropout=0.05) on all attention and MLP projections |
| Quantization | 4-bit (bitsandbytes NF4) for the frozen base model |
| Algorithm | PPO with clipped surrogate objective (ε = 0.2) |
| Optimizer | AdamW, lr = 1e-5, weight decay = 0.01 |
| Gradient clipping | max-norm 1.0 |
| Sandbox | `multiprocessing.Process` with 5 s timeout, killed on overrun |
| Training set | LeetCode problems, filtered to `easy` difficulty, excluding linked-list and tree tags |

Two 8B LoRA models plus the sandbox don't fit on a single Colab T4. The training notebook supports splitting generator, discriminator, and sandbox across three Colab sessions, passing outputs and rewards between them by copy-paste. See `1-stage-reasoning/1stage.md`.

## Evaluation

Trained and untrained checkpoints of both models run on held-out LeetCode problems. Protocol:

- Generator: fraction of baseline tests passed per problem.
- Discriminator: (a) fraction of generated tests that agree with the reference solution, (b) fraction of valid tests that expose bugs in the generator's code.

Across 100 PPO steps, the trained discriminator hit a higher bug-catch rate on held-out problems than its untrained counterpart. That is the main signal that adversarial training taught one of the models something non-trivial about where code-generation models break. Per-problem results are in `evals/results/*.csv`. Loss and reward traces are in `1-stage-reasoning/log1.jsonl` and analyzed in `1-stage-reasoning/analysis.ipynb`.

## Reproducing Results

**1-Stage:**

Open `1-stage-reasoning/csci2470_1stageFinal.ipynb` in Colab with ~48 GB of VRAM available (A100). Import `sandbox.py` and `rl_loop.py`, run all cells.

If you only have multiple smaller GPUs or Colab instances, run `generator.ipynb`, `discriminator.ipynb`, and `sandbox.ipynb` in separate sessions and shuttle outputs between them.

**5-Stage (prototype):**

```bash
cd 5-stage-reasoning
pip install -r requirements.txt
python run_training.py --generator-model Qwen/Qwen2.5-Coder-1.5B-Instruct
python run_inference.py
```

Configuration is in `5-stage-reasoning/training/config.py`.

## Results and Takeaways

- PPO stayed stable under pure execution feedback. Two 8B LoRA models trained jointly against a sandbox, no reward model, no human feedback, clean monotonic loss curves across 100 steps. The hard part of RLHF-style training, minus the H and the F.
- The discriminator learned. After training it writes more parseable tests, more tests that agree with the reference solution, and catches more bugs per valid test than the untrained baseline. The signal transfers to held-out problems.
- The full 5-stage pipeline is implemented. Every stage (informal reasoning, structured reasoning, pseudocode, constraints and invariants, code) has its own prompt template, reward shaping, and trainer. Compute is what kept us from running it at scale, not missing code. Anyone with an A100 budget can run it.
- Honest limits. On our filtered held-out split, generator pass-rate improvements were small and noisy, which is what you would expect for a 1.5B to 8B policy with 100 PPO steps over dozens of problems. Scaling either the step count or the base model is the obvious next move.

The full discussion is in the final report and in `1-stage-reasoning/analysis.ipynb`.

## Key Files

| Purpose | File |
|---|---|
| PPO training step | `1-stage-reasoning/rl_loop.py` |
| Sandbox executor + reward computation | `1-stage-reasoning/sandbox.py` |
| End-to-end training run | `1-stage-reasoning/csci2470_1stageFinal.ipynb` |
| Reasoning stage definitions | `5-stage-reasoning/reasoning/stages.py` |
| Adversarial trainer (5-stage) | `5-stage-reasoning/training/adversarial_trainer.py` |
| Problem dataset loader | `data/problem_dataset.py` |
| Eval harness | `evals/colab.py` |

## Stack

PyTorch, Hugging Face Transformers, PEFT (LoRA), bitsandbytes, TRL-style PPO (reimplemented), Hugging Face datasets format, Google Colab.

## License

MIT.
