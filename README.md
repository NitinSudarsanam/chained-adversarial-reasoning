# Chained Adversarial Reasoning

**A self-play reinforcement learning system in which two LLMs compete — one writing code, one writing tests — to produce more robust code generation.**

Final project for CSCI 2470 (Deep Learning) at Brown University. Built end-to-end from scratch: reasoning pipeline, PPO loop, isolated execution sandbox, reward design, training harness, and evaluation suite.

## Overview

Modern code-generation LLMs jump straight to implementation and produce solutions that look plausible but break on corner cases. We asked a sharper question: **can two small LLMs teach each other to be better at code, with nothing but a Python interpreter in the loop?**

The answer, demonstrated in this repo, is yes. The system has three components:

1. **Generator** — an LLM that produces a Python solution to a coding problem.
2. **Discriminator** — a separate LLM that reads the generator's output and produces adversarial test cases designed to break it.
3. **Sandbox** — a hardened subprocess executor that runs generated code and tests against a ground-truth reference, returning rewards to both models.

Both models are fine-tuned with **PPO (Proximal Policy Optimization)** using LoRA adapters on top of a frozen 4-bit quantized base model. **Rewards come entirely from execution outcomes against a reference implementation** — no human feedback, no labeled reasoning traces, no reward model. The whole learning signal is a pass/fail vector from a Python subprocess.

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

The generator produces a solution through five explicit prompts, each conditioned on the previous outputs:

1. **Informal Reasoning** — intuitive read of the problem
2. **Structured Reasoning** — numbered breakdown, observations, edge cases
3. **Pseudocode** — language-agnostic algorithm
4. **Constraints & Invariants** — pre/post-conditions, complexity, loop invariants
5. **Executable Code** — final Python implementation

The discriminator is invoked at every stage, generating tests targeted at whatever the generator has committed to so far (happy-path tests early, constraint-violating stress tests late).

This design is in `5-stage-reasoning/` and is fully implemented, but the compute required to train both models through five stages per problem exceeded what we had available.

### 1-Stage Reasoning (final system)

We distilled the design to a single forward pass: the generator produces code directly, and the discriminator writes a batch of adversarial test cases from the problem statement alone. Trimming the pipeline made training tractable within a Colab budget while preserving the core adversarial dynamic — and this is the configuration that drove all reported training runs and evaluation results. All supporting code lives in `1-stage-reasoning/`.

## Reward Shaping

Rewards are computed from execution outcomes against a reference solution. For each generated test `(inputs, expected)`:

- If the reference solution produces `expected` on `inputs`, the test is **valid**. The discriminator is credited; the test is then run against the generator's code.
- If the test is valid *and* the generator's code fails it, the discriminator receives an additional **bug-catch** reward and the generator is penalized.
- If the test is invalid (disagrees with the reference, malformed, or unparseable), the discriminator is penalized.

This design keeps the incentives honest in both directions: the discriminator cannot farm rewards by emitting garbage or contradictory tests, and the generator is only penalized by tests that the ground truth actually passes. It also produces a naturally adaptive curriculum — as the generator improves, the only tests that still earn bug-catch reward are the genuinely adversarial ones. The exact reward constants are defined in `1-stage-reasoning/sandbox.py`.

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

Training the two 8B LoRA models plus the sandbox exceeds a single Colab T4's memory. The training notebook supports splitting the generator, discriminator, and sandbox across three separate Colab sessions, passing outputs and rewards between them by copy-paste — see `1-stage-reasoning/1stage.md`.

## Evaluation

Trained and untrained checkpoints of both models are evaluated on held-out LeetCode problems. The protocol is:

- **Generator:** fraction of baseline tests passed per problem.
- **Discriminator:** (a) fraction of generated tests that agree with the reference solution, and (b) fraction of valid tests that expose bugs in the generator's code.

Across 100 PPO steps, the trained discriminator produced a **higher bug-catch rate** on held-out problems than its untrained counterpart — concrete evidence that pure execution-driven adversarial training actually taught one of the models something non-trivial about where code-generation models fail. Raw per-problem results are in `evals/results/*.csv`; training loss and reward traces live in `1-stage-reasoning/log1.jsonl` and are analyzed in `1-stage-reasoning/analysis.ipynb`.

## Reproducing Results

**1-Stage:**

Open `1-stage-reasoning/csci2470_1stageFinal.ipynb` in Colab with ~48 GB of VRAM available (A100). Import `sandbox.py` and `rl_loop.py`, run all cells.

If you only have multiple smaller GPUs / Colab instances, run `generator.ipynb`, `discriminator.ipynb`, and `sandbox.ipynb` in separate sessions and shuttle outputs between them.

**5-Stage (prototype):**

```bash
cd 5-stage-reasoning
pip install -r requirements.txt
python run_training.py --generator-model Qwen/Qwen2.5-Coder-1.5B-Instruct
python run_inference.py
```

Configuration lives in `5-stage-reasoning/training/config.py`.

## Results and Takeaways

- **Stable PPO under pure execution feedback.** Two 8B LoRA models trained jointly against a sandbox, with no reward model and no human feedback, producing clean, monotonic loss curves across 100 steps — the hard part of RLHF-style training, made to work without the H or the F.
- **The discriminator actually learned.** Post-training, it writes more parseable tests, more tests that agree with the reference solution, and catches more bugs per valid test than the untrained baseline. The adversarial signal is real and it transfers to held-out problems.
- **A fully implemented 5-stage reasoning pipeline.** Every stage — informal reasoning, structured reasoning, pseudocode, constraints & invariants, code — is specified with its own prompt template, reward shaping, and trainer. Compute, not code, is what kept us from running it at full scale, and the pipeline is drop-in ready for anyone with an A100 budget.
- **Honest limits.** On our filtered held-out split, generator pass-rate improvements were small and noisy — expected for a 1.5B–8B policy with 100 PPO steps over ~dozens of problems. Scaling either the step count or the base model is the obvious next move.

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

PyTorch · Hugging Face Transformers · PEFT (LoRA) · bitsandbytes · TRL-style PPO (reimplemented) · Hugging Face datasets format · Google Colab

## License

MIT.
