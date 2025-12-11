"""Training configuration dataclass."""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for adversarial RL training."""
    
    # Model settings
    generator_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    discriminator_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    device: str = "cpu"
    
    # Training hyperparameters
    n_discriminator_steps: int = 10
    n_generator_steps: int = 10
    k_alternating_steps: int = 5
    learning_rate: float = 1e-5
    clip_epsilon: float = 0.2
    
    # Execution settings
    sandbox_timeout: int = 5
    num_tests_per_problem: int = 5
    
    # Multi-attempt settings
    enable_multi_attempt: bool = False
    max_attempts: int = 3
    
    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
