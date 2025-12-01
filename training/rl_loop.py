"""Reinforcement learning training loop with PPO."""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple


def compute_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    clip_epsilon: float = 0.2
) -> torch.Tensor:
    """Compute PPO clipped policy loss.
    
    PPO uses a clipped surrogate objective to prevent too large policy updates.
    
    Args:
        log_probs: Log probabilities from current policy
        old_log_probs: Log probabilities from old policy
        rewards: Reward values
        clip_epsilon: Clipping parameter (typically 0.2)
        
    Returns:
        Policy loss tensor
    """
    # Compute probability ratio
    ratio = torch.exp(log_probs - old_log_probs)
    
    # Normalize rewards (advantages)
    if rewards.std() > 1e-8:
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    else:
        advantages = rewards - rewards.mean()
    
    # Compute clipped objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    
    # PPO loss is negative of minimum (we want to maximize)
    policy_loss = -torch.min(surr1, surr2).mean()
    
    return policy_loss


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    prompts: List[str],
    outputs: List[str],
    rewards: List[float],
    old_log_probs_list: List[torch.Tensor],
    clip_epsilon: float = 0.2,
    max_grad_norm: float = 1.0
) -> Dict[str, float]:
    """Execute one RL training step.
    
    Args:
        model: Model to train (generator or discriminator)
        optimizer: Optimizer
        prompts: List of prompts
        outputs: List of generated outputs
        rewards: List of reward values
        old_log_probs_list: List of old log probability tensors
        clip_epsilon: PPO clipping parameter
        max_grad_norm: Maximum gradient norm for clipping
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    # Get current log probabilities
    # Note: This requires the model to have a get_log_probs method
    current_log_probs_list = []
    for prompt, output in zip(prompts, outputs):
        log_probs = model.get_log_probs(prompt, output)
        current_log_probs_list.append(log_probs)
    
    # Compute losses for each example
    losses = []
    for curr_lp, old_lp, reward in zip(current_log_probs_list, old_log_probs_list, rewards):
        # Sum log probs across tokens
        curr_lp_sum = curr_lp.sum()
        old_lp_sum = old_lp.sum()
        
        # Compute loss for this example
        loss = compute_policy_loss(
            curr_lp_sum.unsqueeze(0),
            old_lp_sum.unsqueeze(0),
            torch.tensor([reward], device=curr_lp.device),
            clip_epsilon
        )
        losses.append(loss)
    
    # Average loss across examples
    total_loss = torch.stack(losses).mean()
    
    # Check for NaN
    if torch.isnan(total_loss):
        print("Warning: NaN loss detected, skipping update")
        return {'policy_loss': float('nan')}
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    
    # Update weights
    optimizer.step()
    
    model.eval()
    
    return {
        'policy_loss': total_loss.item()
    }


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01
) -> torch.optim.Optimizer:
    """Create optimizer for model.
    
    Args:
        model: Model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        
    Returns:
        AdamW optimizer
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )


def freeze_model(model: nn.Module):
    """Freeze model parameters.
    
    Args:
        model: Model to freeze
    """
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


def unfreeze_model(model: nn.Module):
    """Unfreeze model parameters.
    
    Args:
        model: Model to unfreeze
    """
    for param in model.parameters():
        param.requires_grad = True
    model.train()
