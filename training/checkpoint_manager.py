"""Checkpoint manager for saving and loading model checkpoints during training."""

import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    stage: int
    epoch: int
    timestamp: str
    metrics: Dict[str, float]
    checkpoint_path: str
    is_best: bool


class CheckpointManager:
    """Manages saving and loading of model checkpoints."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_score = float('-inf')
        self.best_checkpoint_path = None
        
    def save_checkpoint(
        self,
        generator,
        discriminator,
        stage: int,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> str:
        """Save model checkpoint with metadata.
        
        Args:
            generator: Generator model instance
            discriminator: Discriminator model instance
            stage: Current training stage (1-5)
            epoch: Current epoch number
            metrics: Training metrics dictionary
            is_best: Whether this is the best checkpoint so far
            
        Returns:
            Path to saved checkpoint file
        """
        timestamp = datetime.now().isoformat()
        
        # Create checkpoint filename
        checkpoint_name = f"checkpoint_stage_{stage}_epoch_{epoch}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint_data = {
            "stage": stage,
            "epoch": epoch,
            "timestamp": timestamp,
            "generator_state_dict": generator.model.state_dict(),
            "discriminator_state_dict": discriminator.model.state_dict(),
            "metrics": metrics,
            "config": {
                "generator_model": generator.model_name,
                "discriminator_model": discriminator.model_name,
                "device": generator.device
            }
        }
        
        # Save checkpoint
        try:
            torch.save(checkpoint_data, checkpoint_path)
            print(f"✓ Saved checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"✗ Failed to save checkpoint: {e}")
            # Retry once
            try:
                torch.save(checkpoint_data, checkpoint_path)
                print(f"✓ Saved checkpoint on retry: {checkpoint_path}")
            except Exception as e2:
                print(f"✗ Failed to save checkpoint on retry: {e2}")
                raise
        
        # Save metadata separately for easy access
        metadata_path = checkpoint_path.with_suffix('.json')
        metadata = {
            "stage": stage,
            "epoch": epoch,
            "timestamp": timestamp,
            "metrics": metrics,
            "checkpoint_path": str(checkpoint_path),
            "is_best": is_best
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Update best checkpoint if needed
        if is_best:
            self._update_best_checkpoint(checkpoint_path, metrics)
        
        return str(checkpoint_path)
    
    def _update_best_checkpoint(self, checkpoint_path: Path, metrics: Dict[str, float]):  # noqa: ARG002
        """Update the best checkpoint symlink/copy.
        
        Args:
            checkpoint_path: Path to the new best checkpoint
            metrics: Metrics for this checkpoint
        """
        best_checkpoint_path = self.checkpoint_dir / "checkpoint_best.pt"
        best_metadata_path = self.checkpoint_dir / "checkpoint_best.json"
        
        # Copy checkpoint file (symlinks don't work well on all platforms)
        try:
            import shutil
            shutil.copy2(checkpoint_path, best_checkpoint_path)
            
            # Copy metadata
            metadata_path = checkpoint_path.with_suffix('.json')
            if metadata_path.exists():
                shutil.copy2(metadata_path, best_metadata_path)
            
            self.best_checkpoint_path = str(best_checkpoint_path)
            print(f"✓ Updated best checkpoint: {best_checkpoint_path}")
        except Exception as e:
            print(f"✗ Failed to update best checkpoint: {e}")
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        generator,
        discriminator
    ) -> Dict[str, Any]:
        """Load models from checkpoint and return metadata.
        
        Args:
            checkpoint_path: Path to checkpoint file
            generator: Generator model instance to load weights into
            discriminator: Discriminator model instance to load weights into
            
        Returns:
            Dictionary containing checkpoint metadata
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Validate checkpoint exists
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Validate checkpoint integrity
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location=generator.device)
        except Exception as e:
            raise RuntimeError(f"Corrupted checkpoint file: {checkpoint_path}. Error: {e}") from e
        
        # Load model weights
        try:
            generator.model.load_state_dict(checkpoint_data["generator_state_dict"])
            discriminator.model.load_state_dict(checkpoint_data["discriminator_state_dict"])
            print(f"✓ Loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights from checkpoint: {e}") from e
        
        # Return metadata
        metadata = {
            "stage": checkpoint_data.get("stage", 0),
            "epoch": checkpoint_data.get("epoch", 0),
            "timestamp": checkpoint_data.get("timestamp", ""),
            "metrics": checkpoint_data.get("metrics", {}),
            "config": checkpoint_data.get("config", {})
        }
        
        return metadata
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Return path to best performing checkpoint.
        
        Returns:
            Path to best checkpoint, or None if no checkpoints exist
        """
        best_checkpoint_path = self.checkpoint_dir / "checkpoint_best.pt"
        
        if best_checkpoint_path.exists():
            return str(best_checkpoint_path)
        
        return None
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Return path to most recent checkpoint.
        
        Returns:
            Path to latest checkpoint, or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            return None
        
        # Parse checkpoint files to find latest by stage and epoch
        latest_checkpoint = None
        latest_stage = -1
        latest_epoch = -1
        
        for checkpoint_path in checkpoints:
            # Skip best checkpoint symlink
            if "checkpoint_best" in checkpoint_path:
                continue
            
            # Extract stage and epoch from filename
            try:
                filename = Path(checkpoint_path).stem
                parts = filename.split('_')
                stage = int(parts[2])
                epoch = int(parts[4])
                
                if stage > latest_stage or (stage == latest_stage and epoch > latest_epoch):
                    latest_stage = stage
                    latest_epoch = epoch
                    latest_checkpoint = checkpoint_path
            except (IndexError, ValueError):
                continue
        
        return latest_checkpoint
    
    def list_checkpoints(self, stage: Optional[int] = None) -> List[str]:
        """List all available checkpoints, optionally filtered by stage.
        
        Args:
            stage: Optional stage number to filter by
            
        Returns:
            List of checkpoint file paths
        """
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("checkpoint_*.pt"):
            checkpoint_path = str(checkpoint_file)
            
            # Filter by stage if specified
            if stage is not None:
                if f"_stage_{stage}_" not in checkpoint_path:
                    continue
            
            checkpoints.append(checkpoint_path)
        
        return sorted(checkpoints)
    
    def compute_checkpoint_score(self, metrics: Dict[str, float]) -> float:
        """Compute combined score for checkpoint ranking.
        
        Uses weighted combination: 0.7 * generator_reward + 0.3 * test_validity
        
        Args:
            metrics: Dictionary of training metrics
            
        Returns:
            Combined score for ranking checkpoints
        """
        generator_reward = metrics.get("generator_reward", 0.0)
        test_validity = metrics.get("test_validity", 0.0)
        
        score = 0.7 * generator_reward + 0.3 * test_validity
        return score
    
    def should_save_as_best(self, metrics: Dict[str, float]) -> bool:
        """Determine if current checkpoint should be saved as best.
        
        Args:
            metrics: Current training metrics
            
        Returns:
            True if this checkpoint is better than previous best
        """
        current_score = self.compute_checkpoint_score(metrics)
        
        if current_score > self.best_score:
            self.best_score = current_score
            return True
        
        return False
