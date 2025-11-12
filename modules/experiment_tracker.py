"""
Experiment tracking module.
Supports Weights & Biases (W&B) and local file-based tracking.
"""
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


class ExperimentTracker:
    """Track experiments with W&B and/or local storage."""
    
    def __init__(
        self,
        project_name: str = "credit-committee",
        use_wandb: bool = True,
        local_dir: Optional[str] = "experiments"
    ):
        self.project_name = project_name
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.local_dir = Path(local_dir) if local_dir else None
        self.experiment_id = None
        self.start_time = None
        
        if self.use_wandb:
            self.wandb_run = None
        else:
            print("W&B not available, using local tracking only")
    
    def start(self, config: Dict[str, Any], experiment_name: Optional[str] = None):
        """Start a new experiment."""
        self.start_time = time.time()
        self.experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if self.use_wandb:
            self.wandb_run = wandb.init(
                project=self.project_name,
                config=config,
                name=experiment_name or self.experiment_id,
                tags=config.get("experiment", {}).get("tags", [])
            )
        
        if self.local_dir:
            self.local_dir.mkdir(parents=True, exist_ok=True)
            exp_dir = self.local_dir / self.experiment_id
            exp_dir.mkdir(exist_ok=True)
            
            # Save config
            with open(exp_dir / "config.json", 'w') as f:
                json.dump(config, f, indent=2)
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to tracker."""
        if self.use_wandb:
            wandb.log(metrics)
        
        if self.local_dir:
            metrics_file = self.local_dir / self.experiment_id / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    existing = json.load(f)
            else:
                existing = {}
            existing.update(metrics)
            with open(metrics_file, 'w') as f:
                json.dump(existing, f, indent=2)
    
    def log_result(self, result: Dict[str, Any]):
        """Log experiment result."""
        if self.local_dir:
            result_file = self.local_dir / self.experiment_id / "result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
    
    def log_artifact(self, file_path: str, artifact_type: str = "result"):
        """Log an artifact (file)."""
        if self.use_wandb:
            artifact = wandb.Artifact(artifact_type, type="dataset")
            artifact.add_file(file_path)
            wandb.log_artifact(artifact)
    
    def finish(self):
        """Finish the experiment."""
        if self.start_time:
            duration = time.time() - self.start_time
            self.log_metrics({"duration_seconds": duration})
        
        if self.use_wandb:
            wandb.finish()

