# Experiment Tracking & Configuration Management

## Recommended Framework: **Weights & Biases (W&B)**

**Why W&B:**
- ✅ Excellent for ML/AI experiment tracking
- ✅ Free tier with generous limits
- ✅ Great visualization and comparison tools
- ✅ Easy integration with Python
- ✅ Tracks configs, metrics, artifacts automatically
- ✅ Supports hyperparameter sweeps
- ✅ Team collaboration features

**Alternatives:**
- **MLflow**: Open-source, self-hosted option
- **Neptune**: Good for research teams
- **TensorBoard**: Simple but less feature-rich

## Recommended Data Structure

### 1. Configuration Files (YAML/TOML)

Use YAML or TOML for human-readable configs:

```yaml
# configs/experiment_001.yaml
experiment:
  name: "baseline_tavily_7d"
  description: "Baseline with 7-day time range"
  
tools:
  tavily:
    max_results: 5
    include_answer: true
    include_raw_content: false
    include_images: false
    search_depth: "advanced"
    time_range: "7d"
  
  alphavantage:
    function: "INCOME_STATEMENT"
    
graph:
  nodes:
    - name: "sanitize_input"
      enabled: true
    - name: "due_diligence"
      enabled: true
    - name: "industry"
      enabled: true
    - name: "credit_risk"
      enabled: true
    - name: "credit_officer"
      enabled: true
  
  edges:
    - from: "sanitize_input"
      to: ["due_diligence", "credit_risk"]
    - from: "due_diligence"
      to: "industry"
    - from: "credit_risk"
      to: "industry"
    - from: "industry"
      to: "credit_officer"

decision_rules:
  altman_z_threshold: 2.6
  require_no_major_risks: true
```

### 2. Experiment Metadata Structure

```python
{
    "experiment_id": "exp_001",
    "timestamp": "2025-01-XX",
    "config": {...},  # Full config dict
    "metrics": {
        "decision_accuracy": 0.85,
        "avg_processing_time": 2.3,
        "total_api_calls": 15
    },
    "results": {
        "decisions": ["Approve", "Reject", ...],
        "sample_outputs": [...]
    },
    "artifacts": {
        "graph_visualization": "path/to/mermaid.png",
        "full_results": "path/to/results.json"
    }
}
```

## Implementation Approach

### Option 1: Pydantic Config Models (Recommended)

```python
from pydantic import BaseModel
from typing import List, Optional

class TavilyConfig(BaseModel):
    max_results: int = 5
    include_answer: bool = True
    include_raw_content: bool = False
    include_images: bool = False
    search_depth: str = "advanced"
    time_range: Optional[str] = None

class GraphNodeConfig(BaseModel):
    name: str
    enabled: bool = True

class ExperimentConfig(BaseModel):
    name: str
    description: str
    tools: dict
    graph: dict
    decision_rules: dict
```

### Option 2: Simple Dict with Validation

Use YAML loader + basic validation for quick setup.

## Recommended Project Structure

```
TheCreditCommittee/
├── configs/
│   ├── experiments/
│   │   ├── baseline.yaml
│   │   ├── tavily_30d.yaml
│   │   └── extended_graph.yaml
│   └── default.yaml
├── experiments/
│   ├── exp_001/
│   │   ├── config.yaml
│   │   ├── results.json
│   │   └── artifacts/
│   └── exp_002/
├── modules/
│   ├── config_loader.py
│   ├── experiment_tracker.py
│   └── tools.py
└── run_experiment.py
```

## Quick Start with W&B

1. **Install**: `pip install wandb`
2. **Login**: `wandb login`
3. **Initialize in code**:
```python
import wandb

wandb.init(
    project="credit-committee",
    config=config_dict,
    name=experiment_name
)

# Log metrics
wandb.log({"accuracy": 0.85, "latency": 2.3})

# Log artifacts
wandb.log_artifact("results.json")
```

## Benefits

- **Reproducibility**: Config files ensure experiments are reproducible
- **Comparison**: Easy to compare different configurations
- **Versioning**: Track config changes over time
- **Collaboration**: Share experiments with team
- **Visualization**: W&B dashboard for metrics and comparisons

