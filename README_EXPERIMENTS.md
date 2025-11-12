# Running Experiments

## Quick Start

1. **Install dependencies:**
```bash
uv sync
```

2. **Set up W&B (optional but recommended):**
```bash
pip install wandb
wandb login
```

3. **Create a config file** (see `configs/example_experiment.yaml`)

4. **Run an experiment:**
```bash
python run_experiment.py configs/example_experiment.yaml
```

## Configuration Structure

### Tool Configuration

```yaml
tools:
  tavily:
    max_results: 5          # Number of search results (1-20)
    include_answer: true    # Include AI-generated answer
    include_raw_content: false
    include_images: false
    search_depth: "advanced"  # "basic" or "advanced"
    time_range: "7d"        # "1d", "7d", "30d", "1y", or null
```

### Graph Configuration

```yaml
graph:
  nodes:
    - name: "sanitize_input"
      enabled: true
    - name: "due_diligence"
      enabled: true
    # ... more nodes
  
  edges:
    - from: "sanitize_input"
      to: ["due_diligence", "credit_risk"]
    # ... more edges
```

## Experiment Tracking

### With W&B

Experiments are automatically logged to W&B with:
- Configuration parameters
- Metrics (decision accuracy, processing time, etc.)
- Results and artifacts

View at: https://wandb.ai

### Local Tracking

Results are saved to `experiments/exp_YYYYMMDD_HHMMSS/`:
- `config.json` - Full configuration
- `metrics.json` - Metrics over time
- `result.json` - Final results

## Example Experiments

### Experiment 1: Different Time Ranges
```yaml
# configs/tavily_1d.yaml
tools:
  tavily:
    time_range: "1d"  # Only last 24 hours
```

### Experiment 2: More Search Results
```yaml
# configs/tavily_10_results.yaml
tools:
  tavily:
    max_results: 10
    search_depth: "advanced"
```

### Experiment 3: Custom Graph Structure
```yaml
# configs/custom_graph.yaml
graph:
  edges:
    - from: "sanitize_input"
      to: ["due_diligence"]
    - from: "due_diligence"
      to: ["credit_risk"]
    - from: "credit_risk"
      to: ["industry"]
    - from: "industry"
      to: ["credit_officer"]
```

## Comparing Experiments

Use W&B dashboard to compare:
- Different time ranges
- Different search depths
- Different graph structures
- Decision accuracy across configurations

