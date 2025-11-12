"""
Configuration loader for experiments.
Loads YAML configs and creates tool/graph configurations.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class TavilyConfig(BaseModel):
    """Configuration for Tavily search tool."""
    max_results: int = Field(default=5, ge=1, le=20)
    include_answer: bool = True
    include_raw_content: bool = False
    include_images: bool = False
    search_depth: str = Field(default="advanced", pattern="^(basic|advanced)$")
    time_range: Optional[str] = Field(default=None, pattern="^(1d|7d|30d|1y)$")


class ExperimentConfig(BaseModel):
    """Full experiment configuration."""
    experiment: Dict[str, Any]
    tools: Dict[str, Any]
    graph: Dict[str, Any]
    decision_rules: Dict[str, Any]


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load experiment configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return ExperimentConfig(**config_dict)


def get_tavily_config(config: ExperimentConfig) -> TavilyConfig:
    """Extract Tavily configuration from experiment config."""
    tavily_dict = config.tools.get("tavily", {})
    return TavilyConfig(**tavily_dict)

