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
    input_data: Optional[Dict[str, Any]] = None
    tools: Dict[str, Any]
    financials_schema: Optional[Dict[str, Any]] = None
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


def get_financials_schema(config: ExperimentConfig) -> Dict[str, Any]:
    """Extract financials schema from experiment config."""
    if config.financials_schema is None:
        # Return default schema if not specified
        return {
            "fields": [
                {"name": "revenue", "type": "float", "required": True},
                {"name": "assets", "type": "float", "required": True},
                {"name": "equity", "type": "float", "required": True},
                {"name": "debt", "type": "float", "required": True},
                {"name": "cash", "type": "float", "required": True},
                {"name": "interest_coverage_ratio", "type": "float", "required": True},
            ],
            "default_values": {
                "revenue": 5000000,
                "assets": 10000000,
                "equity": 5000000,
                "debt": 5000000,
                "cash": 1000000,
                "interest_coverage_ratio": 10,
            }
        }
    return config.financials_schema



