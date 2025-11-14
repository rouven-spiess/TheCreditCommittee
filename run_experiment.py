#!/usr/bin/env python3
"""
Run experiments with configuration files.
Example: python run_experiment.py configs/example_experiment.yaml
"""
import sys
from pathlib import Path
from modules.config_loader import load_config, get_tavily_config
from modules.experiment_tracker import ExperimentTracker
from modules.tools import create_tavily_tool
from credit_committee import create_graph_from_config

def main(config_path: str):
    """Run an experiment from a config file."""
    # Load configuration
    config = load_config(config_path)
    exp_name = config.experiment.get("name", "unnamed_experiment")
    
    print(f"Starting experiment: {exp_name}")
    print(f"Description: {config.experiment.get('description', 'N/A')}")
    
    # Initialize tracker
    tracker = ExperimentTracker(project_name="credit-committee")
    # Convert Pydantic model to dict (works with both v1 and v2)
    config_dict = config.dict() if hasattr(config, 'dict') else config.model_dump()
    tracker.start(config_dict, experiment_name=exp_name)
    
    try:
        # Create tools from config
        tavily_config = get_tavily_config(config)
        tavily_tool = create_tavily_tool(tavily_config)
        
        # Get financials schema from config
        from modules.config_loader import get_financials_schema
        financials_schema = get_financials_schema(config)
        
        # Create graph from config
        app = create_graph_from_config(config, tavily_tool, financials_schema)
        
        # Get input data from config, or use default
        input_data = config.input_data
        if input_data is None:
            # Fallback to default if not in config
            input_data = {
                "ticker": "IBM"
            }
            print("Warning: No input_data in config, using default ticker: IBM")
        
        ticker = input_data.get('ticker', 'Unknown')
        if not ticker:
            raise ValueError("ticker is required in input_data")
        
        print(f"\nProcessing application for ticker: {ticker}")
        
        result = app.invoke({"input": input_data})
        
        # Log results
        decision = result["credit_officer"]["decision"]
        tracker.log_metrics({
            "decision": 1 if decision == "Approve" else 0,
            "altman_z_score": result["credit_risk"].get("altman_z_score", 0)
        })
        
        tracker.log_result({
            "decision": decision,
            "memo": result["credit_officer"]["memo"],
            "full_result": result
        })
        
        print(f"\nDecision: {decision}")
        print(f"Memo: {result['credit_officer']['memo'][:200]}...")
        
    except Exception as e:
        tracker.log_metrics({"error": 1})
        raise
    finally:
        tracker.finish()
        print(f"\nExperiment completed. ID: {tracker.experiment_id}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_experiment.py <config_file.yaml>")
        sys.exit(1)
    
    main(sys.argv[1])

