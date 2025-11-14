#!/usr/bin/env python3
"""
Generate Mermaid diagram from LangGraph model.
This script extracts the graph structure and generates a Mermaid visualization.
"""
import sys
import os

# Suppress warnings and output from credit_committee import
import warnings
warnings.filterwarnings('ignore')

# Redirect stdout to suppress print statements
from io import StringIO
old_stdout = sys.stdout
sys.stdout = StringIO()

try:
    from credit_committee import app
    
    # Get the graph and generate Mermaid diagram
    graph = app.get_graph()
    mermaid_diagram = graph.draw_mermaid()
    
    # Restore stdout
    sys.stdout = old_stdout
    
    # Output just the Mermaid diagram
    print(mermaid_diagram)
    
except Exception as e:
    sys.stdout = old_stdout
    print(f"Error generating Mermaid diagram: {e}", file=sys.stderr)
    sys.exit(1)



