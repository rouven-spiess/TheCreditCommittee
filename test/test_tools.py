import sys
import os
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from modules.tools import alphavantage_tool, tavily_search_tool

def test_alphavantage_tool():
    tool = alphavantage_tool(symbol="AAPL")
    response = tool._run()
    for result in response:
        print(result)
    assert response is not None
    # Note: Adjust assertions based on actual Alpha Vantage API response structure
    assert "symbol" in response or "Symbol" in response or len(response) > 0

def test_tavily_search_tool():
    tool = tavily_search_tool.invoke(f"{"AAPL"} recent news")
    response = tool
    for result in response:
        print(result)
    assert response is not None
    assert "results" in response or len(response) > 0