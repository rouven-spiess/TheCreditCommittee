import os
import requests
from typing import Optional
from pydantic import Field
from langchain_core.tools import BaseTool
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

# Load environment variables
load_dotenv()

# get api keys from environment variables
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found in environment variables. Please set it in .env file.")

alphavantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY")
if not alphavantage_api_key:
    raise ValueError("ALPHAVANTAGE_API_KEY not found in environment variables. Please set it in .env file.")

# Default tool (for backward compatibility)
tavily_api_wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_api_key)
tavily_search_tool = TavilySearchResults(api_wrapper=tavily_api_wrapper)

def create_tavily_tool(config=None):
    """Create Tavily search tool with custom configuration.
    
    Args:
        config: TavilyConfig object or dict with tavily parameters
        
    Returns:
        TavilySearchResults tool instance
    """
    from modules.config_loader import TavilyConfig
    
    if config is None:
        # Use default
        return TavilySearchResults(api_wrapper=tavily_api_wrapper)
    
    # Convert to TavilyConfig if dict
    if isinstance(config, dict):
        config = TavilyConfig(**config)
    
    # Create wrapper
    wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_api_key)
    
    # Create tool with config
    return TavilySearchResults(
        api_wrapper=wrapper,
        max_results=config.max_results,
        include_answer=config.include_answer,
        include_raw_content=config.include_raw_content,
        include_images=config.include_images,
        search_depth=config.search_depth,
    )

class alphavantage_tool(BaseTool):
    name: str = "alphavantage_tool"
    description: str = "Fetches income statement data from Alpha Vantage API for a given stock symbol"
    symbol: str = Field(description="Stock symbol to fetch income statement for")
    
    def __init__(self, symbol: str, **kwargs):
        super().__init__(symbol=symbol, **kwargs)
    
    def _run(self, symbol: Optional[str] = None) -> dict:
        """Execute the tool. Uses instance symbol if not provided."""
        symbol_to_use = symbol or self.symbol
        url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol_to_use}&apikey={alphavantage_api_key}'
        response = requests.get(url)
        return response.json()
        
    def get(self) -> dict:
        """Legacy method for backward compatibility."""
        return self._run()
