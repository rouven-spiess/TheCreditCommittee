import os
import requests
import json
from typing import Optional, Dict, Any
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

def _map_alphavantage_to_schema(av_response: dict, schema: dict) -> dict:
    """Map Alpha Vantage API response to match the financials schema.
    
    Args:
        av_response: Raw response from Alpha Vantage API
        schema: Schema dict with 'fields' and 'default_values' keys
    
    Returns:
        Dict with keys matching schema field names
    """
    # Get schema field names and default values
    schema_fields = {field['name']: field for field in schema.get('fields', [])}
    default_values = schema.get('default_values', {})
    
    # Initialize result with default values
    result = default_values.copy()
    
    # Alpha Vantage field mappings
    # INCOME_STATEMENT fields
    # BALANCE_SHEET fields (for assets, equity, debt, cash)
    field_mappings = {
        'revenue': ['totalRevenue', 'revenue', 'total_revenue'],
        'assets': ['totalAssets', 'assets', 'total_assets'],
        'equity': ['totalShareholderEquity', 'shareholdersEquity', 'total_shareholder_equity', 'shareholdersEquity'],
        'debt': ['totalDebt', 'totalLiabilities', 'total_debt', 'totalLiabilities'],
        'cash': ['cashAndCashEquivalentsAtCarryingValue', 'cashAndShortTermInvestments', 'cash', 'cashAndCashEquivalents'],
        'interest_coverage_ratio': ['interestCoverageRatio', 'interest_coverage_ratio'],
    }
    
    # Extract annual reports (most recent first)
    annual_reports = av_response.get('annualReports', [])
    if not annual_reports:
        # Try quarterly reports as fallback
        annual_reports = av_response.get('quarterlyReports', [])
    
    if annual_reports:
        # Use most recent report (first in list)
        latest_report = annual_reports[0]
        
        # Map each schema field
        for schema_field_name in schema_fields.keys():
            if schema_field_name in field_mappings:
                # Try each possible Alpha Vantage field name
                for av_field_name in field_mappings[schema_field_name]:
                    if av_field_name in latest_report:
                        value = latest_report[av_field_name]
                        # Convert to float if it's a string
                        if isinstance(value, str):
                            try:
                                # Remove commas and convert
                                value = float(value.replace(',', ''))
                            except (ValueError, AttributeError):
                                continue
                        result[schema_field_name] = float(value)
                        break
        
        # Calculate interest_coverage_ratio if not directly available
        if 'interest_coverage_ratio' in schema_fields and 'interest_coverage_ratio' not in result:
            ebit = None
            interest_expense = None
            
            # Try to get EBIT
            for ebit_field in ['ebit', 'EBIT', 'earningsBeforeInterestAndTaxes']:
                if ebit_field in latest_report:
                    ebit_val = latest_report[ebit_field]
                    if isinstance(ebit_val, str):
                        try:
                            ebit = float(ebit_val.replace(',', ''))
                        except (ValueError, AttributeError):
                            pass
                    else:
                        ebit = float(ebit_val)
                    break
            
            # Try to get interest expense
            for int_field in ['interestExpense', 'interestAndDebtExpense', 'interest_expense']:
                if int_field in latest_report:
                    int_val = latest_report[int_field]
                    if isinstance(int_val, str):
                        try:
                            interest_expense = float(int_val.replace(',', ''))
                        except (ValueError, AttributeError):
                            pass
                    else:
                        interest_expense = float(int_val)
                    break
            
            # Calculate ratio if we have both
            if ebit is not None and interest_expense is not None and interest_expense != 0:
                result['interest_coverage_ratio'] = ebit / interest_expense
    
    # Ensure all required fields are present (use defaults if missing)
    for field_name, field_info in schema_fields.items():
        if field_info.get('required', False) and field_name not in result:
            if field_name in default_values:
                result[field_name] = default_values[field_name]
    
    return result


def create_alphavantage_tool(symbol: str, financials_schema=None, config_path=None):
    """Create Alpha Vantage tool with schema transformation.
    
    Args:
        symbol: Stock symbol to fetch data for
        financials_schema: Dict with 'fields' and 'default_values' keys
        config_path: Optional path to YAML config file to load schema from
    
    Returns:
        alphavantage_tool instance configured with schema
    """
    # Load schema if not provided
    if financials_schema is None:
        if config_path is None:
            # Try default config path
            from pathlib import Path
            default_config = Path("configs/example_experiment.yaml")
            if default_config.exists():
                config_path = str(default_config)
        
        if config_path:
            try:
                from modules.config_loader import load_config, get_financials_schema
                config = load_config(config_path)
                financials_schema = get_financials_schema(config)
            except Exception as e:
                print(f"Warning: Could not load schema from config: {e}")
                financials_schema = None
    
    # Fallback to default schema if still None
    if financials_schema is None:
        financials_schema = {
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
    
    # Create tool instance with schema
    tool = alphavantage_tool(symbol=symbol, financials_schema=financials_schema)
    return tool


class alphavantage_tool(BaseTool):
    name: str = "alphavantage_tool"
    description: str = "Fetches income statement data from Alpha Vantage API for a given stock symbol and transforms it to match the configured financials schema"
    symbol: str = Field(description="Stock symbol to fetch income statement for")
    financials_schema: Optional[dict] = Field(default=None, description="Financials schema to transform response to")
    
    def __init__(self, symbol: str, financials_schema: Optional[dict] = None, **kwargs):
        super().__init__(symbol=symbol, financials_schema=financials_schema, **kwargs)
    
    def _run(self, symbol: Optional[str] = None) -> dict:
        """Execute the tool. Uses instance symbol if not provided.
        Returns data matching the financials schema exactly.
        """
        symbol_to_use = symbol or self.symbol
        
        # Check if we need balance sheet data (for assets, equity, debt, cash)
        needs_balance_sheet = False
        if self.financials_schema:
            schema_fields = {field['name'] for field in self.financials_schema.get('fields', [])}
            balance_sheet_fields = {'assets', 'equity', 'debt', 'cash'}
            needs_balance_sheet = bool(schema_fields & balance_sheet_fields)
        
        # Fetch from Alpha Vantage - INCOME_STATEMENT
        income_url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol_to_use}&apikey={alphavantage_api_key}'
        income_response = requests.get(income_url)
        income_data = income_response.json()
        
        # Fetch BALANCE_SHEET if needed
        balance_data = {}
        if needs_balance_sheet:
            balance_url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={symbol_to_use}&apikey={alphavantage_api_key}'
            balance_response = requests.get(balance_url)
            balance_data = balance_response.json()
        
        # Check for API errors
        has_error = (
            'Error Message' in income_data or 'Note' in income_data or
            'Error Message' in balance_data or 'Note' in balance_data
        )
        
        if has_error:
            # API error or rate limit - return default values from schema
            if self.financials_schema:
                return self.financials_schema.get('default_values', {}).copy()
            return income_data
        
        # Merge income statement and balance sheet data
        merged_data = income_data.copy()
        if balance_data:
            # Merge annual reports
            if 'annualReports' in balance_data and 'annualReports' in merged_data:
                # Merge the most recent reports
                if merged_data['annualReports'] and balance_data['annualReports']:
                    merged_data['annualReports'][0].update(balance_data['annualReports'][0])
            elif 'annualReports' in balance_data:
                merged_data['annualReports'] = balance_data['annualReports']
        
        # Transform to match schema
        if self.financials_schema:
            return _map_alphavantage_to_schema(merged_data, self.financials_schema)
        
        # If no schema, return raw response
        return merged_data
        
    def get(self) -> dict:
        """Legacy method for backward compatibility."""
        return self._run()


def create_openai_synthesis_tool(model: str = "gpt-4o-mini", temperature: float = 0.3):
    """Create an OpenAI tool for synthesizing and summarizing search results.
    
    Args:
        model: OpenAI model to use (default: "gpt-4o-mini")
        temperature: Temperature for generation (default: 0.3 for more focused output)
    
    Returns:
        OpenAI synthesis tool instance
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")
    
    tool = openai_synthesis_tool(
        model=model,
        temperature=temperature,
        api_key=openai_api_key
    )
    return tool


class openai_synthesis_tool(BaseTool):
    name: str = "openai_synthesis_tool"
    description: str = "Synthesizes and summarizes search results using OpenAI. Takes raw search results and creates a compact, coherent summary with key insights."
    query: Optional[str] = Field(default=None, description="The original search query that was used")
    search_results: Optional[Dict[str, Any]] = Field(default=None, description="Raw search results from Tavily or other search tools")
    synthesis_prompt: Optional[str] = Field(default=None, description="Custom prompt for synthesis (optional)")
    model: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
    temperature: float = Field(default=0.3, description="Temperature for generation")
    api_key: str = Field(description="OpenAI API key")
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.3, api_key: str = None, **kwargs):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. Please provide it or set it in .env file.")
        super().__init__(model=model, temperature=temperature, api_key=api_key, **kwargs)
    
    def _run(self, query: str, search_results: Dict[str, Any], synthesis_prompt: Optional[str] = None) -> str:
        """Synthesize search results into a compact summary.
        
        Args:
            query: The original search query
            search_results: Raw search results from Tavily
            synthesis_prompt: Optional custom prompt
        
        Returns:
            Synthesized summary as a string
        """
        try:
            import openai
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")
        
        # Extract relevant information from Tavily results
        content_parts = []
        
        # Handle different Tavily result formats
        if isinstance(search_results, dict):
            # Check if it has an 'answer' field (Tavily's AI-generated answer)
            if "answer" in search_results:
                content_parts.append(f"AI-Generated Answer: {search_results['answer']}")
            
            # Extract content from results array
            if "results" in search_results:
                for i, result in enumerate(search_results["results"][:5], 1):  # Limit to top 5
                    title = result.get("title", "")
                    content = result.get("content", "")
                    url = result.get("url", "")
                    if content:
                        content_parts.append(f"Source {i} ({title}): {content[:500]}")  # Truncate long content
        elif isinstance(search_results, list):
            # Handle list of results
            for i, result in enumerate(search_results[:5], 1):
                if isinstance(result, dict):
                    title = result.get("title", "")
                    content = result.get("content", "")
                    if content:
                        content_parts.append(f"Source {i} ({title}): {content[:500]}")
                else:
                    content_parts.append(f"Source {i}: {str(result)[:500]}")
        else:
            # Fallback: convert to string
            content_parts.append(str(search_results)[:2000])
        
        # Prepare the synthesis prompt
        if synthesis_prompt is None:
            synthesis_prompt = """You are a financial analyst synthesizing search results for a credit risk assessment.

Your task is to:
1. Extract the most relevant and important information
2. Identify key insights, risks, and opportunities
3. Create a concise, well-structured summary (2-4 paragraphs)
4. Focus on facts that are relevant for credit risk evaluation
5. Avoid redundancy and speculation

Be objective, factual, and focus on information that would be useful for making a credit decision."""

        # Build the full prompt
        full_prompt = f"""{synthesis_prompt}

Original Query: {query}

Search Results:
{chr(10).join(content_parts)}

Please provide a concise synthesis of the above search results, focusing on the most relevant information for credit risk assessment."""

        # Call OpenAI API
        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst expert at synthesizing information for credit risk assessments."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=self.temperature,
                max_tokens=500  # Keep summaries compact
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
            
        except Exception as e:
            # Fallback: return a basic summary if API fails
            print(f"Warning: OpenAI API call failed: {e}")
            if content_parts:
                return f"Summary of search results for '{query}': {content_parts[0][:300]}..."
            return f"Unable to synthesize results for '{query}'. Raw results available."
    
    def invoke(self, query: str, search_results: Dict[str, Any], synthesis_prompt: Optional[str] = None) -> str:
        """Invoke the tool (LangChain compatible)."""
        return self._run(query, search_results, synthesis_prompt)
