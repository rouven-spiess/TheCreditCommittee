from typing import TypedDict, Optional
from langgraph.graph import StateGraph
from modules.tools import tavily_search_tool, alphavantage_tool, create_tavily_tool

# Define state schema
class State(TypedDict):
    input: dict
    due_diligence: dict
    industry: dict
    credit_risk: dict
    credit_officer: dict

# Input sanitization function
def sanitize_input(state):
    input_data = state.get("input", {})
    ticker = str(input_data.get("ticker", "")).strip().upper()
    if not ticker:
        raise ValueError("ticker is required in input_data")
    sanitized = {
        "ticker": ticker,
    }
    return {"input": sanitized}

# Define agents (nodes) - these will use the tool passed to them
def create_due_diligence_agent(tavily_tool):
    def due_diligence_agent(state):
        # Use Tavily for recent news, management info etc.
        ticker = state["input"]["ticker"]
        results_news = tavily_tool.invoke(f"{ticker} stock recent news")
        results_history = tavily_tool.invoke(f"{ticker} company history")
        results_management = tavily_tool.invoke(f"{ticker} company management team")
        return {
            "due_diligence": {
                "news": results_news,
                "history": results_history,
                "management": results_management,
            }
        }
    return due_diligence_agent

def create_industry_agent(tavily_tool):
    def industry_agent(state):
        ticker = state["input"]["ticker"]
        # Derive industry information from ticker/company
        trends = tavily_tool.invoke(f"{ticker} stock industry trends")
        competitive = tavily_tool.invoke(f"{ticker} company competitive landscape")
        economic = tavily_tool.invoke(f"{ticker} company economic outlook")
        return {
            "industry": {
                "trends": trends,
                "competitive_landscape": competitive,
                "economic_outlook": economic,
            }
        }
    return industry_agent

# Default agents using default tool (for backward compatibility)
due_diligence_agent = create_due_diligence_agent(tavily_search_tool)
industry_agent = create_industry_agent(tavily_search_tool)

def create_credit_risk_agent(financials_schema=None, config_path=None):
    """Create credit risk agent with configurable financials schema.
    
    Args:
        financials_schema: Dict with 'fields' and 'default_values' keys
        config_path: Optional path to YAML config file to load default schema from.
                     If None and financials_schema is None, tries to load from 
                     'configs/example_experiment.yaml' if it exists.
    """
    if financials_schema is None:
        # Try to load from config file
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
        
        # Fallback to hardcoded default if still None
        if financials_schema is None:
            financials_schema = {
                "default_values": {
                    "revenue": 5000000,
                    "assets": 10000000,
                    "equity": 5000000,
                    "debt": 5000000,
                    "cash": 1000000,
                    "interest_coverage_ratio": 10,
                }
            }
    
    def credit_risk_agent(state):
        ticker = state["input"]["ticker"]
        print(f"[Credit Risk Agent] Fetching financials for ticker: {ticker}")
        
        # Try to retrieve financials from AlphaVantage using the schema
        try:
            from modules.tools import create_alphavantage_tool
            tool = create_alphavantage_tool(
                symbol=ticker,  # Use ticker directly
                financials_schema=financials_schema
            )
            print(f"[Credit Risk Agent] Alpha Vantage tool created, fetching data...")
            financials = tool._run()
            print(f"[Credit Risk Agent] Received financials: {list(financials.keys())}")
            print(f"[Credit Risk Agent] Revenue: {financials.get('revenue', 'N/A')}, Assets: {financials.get('assets', 'N/A')}")
            
            # Ensure all schema fields are present
            schema_field_names = {f['name'] for f in financials_schema.get('fields', [])}
            for field_name in schema_field_names:
                if field_name not in financials:
                    # Use default if missing
                    print(f"[Credit Risk Agent] Missing field {field_name}, using default")
                    financials[field_name] = financials_schema.get('default_values', {}).get(field_name)
        except Exception as e:
            # Fallback to default values if AlphaVantage fails
            import traceback
            print(f"[Credit Risk Agent] ERROR: Could not fetch financials from AlphaVantage: {e}")
            print(f"[Credit Risk Agent] Traceback: {traceback.format_exc()}")
            financials = financials_schema.get("default_values", {}).copy()
            print(f"[Credit Risk Agent] Using default values: {financials}")
        
        # Compute Altman-Z score using financials data (simplified here)
        altman_z = compute_altman_z(financials)
        major_risks = identify_major_credit_risks(financials, state.get("industry", {}))
        return {
            "credit_risk": {
                "financials": financials,
                "altman_z_score": altman_z,
                "major_risks": major_risks,
            }
        }
    return credit_risk_agent

# Default credit risk agent (for backward compatibility)
credit_risk_agent = create_credit_risk_agent()

def credit_officer_agent(state):
    # Combine all agent outputs and make rule-based decision
    dd = state.get("due_diligence", {})
    ind = state.get("industry", {})
    cr = state.get("credit_risk", {})
    
    decision = "Reject"
    reasons = []
    # Rule example: Approve if Altman Z > 2.6 and no major risks
    if cr.get("altman_z_score", 0) > 2.6 and not cr.get("major_risks"):
        decision = "Approve"
    else:
        reasons.append("Credit risk too high or major risks identified")
    memo = f"Credit memo for {state['input']['ticker']}:\nDue Diligence: {dd}\nIndustry: {ind}\nCredit Risk: {cr}\nFinal Decision: {decision}"
    return {
        "credit_officer": {
            "memo": memo,
            "decision": decision,
            "reasons": reasons,
        }
    }

# Placeholder functions for credit risk computations
def compute_altman_z(financials):
    """
    Calculate Altman Z-Score for credit risk assessment.
    
    Z-Score = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
    
    Where:
    A = Working Capital / Total Assets
    B = Retained Earnings / Total Assets
    C = EBIT / Total Assets
    D = Market Value of Equity / Total Liabilities
    E = Sales / Total Assets
    
    For simplicity, we'll use available financials:
    - Working Capital ≈ (Cash + Assets - Debt) / Assets (simplified)
    - Retained Earnings ≈ Equity (simplified)
    - EBIT ≈ Revenue * 0.1 (estimated, or use interest_coverage_ratio)
    - Market Value ≈ Equity (simplified)
    - Sales = Revenue
    """
    assets = financials.get('assets', 0)
    equity = financials.get('equity', 0)
    debt = financials.get('debt', 0)
    cash = financials.get('cash', 0)
    revenue = financials.get('revenue', 0)
    interest_coverage = financials.get('interest_coverage_ratio', 0)
    
    if assets == 0:
        return 0.0
    
    # Calculate components (simplified versions)
    # A: Working Capital / Total Assets
    working_capital = cash + assets - debt  # Simplified
    A = working_capital / assets if assets > 0 else 0
    
    # B: Retained Earnings / Total Assets (using equity as proxy)
    B = equity / assets if assets > 0 else 0
    
    # C: EBIT / Total Assets
    # Estimate EBIT from revenue or use interest coverage
    if interest_coverage > 0:
        # If we have interest coverage, estimate EBIT
        # EBIT ≈ Revenue * (some margin estimate)
        ebit = revenue * 0.15  # Rough estimate: 15% margin
    else:
        ebit = revenue * 0.1  # Conservative estimate
    C = ebit / assets if assets > 0 else 0
    
    # D: Market Value of Equity / Total Liabilities (using equity as proxy)
    D = equity / debt if debt > 0 else 0
    
    # E: Sales / Total Assets
    E = revenue / assets if assets > 0 else 0
    
    # Calculate Z-Score
    z_score = 1.2 * A + 1.4 * B + 3.3 * C + 0.6 * D + 1.0 * E
    
    print(f"[Altman Z] Components - A: {A:.3f}, B: {B:.3f}, C: {C:.3f}, D: {D:.3f}, E: {E:.3f}")
    print(f"[Altman Z] Calculated Z-Score: {z_score:.2f}")
    
    return z_score

def identify_major_credit_risks(financials, industry_data):
    # Analyze financials and industry outlook for risks
    return []  # example no risks

# Define graph and flow
graph = StateGraph(State)

graph.add_node("sanitize_input", sanitize_input)
graph.add_node("due_diligence", due_diligence_agent)
graph.add_node("industry", industry_agent)
graph.add_node("credit_risk", credit_risk_agent)
graph.add_node("credit_officer", credit_officer_agent)

graph.set_entry_point("sanitize_input")

graph.add_edge("sanitize_input", "due_diligence")
graph.add_edge("sanitize_input", "credit_risk")

graph.add_edge("due_diligence", "industry")
graph.add_edge("credit_risk", "industry")

# After running due diligence, industry, and credit risk agents, go to credit officer

graph.add_edge("industry", "credit_officer")

graph.set_finish_point("credit_officer")

def create_graph_from_config(config, tavily_tool=None, financials_schema=None):
    """Create graph from configuration.
    
    Args:
        config: ExperimentConfig object or dict with graph configuration
        tavily_tool: Optional custom Tavily tool (uses default if None)
        financials_schema: Optional financials schema from config
    
    Returns:
        Compiled LangGraph application
    """
    if tavily_tool is None:
        tavily_tool = tavily_search_tool
    
    # Get financials schema from config if not provided
    if financials_schema is None:
        from modules.config_loader import get_financials_schema
        if hasattr(config, 'financials_schema'):
            financials_schema = get_financials_schema(config)
        elif isinstance(config, dict):
            financials_schema = config.get("financials_schema")
    
    # Create agents with custom tool and schema
    dd_agent = create_due_diligence_agent(tavily_tool)
    ind_agent = create_industry_agent(tavily_tool)
    cr_agent = create_credit_risk_agent(financials_schema)
    
    # Build graph
    graph = StateGraph(State)
    
    # Get node config
    if isinstance(config, dict):
        nodes_config = config.get("graph", {}).get("nodes", [])
    else:
        nodes_config = config.graph.get("nodes", [])
    
    # Add nodes based on config
    node_map = {
        "sanitize_input": sanitize_input,
        "due_diligence": dd_agent,
        "industry": ind_agent,
        "credit_risk": cr_agent,
        "credit_officer": credit_officer_agent,
    }
    
    for node_config in nodes_config:
        node_name = node_config.get("name")
        if node_config.get("enabled", True) and node_name in node_map:
            graph.add_node(node_name, node_map[node_name])
    
    # Get edges config
    if isinstance(config, dict):
        edges_config = config.get("graph", {}).get("edges", [])
    else:
        edges_config = config.graph.get("edges", [])
    
    # Add edges
    for edge_config in edges_config:
        from_node = edge_config.get("from")
        to_nodes = edge_config.get("to", [])
        if isinstance(to_nodes, str):
            to_nodes = [to_nodes]
        
        for to_node in to_nodes:
            graph.add_edge(from_node, to_node)
    
    # Set entry and finish points
    graph.set_entry_point("sanitize_input")
    graph.set_finish_point("credit_officer")
    
    return graph.compile()

# Compile default graph
app = graph.compile()

# Example usage (only if run directly)
if __name__ == "__main__":
    input_data = {
        "ticker": "IBM"
    }

    result = app.invoke({"input": input_data})

    print(result["credit_officer"]["memo"])
    print("Decision:", result["credit_officer"]["decision"])
