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
    sanitized = {
        "name": str(input_data.get("name", "")).strip(),
        "industry": str(input_data.get("industry", "")).strip(),
        "revenue": float(input_data.get("revenue", 0)),
        "loan_amount": float(input_data.get("requested_loan_amount", 0)),
        "purpose": str(input_data.get("purpose", "")).strip(),
    }
    return {"input": sanitized}

# Define agents (nodes) - these will use the tool passed to them
def create_due_diligence_agent(tavily_tool):
    def due_diligence_agent(state):
        # Use Tavily for recent news, management info etc.
        company = state["input"]["name"]
        results_news = tavily_tool.invoke(f"{company} recent news")
        results_history = tavily_tool.invoke(f"{company} company history")
        results_management = tavily_tool.invoke(f"{company} management team")
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
        industry = state["input"]["industry"]
        trends = tavily_tool.invoke(f"{industry} industry trends")
        competitive = tavily_tool.invoke(f"{industry} competitive landscape")
        economic = tavily_tool.invoke(f"{industry} economic outlook")
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

def credit_risk_agent(state):
    company = state["input"]["name"]
    # Retrieve financials from AlphaVantage
    # financials = alphavantage_tool.get_financial_report(company)
    financials = {
        "revenue": 5000000,
        "assets": 10000000,
        "equity": 5000000,
        "debt": 5000000,
        "cash": 1000000,
        "interest_coverage_ratio": 10,
    }
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
    memo = f"Credit memo for {state['input']['name']}:\nDue Diligence: {dd}\nIndustry: {ind}\nCredit Risk: {cr}\nFinal Decision: {decision}"
    return {
        "credit_officer": {
            "memo": memo,
            "decision": decision,
            "reasons": reasons,
        }
    }

# Placeholder functions for credit risk computations
def compute_altman_z(financials):
    # Implement proper Altman-Z score calculation here
    return 3.0  # example score

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

def create_graph_from_config(config, tavily_tool=None):
    """Create graph from configuration.
    
    Args:
        config: ExperimentConfig object or dict with graph configuration
        tavily_tool: Optional custom Tavily tool (uses default if None)
    
    Returns:
        Compiled LangGraph application
    """
    if tavily_tool is None:
        tavily_tool = tavily_search_tool
    
    # Create agents with custom tool
    dd_agent = create_due_diligence_agent(tavily_tool)
    ind_agent = create_industry_agent(tavily_tool)
    
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
        "credit_risk": credit_risk_agent,
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
        "name": "Example Corp",
        "industry": "Fintech",
        "revenue": 5000000,
        "requested_loan_amount": 1000000,
        "purpose": "Expansion"
    }

    result = app.invoke({"input": input_data})

    print(result["credit_officer"]["memo"])
    print("Decision:", result["credit_officer"]["decision"])
