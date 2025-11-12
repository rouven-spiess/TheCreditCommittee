import os
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph
# from langchain_community.tools.alpha_vantage.tool import AlphaVantageTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

# Load environment variables
load_dotenv()

# Define state schema
class State(TypedDict):
    input: dict
    due_diligence: dict
    industry: dict
    credit_risk: dict
    credit_officer: dict

# Define tools
# alphavantage_tool = AlphaVantageTool(api_key=os.getenv("ALPHA_VANTAGE_API_KEY"))
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found in environment variables. Please set it in .env file.")
tavily_api_wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_api_key)
tavily_search_tool = TavilySearchResults(api_wrapper=tavily_api_wrapper)

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

# Define agents (nodes)
def due_diligence_agent(state):
    # Use Tavily for recent news, management info etc.
    company = state["input"]["name"]
    results_news = tavily_search_tool.invoke(f"{company} recent news")
    results_history = tavily_search_tool.invoke(f"{company} company history")
    results_management = tavily_search_tool.invoke(f"{company} management team")
    return {
        "due_diligence": {
            "news": results_news,
            "history": results_history,
            "management": results_management,
        }
    }

def industry_agent(state):
    industry = state["input"]["industry"]
    trends = tavily_search_tool.invoke(f"{industry} industry trends")
    competitive = tavily_search_tool.invoke(f"{industry} competitive landscape")
    economic = tavily_search_tool.invoke(f"{industry} economic outlook")
    return {
        "industry": {
            "trends": trends,
            "competitive_landscape": competitive,
            "economic_outlook": economic,
        }
    }

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

# Compile graph into callable application
app = graph.compile()

# Example usage
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
