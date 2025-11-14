from typing import TypedDict, Optional
from langgraph.graph import StateGraph
from modules.tools import tavily_search_tool, alphavantage_tool, create_tavily_tool, create_openai_synthesis_tool

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
def create_due_diligence_agent(tavily_tool, synthesis_tool=None):
    def due_diligence_agent(state):
        # Use Tavily for recent news, management info etc.
        ticker = state["input"]["ticker"]
        
        # Search for information
        raw_news = tavily_tool.invoke(f"{ticker} stock recent news")
        raw_history = tavily_tool.invoke(f"{ticker} company history")
        raw_management = tavily_tool.invoke(f"{ticker} company management team")
        
        # Synthesize results if synthesis tool is available
        if synthesis_tool:
            try:
                news = synthesis_tool.invoke(f"{ticker} stock recent news", raw_news)
                history = synthesis_tool.invoke(f"{ticker} company history", raw_history)
                management = synthesis_tool.invoke(f"{ticker} company management team", raw_management)
            except Exception as e:
                print(f"Warning: Synthesis failed, using raw results: {e}")
                news = raw_news.get("answer", str(raw_news)) if isinstance(raw_news, dict) else str(raw_news)
                history = raw_history.get("answer", str(raw_history)) if isinstance(raw_history, dict) else str(raw_history)
                management = raw_management.get("answer", str(raw_management)) if isinstance(raw_management, dict) else str(raw_management)
        else:
            # Use raw results or extract answer if available
            news = raw_news.get("answer", raw_news) if isinstance(raw_news, dict) else raw_news
            history = raw_history.get("answer", raw_history) if isinstance(raw_history, dict) else raw_history
            management = raw_management.get("answer", raw_management) if isinstance(raw_management, dict) else raw_management
        
        return {
            "due_diligence": {
                "news": news,
                "history": history,
                "management": management,
            }
        }
    return due_diligence_agent

def create_industry_agent(tavily_tool, synthesis_tool=None):
    def industry_agent(state):
        ticker = state["input"]["ticker"]
        # Derive industry information from ticker/company
        raw_trends = tavily_tool.invoke(f"{ticker} stock industry trends")
        raw_competitive = tavily_tool.invoke(f"{ticker} company competitive landscape")
        raw_economic = tavily_tool.invoke(f"{ticker} company economic outlook")
        
        # Synthesize results if synthesis tool is available
        if synthesis_tool:
            try:
                trends = synthesis_tool.invoke(f"{ticker} stock industry trends", raw_trends)
                competitive = synthesis_tool.invoke(f"{ticker} company competitive landscape", raw_competitive)
                economic = synthesis_tool.invoke(f"{ticker} company economic outlook", raw_economic)
            except Exception as e:
                print(f"Warning: Synthesis failed, using raw results: {e}")
                trends = raw_trends.get("answer", str(raw_trends)) if isinstance(raw_trends, dict) else str(raw_trends)
                competitive = raw_competitive.get("answer", str(raw_competitive)) if isinstance(raw_competitive, dict) else str(raw_competitive)
                economic = raw_economic.get("answer", str(raw_economic)) if isinstance(raw_economic, dict) else str(raw_economic)
        else:
            # Use raw results or extract answer if available
            trends = raw_trends.get("answer", raw_trends) if isinstance(raw_trends, dict) else raw_trends
            competitive = raw_competitive.get("answer", raw_competitive) if isinstance(raw_competitive, dict) else raw_competitive
            economic = raw_economic.get("answer", raw_economic) if isinstance(raw_economic, dict) else raw_economic
        
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

def create_credit_officer_agent(synthesis_tool=None):
    """Create credit officer agent with optional reasoning capability.
    
    Args:
        synthesis_tool: Optional OpenAI synthesis tool for reasoning about all information
    """
    def credit_officer_agent(state):
        """Generate a comprehensive markdown credit memo based on all agent analyses."""
        from datetime import datetime
        
        ticker = state['input']['ticker']
        dd = state.get("due_diligence", {})
        ind = state.get("industry", {})
        cr = state.get("credit_risk", {})
        
        financials = cr.get("financials", {})
        altman_z = cr.get("altman_z_score", 0)
        major_risks = cr.get("major_risks", [])
        
        # Decision logic - use OpenAI reasoning if available, otherwise use rule-based
        decision = "Reject"
        reasons = []
        recommendation = ""
        reasoning_summary = ""
        
        if synthesis_tool:
            # Use OpenAI to reason about all available information
            try:
                # Prepare comprehensive context for reasoning
                context = f"""Credit Risk Assessment for {ticker}

Financial Analysis:
- Altman Z-Score: {altman_z:.2f} ({'Above' if altman_z > 2.6 else 'Below'} threshold of 2.6)
- Revenue: ${financials.get('revenue', 0):,.0f}
- Assets: ${financials.get('assets', 0):,.0f}
- Equity: ${financials.get('equity', 0):,.0f}
- Debt: ${financials.get('debt', 0):,.0f}
- Cash: ${financials.get('cash', 0):,.0f}
- Interest Coverage Ratio: {financials.get('interest_coverage_ratio', 0):.2f}x
- Major Risks Identified: {major_risks if major_risks else 'None'}

Due Diligence Findings:
- Recent News: {str(dd.get('news', 'N/A'))[:500]}
- Company History: {str(dd.get('history', 'N/A'))[:500]}
- Management: {str(dd.get('management', 'N/A'))[:500]}

Industry Analysis:
- Trends: {str(ind.get('trends', 'N/A'))[:500]}
- Competitive Landscape: {str(ind.get('competitive_landscape', 'N/A'))[:500]}
- Economic Outlook: {str(ind.get('economic_outlook', 'N/A'))[:500]}

Please analyze all the above information and provide:
1. A recommendation (Approve or Reject)
2. Key reasons for the decision
3. A brief summary of your reasoning (2-3 sentences)

Consider:
- The Altman Z-Score is a key indicator but not the only factor
- Due diligence findings may reveal risks or opportunities not captured in financials
- Industry trends and competitive position affect long-term viability
- Economic outlook impacts future performance
- Balance quantitative metrics with qualitative insights"""
                
                reasoning_result = synthesis_tool.invoke(
                    f"Credit risk assessment for {ticker}",
                    {"context": context},
                    synthesis_prompt="""You are a senior credit officer making a loan decision. 
Analyze all provided information comprehensively. Consider financial metrics, due diligence findings, 
industry context, and risk factors. Provide a clear recommendation (Approve or Reject) with 
well-reasoned justification. Be objective and consider both quantitative and qualitative factors."""
                )
                
                reasoning_summary = reasoning_result
                
                # Parse the reasoning result to extract decision
                reasoning_lower = reasoning_result.lower()
                if "recommendation: approve" in reasoning_lower or "recommend: approve" in reasoning_lower or ("approve" in reasoning_lower and "reject" not in reasoning_lower[:100]):
                    decision = "Approve"
                    recommendation = f"**RECOMMENDATION: APPROVE** - {reasoning_result[:200]}"
                else:
                    decision = "Reject"
                    recommendation = f"**RECOMMENDATION: REJECT** - {reasoning_result[:200]}"
                
                # Extract reasons from reasoning
                if "reason" in reasoning_lower or "because" in reasoning_lower:
                    # Try to extract structured reasons
                    lines = reasoning_result.split('\n')
                    for line in lines:
                        if any(keyword in line.lower() for keyword in ['risk', 'concern', 'issue', 'negative', 'weak']):
                            reasons.append(line.strip())
                
            except Exception as e:
                print(f"Warning: OpenAI reasoning failed, falling back to rule-based decision: {e}")
                # Fall back to rule-based logic
                if altman_z > 2.6 and not major_risks:
                    decision = "Approve"
                    recommendation = "**RECOMMENDATION: APPROVE** - The applicant demonstrates strong financial health with a Z-score above the threshold and no major risks identified."
                else:
                    if altman_z <= 2.6:
                        reasons.append(f"Altman Z-Score ({altman_z:.2f}) is below the threshold of 2.6, indicating elevated credit risk")
                    if major_risks:
                        reasons.append(f"Major risks identified: {', '.join(str(r) for r in major_risks)}")
                    recommendation = "**RECOMMENDATION: REJECT** - Credit risk assessment indicates the application does not meet approval criteria."
        else:
            # Rule-based decision logic (fallback)
            if altman_z > 2.6 and not major_risks:
                decision = "Approve"
                recommendation = "**RECOMMENDATION: APPROVE** - The applicant demonstrates strong financial health with a Z-score above the threshold and no major risks identified."
            else:
                if altman_z <= 2.6:
                    reasons.append(f"Altman Z-Score ({altman_z:.2f}) is below the threshold of 2.6, indicating elevated credit risk")
                if major_risks:
                    reasons.append(f"Major risks identified: {', '.join(str(r) for r in major_risks)}")
                recommendation = "**RECOMMENDATION: REJECT** - Credit risk assessment indicates the application does not meet approval criteria."
        
        # Format financials for display
        def format_currency(value):
            """Format large numbers as currency."""
            if value >= 1_000_000_000:
                return f"${value/1_000_000_000:.2f}B"
            elif value >= 1_000_000:
                return f"${value/1_000_000:.2f}M"
            else:
                return f"${value:,.0f}"
        
        # Generate markdown memo
        memo = f"""# Credit Memo

**Ticker Symbol:** {ticker}  
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Status:** {decision}

---

## Executive Summary

{recommendation}

**Altman Z-Score:** {altman_z:.2f}  
**Risk Level:** {'Low' if altman_z > 2.6 else 'High' if altman_z < 1.8 else 'Moderate'}

---

## Financial Analysis

### Key Financial Metrics

| Metric | Value |
|--------|-------|
| Revenue | {format_currency(financials.get('revenue', 0))} |
| Total Assets | {format_currency(financials.get('assets', 0))} |
| Shareholders' Equity | {format_currency(financials.get('equity', 0))} |
| Total Debt | {format_currency(financials.get('debt', 0))} |
| Cash & Equivalents | {format_currency(financials.get('cash', 0))} |
| Interest Coverage Ratio | {financials.get('interest_coverage_ratio', 0):.2f}x |

### Altman Z-Score Analysis

The Altman Z-Score is a financial model used to predict the probability of bankruptcy. Scores above 2.6 indicate financial stability, while scores below 1.8 suggest financial distress.

**Calculated Z-Score:** {altman_z:.2f}

**Interpretation:**
"""
        
        if altman_z > 2.6:
            memo += "- ✅ **Safe Zone**: Strong financial position, low bankruptcy risk\n"
        elif altman_z > 1.8:
            memo += "- ⚠️ **Grey Zone**: Moderate financial risk, requires careful monitoring\n"
        else:
            memo += "- ❌ **Distress Zone**: High bankruptcy risk, significant financial distress\n"
        
        memo += f"""
---

## Due Diligence Findings

### Recent News & Events
"""
        
        # Extract and format due diligence information
        # Handle both synthesized (string) and raw (dict) results
        dd_news = dd.get("news", {})
        if isinstance(dd_news, str):
            memo += f"{dd_news}\n\n"
        elif isinstance(dd_news, dict) and "answer" in dd_news:
            memo += f"{dd_news.get('answer', 'No recent news information available.')}\n\n"
        elif dd_news:
            memo += f"{str(dd_news)[:500]}...\n\n"
        else:
            memo += "No recent news information available.\n\n"
        
        memo += """### Company History
"""
        dd_history = dd.get("history", {})
        if isinstance(dd_history, str):
            memo += f"{dd_history}\n\n"
        elif isinstance(dd_history, dict) and "answer" in dd_history:
            memo += f"{dd_history.get('answer', 'No company history information available.')}\n\n"
        elif dd_history:
            memo += f"{str(dd_history)[:500]}...\n\n"
        else:
            memo += "No company history information available.\n\n"
        
        memo += """### Management Team
"""
        dd_management = dd.get("management", {})
        if isinstance(dd_management, str):
            memo += f"{dd_management}\n\n"
        elif isinstance(dd_management, dict) and "answer" in dd_management:
            memo += f"{dd_management.get('answer', 'No management information available.')}\n\n"
        elif dd_management:
            memo += f"{str(dd_management)[:500]}...\n\n"
        else:
            memo += "No management information available.\n\n"
        
        memo += """---

## Industry Analysis

### Industry Trends
"""
        ind_trends = ind.get("trends", {})
        if isinstance(ind_trends, str):
            memo += f"{ind_trends}\n\n"
        elif isinstance(ind_trends, dict) and "answer" in ind_trends:
            memo += f"{ind_trends.get('answer', 'No industry trends information available.')}\n\n"
        elif ind_trends:
            memo += f"{str(ind_trends)[:500]}...\n\n"
        else:
            memo += "No industry trends information available.\n\n"
        
        memo += """### Competitive Landscape
"""
        ind_competitive = ind.get("competitive_landscape", {})
        if isinstance(ind_competitive, str):
            memo += f"{ind_competitive}\n\n"
        elif isinstance(ind_competitive, dict) and "answer" in ind_competitive:
            memo += f"{ind_competitive.get('answer', 'No competitive landscape information available.')}\n\n"
        elif ind_competitive:
            memo += f"{str(ind_competitive)[:500]}...\n\n"
        else:
            memo += "No competitive landscape information available.\n\n"
        
        memo += """### Economic Outlook
"""
        ind_economic = ind.get("economic_outlook", {})
        if isinstance(ind_economic, str):
            memo += f"{ind_economic}\n\n"
        elif isinstance(ind_economic, dict) and "answer" in ind_economic:
            memo += f"{ind_economic.get('answer', 'No economic outlook information available.')}\n\n"
        elif ind_economic:
            memo += f"{str(ind_economic)[:500]}...\n\n"
        else:
            memo += "No economic outlook information available.\n\n"
        
        memo += """---

## Risk Assessment

### Identified Risks
"""
        
        if major_risks:
            for i, risk in enumerate(major_risks, 1):
                memo += f"{i}. {risk}\n"
        else:
            memo += "No major risks identified in the analysis.\n"
        
        memo += f"""
### Risk Factors Considered

- **Financial Stability**: Based on Altman Z-Score analysis
- **Market Position**: Industry and competitive analysis
- **Operational History**: Due diligence findings
- **Economic Environment**: Industry economic outlook

---

## Final Decision

**Decision:** {decision}

**Rationale:**
"""
        
        if reasons:
            for i, reason in enumerate(reasons, 1):
                memo += f"{i}. {reason}\n"
        else:
            memo += "All criteria met for approval.\n"
        
        memo += f"""

---

## Decision Reasoning
"""
        
        if reasoning_summary:
            memo += f"""
{reasoning_summary}

"""
        else:
            memo += f"""
Based on comprehensive analysis of financial metrics, due diligence findings, industry context, and risk assessment, the recommendation is to **{decision}** this application.

**Key Factors:**
- Altman Z-Score: {altman_z:.2f} ({'Above' if altman_z > 2.6 else 'Below'} threshold of 2.6)
- Major Risks: {'Present' if major_risks else 'None identified'}
- Financial Position: {'Strong' if financials.get('equity', 0) > financials.get('debt', 0) else 'Moderate' if financials.get('equity', 0) > 0 else 'Weak'}

"""
        
        memo += """---

## Conclusion

The decision is based on a comprehensive evaluation of:
- Financial health metrics (Altman Z-Score and key ratios)
- Due diligence findings (news, history, management)
- Industry analysis (trends, competition, economic outlook)
- Risk assessment

---

*This memo was generated automatically by the Credit Committee AI system.*
"""
        
        return {
            "credit_officer": {
                "memo": memo,
                "decision": decision,
                "reasons": reasons,
                "recommendation": recommendation,
                "reasoning": reasoning_summary if reasoning_summary else None,
            }
        }
    return credit_officer_agent

# Default credit officer agent (for backward compatibility)
credit_officer_agent = create_credit_officer_agent()

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
# Note: credit_officer_agent will be set in create_graph_from_config
# For default graph, use the default agent
default_credit_officer = create_credit_officer_agent()
graph.add_node("credit_officer", default_credit_officer)

graph.set_entry_point("sanitize_input")

graph.add_edge("sanitize_input", "due_diligence")
graph.add_edge("sanitize_input", "credit_risk")

graph.add_edge("due_diligence", "industry")
graph.add_edge("credit_risk", "industry")

# After running due diligence, industry, and credit risk agents, go to credit officer

graph.add_edge("industry", "credit_officer")

graph.set_finish_point("credit_officer")

def create_graph_from_config(config, tavily_tool=None, financials_schema=None, synthesis_tool=None):
    """Create graph from configuration.
    
    Args:
        config: ExperimentConfig object or dict with graph configuration
        tavily_tool: Optional custom Tavily tool (uses default if None)
        financials_schema: Optional financials schema from config
        synthesis_tool: Optional OpenAI synthesis tool for summarizing search results
    
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
    
    # Create synthesis tool if not provided but enabled in config
    if synthesis_tool is None:
        # Check if OpenAI synthesis is enabled in config
        if isinstance(config, dict):
            openai_config = config.get("tools", {}).get("openai", {})
        else:
            openai_config = getattr(config, 'tools', {}).get("openai", {}) if hasattr(config, 'tools') else {}
        
        if openai_config.get("enabled", False):
            try:
                model = openai_config.get("model", "gpt-4o-mini")
                temperature = openai_config.get("temperature", 0.3)
                synthesis_tool = create_openai_synthesis_tool(model=model, temperature=temperature)
                print(f"[Config] OpenAI synthesis enabled (model: {model})")
            except Exception as e:
                print(f"Warning: Could not create OpenAI synthesis tool: {e}")
                synthesis_tool = None
    
    # Create agents with custom tools and schema
    dd_agent = create_due_diligence_agent(tavily_tool, synthesis_tool)
    ind_agent = create_industry_agent(tavily_tool, synthesis_tool)
    cr_agent = create_credit_risk_agent(financials_schema)
    co_agent = create_credit_officer_agent(synthesis_tool)
    
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
        "credit_officer": co_agent,
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
