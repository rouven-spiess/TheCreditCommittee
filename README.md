# The Credit Committee

Automated credit decision-making system using LangGraph with multi-agent workflow for loan application analysis.

## Table of Contents

- [General Info](#general-info)
- [Baseline Model](#baseline-model)
- [Screenshots](#screenshots)
- [Technologies](#technologies)
- [Code Examples](#code-examples)
- [Features](#features)
- [Status and Report](#status-and-report)

## General Info

The Credit Committee is an intelligent loan approval system that uses a multi-agent LangGraph workflow to analyze loan applications. The system processes applications through parallel agents that conduct due diligence, assess credit risk, analyze industry trends, and make final approval decisions.

**Key Components:**
- Input sanitization and validation
- Parallel agent execution for efficiency
- Web research integration via Tavily Search API
- Rule-based decision making with Altman-Z score analysis

## Baseline Model

The system uses a **LangGraph StateGraph** with the following workflow:

1. **Input Sanitization** - Validates and cleans input data
2. **Parallel Analysis** - Due Diligence and Credit Risk agents run simultaneously
3. **Industry Analysis** - Runs after parallel agents complete
4. **Final Decision** - Credit Officer combines all analyses and makes decision

**Decision Logic:**
- Approve if Altman-Z score > 2.6 AND no major risks identified
- Reject otherwise

See [GRAPH_VISUALIZATION.md](./GRAPH_VISUALIZATION.md) for detailed workflow diagram.

## Screenshots

_Add screenshots of the system in action here_

## Technologies

- **Python 3.12+**
- **LangGraph** - Graph-based workflow orchestration
- **LangChain Community** - Tool integrations
- **Tavily Search API** - Web research and information gathering
- **python-dotenv** - Environment variable management

## Code Examples

### Basic Usage

```python
from credit_committee import app

# Prepare input data
input_data = {
    "name": "Example Corp",
    "industry": "Fintech",
    "revenue": 5000000,
    "requested_loan_amount": 1000000,
    "purpose": "Expansion"
}

# Run the credit committee workflow
result = app.invoke({"input": input_data})

# Access results
print(result["credit_officer"]["memo"])
print("Decision:", result["credit_officer"]["decision"])
```

### Environment Setup

1. Create a `.env` file:
```bash
TAVILY_API_KEY=your_api_key_here
```

2. Install dependencies:
```bash
uv sync
```

3. Run the application:
```bash
python credit_committee.py
```

## Features

- ✅ Multi-agent parallel processing
- ✅ Automated due diligence research
- ✅ Credit risk assessment with Altman-Z scoring
- ✅ Industry trend analysis
- ✅ Rule-based decision making
- ✅ Comprehensive credit memos
- ✅ State-based workflow management

## Status and Report

**Current Status:** Active Development

**Version:** 0.1.0

**Known Limitations:**
- Financial data currently uses mock data (AlphaVantage integration commented out)
- Altman-Z score calculation is simplified
- Risk identification logic is placeholder

**Future Improvements:**
- [ ] Integrate AlphaVantage API for real financial data
- [ ] Implement full Altman-Z score calculation
- [ ] Add comprehensive risk identification logic
- [ ] Support for multiple decision rules
- [ ] Add logging and monitoring

---

For detailed graph visualization, see [GRAPH_VISUALIZATION.md](./GRAPH_VISUALIZATION.md).

