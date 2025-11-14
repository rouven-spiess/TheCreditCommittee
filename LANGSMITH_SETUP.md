# LangSmith Integration Guide

LangSmith can be integrated **without any code changes** by simply setting environment variables. LangChain/LangGraph automatically detects and uses LangSmith when these variables are set.

## Quick Setup

### 1. Get Your LangSmith API Key

1. Sign up at https://smith.langchain.com (free tier available)
2. Go to Settings → API Keys
3. Create a new API key

### 2. Add to `.env` File

Add these environment variables to your `.env` file:

```bash
# LangSmith Configuration
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=credit-committee
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### 3. That's It!

No code changes needed. LangChain/LangGraph will automatically:
- ✅ Track all LangGraph executions
- ✅ Log tool calls (Tavily, AlphaVantage)
- ✅ Record agent state transitions
- ✅ Capture inputs/outputs
- ✅ Track latency and errors

## Environment Variables Explained

| Variable | Description | Required |
|----------|-------------|----------|
| `LANGSMITH_API_KEY` | Your LangSmith API key | Yes |
| `LANGCHAIN_TRACING_V2` | Enable tracing (set to `true`) | Yes |
| `LANGCHAIN_PROJECT` | Project name in LangSmith dashboard | Optional (defaults to "default") |
| `LANGCHAIN_ENDPOINT` | LangSmith API endpoint | Optional (defaults to cloud) |

## What Gets Tracked

Once configured, LangSmith automatically tracks:

- **Graph Executions**: Full workflow runs
- **Node Executions**: Each agent/node execution
- **Tool Calls**: Tavily searches, AlphaVantage API calls
- **State Transitions**: State changes between nodes
- **Errors**: Any exceptions or failures
- **Latency**: Execution times for each step
- **Inputs/Outputs**: Full state data at each step

## Viewing Traces

1. Go to https://smith.langchain.com
2. Navigate to your project (default: "credit-committee")
3. View traces, filter by date, search by input/output
4. Compare different runs
5. Debug failed executions

## Advanced Configuration

### Custom Project Names per Experiment

You can set the project dynamically:

```bash
# In your .env or when running
LANGCHAIN_PROJECT=credit-committee-exp-001
```

### Disable Tracing (for Testing)

```bash
LANGCHAIN_TRACING_V2=false
```

### Local LangSmith (Self-Hosted)

If using self-hosted LangSmith:

```bash
LANGCHAIN_ENDPOINT=http://your-langsmith-server:8000
```

## Integration with Experiment Tracking

LangSmith works alongside W&B:
- **W&B**: High-level experiment metrics, configs, comparisons
- **LangSmith**: Detailed execution traces, debugging, observability

Both can run simultaneously without conflicts.

## Example `.env` File

```bash
# API Keys
TAVILY_API_KEY=your_tavily_key
ALPHAVANTAGE_API_KEY=your_alphavantage_key

# LangSmith (Optional but Recommended)
LANGSMITH_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=credit-committee
```

## Verification

After setting up, run your code:

```bash
python credit_committee.py
```

Then check https://smith.langchain.com - you should see traces appearing automatically!



