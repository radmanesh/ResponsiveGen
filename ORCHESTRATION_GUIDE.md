# LangGraph Orchestration Guide

Complete guide to using the multi-agent orchestration system for iterative responsive HTML generation.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

### 2. Set API Key

```bash
export OPENAI_API_KEY=sk-...
# or add to .env file
```

### 3. Run Orchestration

**CLI:**
```bash
python cli.py orchestrate \
  -m examples/sample_001/mobile.png \
  -t examples/sample_001/tablet.png \
  -d examples/sample_001/desktop.png \
  --max-iterations 5 \
  --score-threshold 0.7
```

**Streamlit:**
```bash
streamlit run streamlit_app.py
# Navigate to "🤖 Orchestration" tab
```

## How It Works

### Workflow Steps

1. **Orchestrator** analyzes current state and decides next action
2. **Generator** creates initial HTML from wireframes (first iteration)
3. **Evaluator** runs metrics (IoU, screenshots, composite score)
4. **Reviewer** identifies issues and provides feedback
5. **Editor** applies targeted improvements
6. Loop back to Orchestrator until threshold or max iterations

### State Persistence

Each run uses a **thread_id** for state persistence:
- State saved after each step
- Can resume interrupted workflows
- Conversation history maintained
- Edit history tracked

### Stopping Conditions

Orchestration stops when:
- `responsive_score >= score_threshold` (default: 0.7)
- `iteration >= max_iterations` (default: 5)
- Orchestrator decides `FINISH`

## Configuration

### CLI Options

```bash
python cli.py orchestrate \
  --mobile PATH          # Mobile wireframe (required)
  --tablet PATH          # Tablet wireframe (required)
  --desktop PATH         # Desktop wireframe (required)
  --sample-id ID         # Sample identifier (optional)
  --output DIR           # Output directory (default: outputs)
  --thread-id ID         # Thread ID for persistence (optional)
  --max-iterations N     # Max iterations (default: 5)
  --score-threshold F    # Score to stop (default: 0.7)
  --checkpoint-dir DIR   # Checkpoint directory (default: .checkpoints)
```

### Python API

```python
from responsive_gen.orchestration import get_responsive_app

app = get_responsive_app(
    checkpoint_dir=".checkpoints",
    use_checkpointing=True
)

# Use app.invoke() or app.stream()
```

## Understanding Output

### State Fields

- `html`: Generated HTML content
- `html_path`: Path to saved HTML file
- `responsive_score`: Composite quality score (0.0-1.0)
- `eval_results`: Per-viewport evaluation metrics
- `iteration`: Current iteration number
- `next_step`: Next workflow step
- `active_view`: Viewport currently being focused
- `focus_selector`: CSS selector being edited
- `edit_history`: List of all edits applied

### Metrics

- **IoU Score**: Layout similarity per viewport (0.0-1.0)
- **Responsive Score**: Weighted composite of all metrics
- **Per-Component IoU**: Breakdown by component type

## Troubleshooting

### Import Errors

```bash
# Install missing dependencies
pip install langgraph langgraph-checkpoint langchain-openai
```

### API Key Issues

```bash
# Check if set
echo $OPENAI_API_KEY

# Or use .env file
cat .env | grep OPENAI_API_KEY
```

### Playwright Issues

```bash
# Reinstall browsers
playwright install chromium
```

### State Persistence

Checkpoints are stored in `.checkpoints/checkpoints.db`. To reset:
```bash
rm -rf .checkpoints/
```

## Advanced Usage

### Custom Stopping Conditions

Modify orchestrator logic in `agents.py`:
```python
# In orchestrator_node()
if state.get('responsive_score', 0) < 0.8:  # Custom threshold
    decision = {"next_step": "REVIEW_EDIT"}
```

### Custom Tools

Add new tools in `tools.py`:
```python
@tool
def my_custom_tool(param: str) -> str:
    """Tool description."""
    # Implementation
    return result
```

### Custom Agents

Add new agent nodes in `agents.py`:
```python
def my_agent_node(state: ResponsiveState) -> Dict[str, Any]:
    """Agent description."""
    # Implementation
    return updates
```

Then add to graph in `graph.py`.

## Best Practices

1. **Start with simple wireframes** to test the system
2. **Set reasonable thresholds** (0.7 is a good starting point)
3. **Monitor iteration count** to avoid infinite loops
4. **Use checkpointing** for long-running workflows
5. **Review edit_history** to understand changes made

## Limitations

- LLM judge tools are stubs (return 0.0)
- Perceptual evaluation not yet implemented
- Ground truth comparison optional
- Single-threaded execution (no parallel agents)

## Next Steps

1. Implement CLIP-based perceptual evaluation
2. Complete LLM judge with image encoding
3. Add parallel viewport evaluation
4. Implement visual diff visualization
5. Add edit conflict resolution

