# LangGraph Orchestration System

Multi-agent orchestration system for iterative responsive HTML generation and refinement.

## Overview

This module provides a stateful LangGraph workflow with multiple specialized agents that can generate, evaluate, review, and edit responsive HTML with intelligent routing and memory.

## Architecture

### Agents

1. **Orchestrator Agent**: Routes workflow based on current state
   - Decides next step: GENERATE → EVALUATE → REVIEW_EDIT → EDIT → FINISH
   - Considers iteration count, scores, and state

2. **Generator Agent**: Creates initial HTML from wireframe sketches
   - Wraps ResponsiveGenerator
   - Produces responsive HTML with inline CSS

3. **Evaluator Agent**: Runs comprehensive evaluation metrics
   - Takes screenshots at all viewports
   - Computes IoU-based layout similarity
   - Calculates composite ResponsiveMeter score

4. **Reviewer Agent**: Analyzes issues and provides feedback
   - Identifies worst-performing viewport/component
   - Inspects HTML fragments
   - Produces actionable suggestions

5. **Editor Agent**: Applies targeted HTML edits
   - Reads current HTML fragment
   - Generates improved version
   - Applies changes via modify_html tool

### State Management

`ResponsiveState` extends TypedDict with:
- `messages`: Chat history (for memory)
- `html`: Current HTML content
- `eval_results`: Evaluation metrics per viewport
- `responsive_score`: Composite quality score
- `edit_history`: List of edits applied
- `active_view`: Current focus viewport
- `focus_selector`: CSS selector being edited
- `iteration`: Current iteration number
- `next_step`: Next workflow step

### Tools

All tools are LangChain-compatible and wrap existing ResponsiveGen functionality:

- `generate_html`: Generate from wireframe sketches
- `read_html`: Extract HTML fragments by selector
- `modify_html`: Replace HTML elements
- `take_screenshot`: Render at viewport
- `run_iou_evaluation`: Compute IoU metrics
- `run_perceptual_evaluation`: Perceptual similarity (stub)
- `llm_judge_layout`: LLM-based layout assessment (stub)
- `llm_judge_responsiveness`: Cross-device consistency (stub)
- `compute_responsive_meter`: Aggregate all metrics

## Usage

### Python API

```python
from responsive_gen.orchestration import get_responsive_app
from responsive_gen.io.sketch_loader import SketchLoader
from langchain_core.messages import HumanMessage

# Load sketches
loader = SketchLoader()
triplet = loader.load_triplet(
    "mobile.png", "tablet.png", "desktop.png",
    sample_id="sample_001"
)

# Initialize app
app = get_responsive_app(checkpoint_dir=".checkpoints")

# Prepare initial state
initial_state = {
    "messages": [HumanMessage(content="Generate responsive HTML")],
    "sample_id": "sample_001",
    "sketch_triplet": triplet,
    # ... other fields initialized to None/empty
}

# Run orchestration
config = {"configurable": {"thread_id": "thread_001"}}
result = app.invoke(initial_state, config=config)

print(f"Score: {result['responsive_score']}")
print(f"HTML: {result['html'][:100]}...")
```

### CLI

```bash
python cli.py orchestrate \
  -m examples/sample_001/mobile.png \
  -t examples/sample_001/tablet.png \
  -d examples/sample_001/desktop.png \
  --max-iterations 5 \
  --score-threshold 0.7
```

### Streamlit

Navigate to the "🤖 Orchestration" tab in the Streamlit app.

## Workflow

```
START
  ↓
Orchestrator (routes)
  ↓
┌─────────────────┐
│   GENERATE      │ → Generator → Evaluator
│   EVALUATE      │ → Evaluator
│   REVIEW_EDIT   │ → Reviewer → Editor → Evaluator
│   EDIT          │ → Editor → Evaluator
│   FINISH        │ → END
└─────────────────┘
  ↑                ↓
  └─── Loop ───────┘
```

## Checkpointing

State is persisted using SqliteSaver, enabling:
- Resume interrupted workflows
- Thread-based conversation memory
- Cross-session state persistence

Checkpoints are stored in `.checkpoints/checkpoints.db` by default.

## Error Handling

All agents include error handling:
- Graceful degradation on tool failures
- Fallback logic for LLM parsing errors
- Warning messages for non-critical failures
- State validation before operations

## Future Enhancements

- [ ] Implement CLIP-based perceptual evaluation
- [ ] Complete LLM judge implementations
- [ ] Add ground truth comparison support
- [ ] Parallel viewport evaluation
- [ ] Advanced edit conflict resolution
- [ ] Visual diff visualization

