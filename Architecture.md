# ResponsiveGen Architecture Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [Agent Architecture](#agent-architecture)
3. [State Management](#state-management)
4. [Workflow Graph](#workflow-graph)
5. [Tools & Capabilities](#tools--capabilities)
6. [Evaluation Framework](#evaluation-framework)
7. [Integration Points](#integration-points)
8. [Data Flow](#data-flow)
9. [Technical Stack](#technical-stack)
10. [Error Handling & Resilience](#error-handling--resilience)

---

## System Overview

ResponsiveGen is a multi-agent orchestration system for generating responsive HTML from wireframe sketches. The system uses **LangGraph** to coordinate five specialized agents that work together to iteratively generate, evaluate, review, and refine responsive web pages.

### Core Principles

- **Multi-Agent Collaboration**: Specialized agents handle distinct responsibilities (generation, evaluation, review, editing)
- **Iterative Refinement**: The system loops until quality thresholds are met or max iterations reached
- **Stateful Workflow**: All state is managed through a centralized `ResponsiveState` that persists across iterations
- **Tool-Based Actions**: Agents interact with the system through well-defined LangChain tools
- **Intelligent Routing**: An orchestrator agent makes routing decisions based on current state and quality metrics

### High-Level Flow

```
User Input (Wireframe Sketches)
    ↓
Orchestrator Agent (Routes Workflow)
    ↓
┌─────────────────────────────────────┐
│  Generation Phase                   │
│  Generator Agent → Evaluator Agent  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Refinement Phase (if needed)       │
│  Reviewer Agent → Editor Agent      │
│  → Re-evaluation                    │
└─────────────────────────────────────┘
    ↓
Final HTML Output + Evaluation Metrics
```

---

## Agent Architecture

The system consists of five specialized agents, each with distinct roles and responsibilities.

### 1. Orchestrator Agent

**Purpose**: Central decision-making agent that routes the workflow based on current state.

**Responsibilities**:
- Analyze current state (HTML existence, scores, iteration count)
- Decide next workflow step: `GENERATE`, `EVALUATE`, `REVIEW_EDIT`, `EDIT`, or `FINISH`
- Set focus viewport and CSS selector for targeted editing
- Manage iteration limits and quality thresholds

**Decision Logic**:
```python
if html is None:
    → GENERATE
elif no eval_results or iteration == 0:
    → EVALUATE
elif responsive_score < 0.7 and iteration < 5:
    → REVIEW_EDIT (if no feedback) or EDIT (if feedback exists)
elif responsive_score >= 0.7 or iteration >= 5:
    → FINISH
```

**System Prompt**: Contains decision rules and state context requirements.

**Tools**: None (pure routing logic, though uses LLM for intelligent decision-making)

**LLM Configuration**:
- Provider: Configurable (OpenAI/Anthropic), defaults to `ORCHESTRATOR_PROVIDER` env var
- Model: Defaults to `gpt-4o` (configurable via `ORCHESTRATOR_MODEL`)
- Temperature: Defaults to `0.3` (configurable via `ORCHESTRATOR_TEMPERATURE`)

**Output**: Updates state with `next_step`, `active_view`, and `focus_selector`.

---

### 2. Generator Agent

**Purpose**: Creates initial responsive HTML from wireframe sketch triplets.

**Responsibilities**:
- Load and prepare wireframe sketches (mobile, tablet, desktop)
- Invoke HTML generation using vision-capable LLM
- Parse and extract HTML from LLM response
- Save generated HTML to disk
- Initialize state with HTML content and metadata

**Workflow**:
1. Receive `sample_id` and `sketch_triplet` from state
2. Extract sketch directory path
3. Call `generate_html` tool with sketch triplet
4. Update state with generated HTML and file path
5. Initialize `iteration` to 0 and `edit_history` to empty list

**Tools Used**:
- `generate_html`: Wraps `ResponsiveGenerator` to produce HTML from sketches

**LLM Integration**:
- Uses `ResponsiveGenerator` class which supports OpenAI (GPT-4 Vision) and Anthropic (Claude 3) models
- Sends wireframe images as base64-encoded data URLs
- System prompt emphasizes responsive design, CSS media queries, and semantic HTML

**Output**: Updates state with `html`, `html_path`, `iteration`, and `edit_history`.

---

### 3. Evaluator Agent

**Purpose**: Runs comprehensive evaluation metrics on generated HTML across all viewports.

**Responsibilities**:
- Capture screenshots at mobile, tablet, and desktop viewports
- Compute IoU-based layout similarity metrics per viewport
- Calculate composite ResponsiveMeter score
- Aggregate evaluation results into state

**Workflow**:
1. Verify HTML exists in state
2. For each viewport (mobile, tablet, desktop):
   - Take screenshot using `take_screenshot` tool
   - Run IoU evaluation using `run_iou_evaluation` tool
3. Compute composite score using `compute_responsive_meter` tool
4. Update state with screenshots, eval results, and responsive score
5. Increment iteration counter

**Tools Used**:
- `take_screenshot`: Renders HTML at specific viewport and saves screenshot
- `run_iou_evaluation`: Computes IoU-based layout similarity (with optional ground truth)
- `run_perceptual_evaluation`: Placeholder for CLIP-based perceptual similarity
- `llm_judge_layout`: Placeholder for LLM-based layout assessment
- `llm_judge_responsiveness`: Placeholder for cross-device consistency check
- `compute_responsive_meter`: Aggregates all metrics into composite score

**Metrics Computed**:
- **IoU Score**: Intersection over Union for layout components (text, images, buttons, etc.)
- **Per-Component IoU**: Breakdown by component type
- **Composite Score**: Weighted average of IoU, consistency, LLM judge, and perceptual metrics

**Output**: Updates state with `screenshots`, `eval_results`, `responsive_score`, and `iteration`.

---

### 4. Reviewer Agent

**Purpose**: Analyzes evaluation results to identify issues and provide actionable feedback.

**Responsibilities**:
- Identify worst-performing viewport based on IoU scores
- Inspect HTML fragments using CSS selectors
- Generate structured feedback with specific issues and suggestions
- Set focus viewport and selector for targeted editing

**Workflow**:
1. Verify evaluation results exist in state
2. Find worst-performing viewport (lowest IoU score)
3. Use `read_html` tool to inspect problematic HTML sections
4. Generate feedback using LLM with evaluation context
5. Parse feedback into structured format (view, selector, issues, suggestions)
6. Update state with feedback and focus information
7. Add review entry to edit history

**Tools Used**:
- `read_html`: Extracts HTML fragments by CSS selector for inspection

**LLM Configuration**:
- Component: `"reviewer"`
- Uses same provider/model as orchestrator (configurable)
- System prompt emphasizes issue identification and actionable suggestions

**Feedback Format**:
```json
{
  "view": "mobile" | "tablet" | "desktop",
  "selector": "CSS selector string",
  "issues": ["issue1", "issue2", ...],
  "suggestions": ["suggestion1", "suggestion2", ...]
}
```

**Output**: Updates state with `feedback`, `active_view`, `focus_selector`, and `edit_history`.

---

### 5. Editor Agent

**Purpose**: Applies targeted HTML edits to fix identified issues.

**Responsibilities**:
- Read current HTML fragment using focus selector
- Generate improved HTML fragment addressing feedback
- Apply edit using `modify_html` tool
- Update state with modified HTML
- Track edit history

**Workflow**:
1. Verify HTML, focus_selector, and feedback exist in state
2. Read current HTML fragment using `read_html` tool
3. Generate improved fragment using LLM with feedback context
4. Apply edit using `modify_html` tool
5. Update state with new HTML and edit history
6. Handle edge cases (selector not found, body tag missing, etc.)

**Tools Used**:
- `read_html`: Reads current HTML fragment for editing
- `modify_html`: Replaces HTML element with improved version

**LLM Configuration**:
- Component: `"editor"`
- Uses same provider/model as orchestrator
- System prompt emphasizes maintaining structure, IDs, classes, and responsive behavior

**Error Handling**:
- Fallback strategies for missing selectors (e.g., create body tag if missing)
- Graceful degradation if edit fails
- Preserves HTML structure and semantic elements

**Output**: Updates state with `html`, `edit_target`, and `edit_history`.

---

## State Management

### ResponsiveState Structure

`ResponsiveState` is a `TypedDict` that extends LangGraph's message-based state with custom fields for HTML content, evaluation results, and workflow control.

```python
class ResponsiveState(TypedDict, total=False):
    # Chat memory (LangGraph standard)
    messages: Annotated[List[BaseMessage], add_messages]

    # HTML content
    html: Optional[str]                    # Current HTML content
    html_path: Optional[str]               # Path to saved HTML file

    # Evaluation results
    eval_results: Dict[str, Any]           # Per-viewport evaluation metrics
    responsive_score: Optional[float]      # Composite quality score (0.0-1.0)

    # Edit history
    edit_history: List[Dict[str, Any]]     # List of edits and reviews applied

    # Workflow control
    active_view: Optional[str]             # "mobile" | "tablet" | "desktop"
    focus_selector: Optional[str]          # CSS selector being edited
    iteration: int                         # Current iteration number
    next_step: Optional[str]               # "GENERATE" | "EVALUATE" | "REVIEW_EDIT" | "EDIT" | "FINISH"

    # Sample metadata
    sample_id: Optional[str]               # Unique identifier for this generation
    sketch_triplet: Optional[SketchTriplet] # Original wireframe sketches

    # Screenshot paths
    screenshots: Dict[str, Optional[str]]  # {"mobile": path, "tablet": path, "desktop": path}

    # Feedback and suggestions
    feedback: Optional[Dict[str, Any]]     # Structured feedback from reviewer
    edit_target: Optional[str]             # HTML fragment to replace
```

### State Updates

- **Incremental Updates**: Agents return dictionaries with only changed fields
- **Automatic Merging**: LangGraph merges updates into state automatically
- **Message History**: `messages` field maintains conversation history for context
- **Persistence**: State can be persisted using SqliteSaver checkpointing

### State Validation

The system includes validation utilities:
- `validate_state()`: Checks required fields before operations
- `get_state_summary()`: Generates human-readable state summary
- `merge_state_updates()`: Safely merges updates into current state

---

## Workflow Graph

### Graph Structure

The workflow is implemented as a **LangGraph StateGraph** with the following structure:

```
                    START
                      ↓
              [Orchestrator Node]
                      ↓
         ┌────────────┼────────────┐
         │            │            │
    [GENERATE]  [EVALUATE]  [REVIEW_EDIT]
         │            │            │
         ↓            ↓            ↓
    [Generator]  [Evaluator]  [Reviewer]
         │            │            │
         └──────┬─────┘            │
                │                  │
                ↓                  ↓
          [Evaluator]        [Editor]
                │                  │
                └────────┬─────────┘
                         │
                    [Evaluator]
                         │
                         ↓
                  [Orchestrator]
                         │
                    [FINISH?]
                         │
                         ↓
                       END
```

### Node Definitions

1. **orchestrator**: Entry point, routes to next node based on `next_step`
2. **generator**: Generates initial HTML, always followed by evaluator
3. **evaluator**: Runs metrics, always returns to orchestrator
4. **reviewer**: Analyzes issues, always followed by editor
5. **editor**: Applies edits, always followed by evaluator

### Edge Types

- **Conditional Edges**: From orchestrator based on `next_step` decision
- **Linear Edges**: Fixed sequences (generator→evaluator, reviewer→editor, editor→evaluator)
- **Loop Edges**: Evaluator→orchestrator creates iterative refinement loop

### Routing Function

`route_orchestrator()` maps `next_step` values to node names:
- `"GENERATE"` → `"generator"`
- `"EVALUATE"` → `"evaluator"`
- `"REVIEW_EDIT"` → `"reviewer"`
- `"EDIT"` → `"editor"`
- `"FINISH"` → `END`

### Checkpointing

The graph supports optional checkpointing using `SqliteSaver`:
- **Thread-based Persistence**: Each conversation thread has isolated state
- **Resume Capability**: Interrupted workflows can be resumed
- **Cross-session Memory**: State persists across application restarts
- **Default Location**: `.checkpoints/checkpoints.db`

---

## Tools & Capabilities

All tools are implemented as LangChain `@tool` decorators, making them available to agents through function calling.

### Core Generation Tools

#### `generate_html(sample_id: str, sketch_triplet_path: str) -> Dict[str, str]`

Generates responsive HTML from wireframe sketch triplet.

**Inputs**:
- `sample_id`: Unique identifier for this generation
- `sketch_triplet_path`: Path to directory containing `mobile.png`, `tablet.png`, `desktop.png`

**Process**:
1. Loads sketches using `SketchLoader`
2. Creates `SketchTriplet` object
3. Invokes `ResponsiveGenerator.generate_and_save()`
4. Returns HTML content and file path

**Output**: `{"html": str, "html_path": str, "sample_id": str}`

---

### HTML Manipulation Tools

#### `read_html(html_source: str, selector: Optional[str] = None) -> str`

Reads HTML content, optionally extracting a fragment by CSS selector.

**Inputs**:
- `html_source`: Raw HTML string or path to HTML file
- `selector`: Optional CSS selector (e.g., `"#hero"`, `".nav"`, `"main > section:nth-child(2)"`)

**Process**:
1. Parses HTML using BeautifulSoup
2. If selector provided, extracts matching element
3. Handles fallback strategies for common selectors (`body`, `html`)

**Output**: Full HTML or selected fragment

#### `modify_html(html_source: str, selector: str, new_fragment: str) -> str`

Replaces HTML element matching selector with new fragment.

**Inputs**:
- `html_source`: Raw HTML string or path to HTML file
- `selector`: CSS selector for element to replace
- `new_fragment`: New HTML fragment to insert

**Process**:
1. Parses HTML using BeautifulSoup
2. Finds element by selector
3. Replaces element with parsed new fragment
4. Handles edge cases (missing body tag, etc.)

**Output**: Updated full HTML string

---

### Rendering Tools

#### `take_screenshot(html_source: str, view: str) -> str`

Takes screenshot of HTML at specified viewport.

**Inputs**:
- `html_source`: Raw HTML string or path to HTML file
- `view`: Viewport type - `"mobile"`, `"tablet"`, or `"desktop"`

**Process**:
1. If HTML string, saves to temporary file
2. Maps view to `ViewportConfig` (375px, 768px, 1280px)
3. Uses `HTMLRenderer` with Playwright to render at viewport
4. Saves screenshot to `outputs/screenshots/{view}.png`

**Output**: Path to saved screenshot file

---

### Evaluation Tools

#### `run_iou_evaluation(sample_id: str, html_path: str, view: str, ground_truth_path: Optional[str] = None) -> Dict[str, Any]`

Runs IoU-based layout evaluation for a specific viewport.

**Inputs**:
- `sample_id`: Sample identifier
- `html_path`: Path to generated HTML file
- `view`: Viewport type
- `ground_truth_path`: Optional path to ground truth HTML

**Process**:
1. If ground truth provided, compares layouts using `layout_similarity()`
2. Otherwise, extracts components only (no comparison)
3. Computes IoU scores per component type
4. Returns structured metrics

**Output**: `{"view": str, "iou_score": float, "per_component_iou": dict, "has_ground_truth": bool}`

#### `run_perceptual_evaluation(sample_id: str, html_path: str, view: str) -> Dict[str, Any]`

Runs perceptual similarity evaluation (CLIP-based) - **TODO: Implement**

**Status**: Placeholder returning zero scores

#### `llm_judge_layout(html: str, screenshots: Dict[str, str]) -> float`

LLM-based layout accuracy assessment - **TODO: Implement**

**Status**: Placeholder returning 0.0

#### `llm_judge_responsiveness(html: str, screenshots: Dict[str, str]) -> float`

LLM-based cross-device responsive consistency assessment - **TODO: Implement**

**Status**: Placeholder returning 0.0

#### `compute_responsive_meter(iou_metrics: Dict[str, float], perceptual_metrics: Optional[Dict[str, float]] = None, llm_judge_metrics: Optional[Dict[str, float]] = None) -> float`

Computes composite ResponsiveMeter score from all metrics.

**Inputs**:
- `iou_metrics`: Dictionary with `"mobile"`, `"tablet"`, `"desktop"` IoU scores
- `perceptual_metrics`: Optional perceptual similarity scores
- `llm_judge_metrics`: Optional LLM judge scores

**Process**:
1. Calculates average IoU across viewports
2. Applies default weights:
   - `w1 = 0.35` (IoU)
   - `w2 = 0.25` (Consistency)
   - `w3 = 0.25` (LLM judge)
   - `w4 = 0.15` (Perceptual)
3. Computes weighted composite score

**Output**: Composite score (0.0 to 1.0)

---

## Evaluation Framework

### ResponsiveMeter

The `ResponsiveMeter` class provides a comprehensive evaluation system that combines multiple metrics into a unified score.

**Components**:
1. **IoU Metrics** (`IoUMetrics`): Layout similarity using Intersection over Union
2. **Perceptual Metrics** (`PerceptualMetrics`): CLIP-based visual similarity (TODO)
3. **LLM Judge Scores** (`LLMJudgeScore`): Qualitative assessment (TODO)

**Composite Score Calculation**:
```python
composite_score = (
    w1 * average_iou +
    w2 * cross_device_consistency +
    w3 * llm_judge_overall +
    w4 * average_perceptual_similarity
)
```

**Default Weights**:
- `w1 = 0.35`: IoU weight
- `w2 = 0.25`: Cross-device consistency weight
- `w3 = 0.25`: LLM judge weight
- `w4 = 0.15`: Perceptual similarity weight

### IoU-Based Layout Similarity

**Implementation**: `responsive_gen.evaluation.layout_similarity`

**Process**:
1. Extract visual components from HTML using Playwright
2. Component types: text, image, navigation, button, form, table, divider, card
3. Compute bounding boxes for each component
4. Calculate IoU using Shapely polygons
5. Weighted average across component types

**Metrics**:
- Per-viewport IoU scores (mobile, tablet, desktop)
- Per-component IoU breakdown
- Average IoU across all viewports

### Perceptual Similarity (TODO)

**Planned Implementation**:
- CLIP-based image similarity
- Block-level similarity
- Text region similarity
- Positional alignment metrics

### LLM-as-a-Judge (TODO)

**Planned Rubrics**:
1. **Layout Accuracy**: Component presence, ordering, structural fidelity per device
2. **Visual Hierarchy**: Information architecture and visual flow
3. **Cross-Device Consistency**: Responsive behavior quality
4. **Overall Quality**: Holistic assessment

---

## Integration Points

### Streamlit Web Interface

**File**: `streamlit_app.py`

**Features**:
- **Upload & Generate Tab**: Direct HTML generation from wireframe uploads
- **Results Tab**: View generated HTML and screenshots
- **Evaluation Tab**: View evaluation metrics and scores
- **Orchestration Tab**: Interactive multi-agent workflow with chat interface

**Orchestration Integration**:
```python
# Initialize graph
app = get_responsive_app(checkpoint_dir=".checkpoints", use_checkpointing=True)

# Prepare state
initial_state = {
    "messages": [HumanMessage(content=user_input)],
    "sample_id": sample_id,
    "sketch_triplet": triplet,
    # ... other fields
}

# Run workflow
config = {"configurable": {"thread_id": thread_id}}
result = app.invoke(initial_state, config=config)
```

**State Persistence**: Uses Streamlit session state and LangGraph checkpointing for thread-based conversations.

---

### Command-Line Interface

**File**: `cli.py`

**Commands**:
- `generate`: Generate HTML from sketch triplet
- `render`: Render HTML at all viewports
- `evaluate`: Evaluate generated HTML against ground truth
- `orchestrate`: Run full orchestration workflow (TODO)

**Example**:
```bash
python cli.py generate \
  --mobile examples/sample_001/mobile.png \
  --tablet examples/sample_001/tablet.png \
  --desktop examples/sample_001/desktop.png \
  --output outputs/sample_001
```

---

### Python API

**Direct Usage**:
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
    "html": None,
    "html_path": None,
    "eval_results": {},
    "responsive_score": None,
    "edit_history": [],
    "active_view": None,
    "focus_selector": None,
    "iteration": 0,
    "next_step": None,
    "screenshots": {},
    "feedback": None,
    "edit_target": None,
}

# Run orchestration
config = {"configurable": {"thread_id": "thread_001"}}
result = app.invoke(initial_state, config=config)

print(f"Score: {result['responsive_score']}")
print(f"HTML: {result['html'][:100]}...")
```

---

## Data Flow

### Complete Workflow Data Flow

```
1. User Input
   └─> Wireframe Sketches (mobile.png, tablet.png, desktop.png)
       └─> SketchLoader.load_triplet()
           └─> SketchTriplet object
               └─> ResponsiveState.sketch_triplet

2. Orchestrator Decision
   └─> Analyzes state
       └─> Sets ResponsiveState.next_step = "GENERATE"

3. Generator Agent
   └─> generate_html tool
       └─> ResponsiveGenerator.generate()
           └─> LLM (GPT-4 Vision / Claude 3)
               └─> HTML content
                   └─> ResponsiveState.html
                       └─> Saved to disk: ResponsiveState.html_path

4. Evaluator Agent
   └─> take_screenshot (×3 viewports)
       └─> HTMLRenderer.render_viewport()
           └─> Screenshots → ResponsiveState.screenshots
   └─> run_iou_evaluation (×3 viewports)
       └─> layout_similarity()
           └─> IoU scores → ResponsiveState.eval_results
   └─> compute_responsive_meter()
       └─> Composite score → ResponsiveState.responsive_score

5. Orchestrator Decision (if score < 0.7)
   └─> Sets ResponsiveState.next_step = "REVIEW_EDIT"

6. Reviewer Agent
   └─> read_html (inspect problematic sections)
       └─> LLM analysis
           └─> Feedback → ResponsiveState.feedback
               └─> Sets ResponsiveState.active_view, focus_selector

7. Editor Agent
   └─> read_html (get current fragment)
       └─> LLM generation (improved fragment)
           └─> modify_html (apply edit)
               └─> Updated HTML → ResponsiveState.html
                   └─> Edit history → ResponsiveState.edit_history

8. Loop Back to Step 4 (Re-evaluation)
   └─> Continue until score >= 0.7 or max iterations reached

9. Final Output
   └─> ResponsiveState.html (final HTML)
       └─> ResponsiveState.responsive_score (final score)
           └─> ResponsiveState.eval_results (all metrics)
               └─> ResponsiveState.edit_history (refinement log)
```

### State Transitions

```
Initial State:
{
  html: None,
  iteration: 0,
  next_step: None
}

After Generation:
{
  html: "<!DOCTYPE html>...",
  html_path: "outputs/sample_001/generated.html",
  iteration: 0,
  next_step: "EVALUATE"
}

After Evaluation:
{
  screenshots: {"mobile": "...", "tablet": "...", "desktop": "..."},
  eval_results: {"mobile": {...}, "tablet": {...}, "desktop": {...}},
  responsive_score: 0.65,
  iteration: 1,
  next_step: "REVIEW_EDIT"
}

After Review:
{
  feedback: {"view": "mobile", "selector": "#hero", "issues": [...], "suggestions": [...]},
  active_view: "mobile",
  focus_selector: "#hero",
  next_step: "EDIT"
}

After Edit:
{
  html: "<!DOCTYPE html>...<updated>...",
  edit_history: [..., {"iteration": 1, "type": "edit", "selector": "#hero"}],
  next_step: "EVALUATE"
}

After Re-evaluation:
{
  responsive_score: 0.78,
  iteration: 2,
  next_step: "FINISH"
}
```

---

## Technical Stack

### Core Frameworks

- **LangGraph**: Workflow orchestration and state management
- **LangChain**: LLM integration and tool framework
- **Streamlit**: Web interface
- **Playwright**: HTML rendering and screenshot capture
- **BeautifulSoup**: HTML parsing and manipulation
- **Shapely**: Geometric calculations for IoU metrics
- **Pydantic**: Data validation and models

### LLM Providers

- **OpenAI**: GPT-4 Vision, GPT-4o, GPT-4 Turbo
- **Anthropic**: Claude 3 Opus, Claude 3 Sonnet

### Configuration

**Environment Variables**:
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `ORCHESTRATOR_PROVIDER`: LLM provider for orchestrator (default: "openai")
- `ORCHESTRATOR_MODEL`: Model name for orchestrator (default: "gpt-4o")
- `ORCHESTRATOR_TEMPERATURE`: Temperature for orchestrator (default: 0.3)
- `GENERATOR_PROVIDER`: LLM provider for generator (default: "openai")
- `GENERATOR_MODEL`: Model name for generator (default: "gpt-4o")
- `GENERATOR_TEMPERATURE`: Temperature for generator (default: 0.3)
- `GENERATOR_MAX_TOKENS`: Max tokens for generator (default: 4096)
- `EVALUATOR_PROVIDER`: LLM provider for evaluator/judge (default: "openai")
- `EVALUATOR_MODEL`: Model name for evaluator (default: "gpt-4o")
- `EVALUATOR_TEMPERATURE`: Temperature for evaluator (default: 0.1)
- `LLM_DEBUG_LEVEL`: Logging level ("NONE", "DEBUG", "INFO")
- `LLM_LOG_TO_FILE`: Enable file logging ("true"/"false")
- `LLM_LOG_DIR`: Directory for LLM logs (default: "outputs")

**Streamlit Secrets** (`.streamlit/secrets.toml`):
```toml
[llm_providers]
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "..."

[orchestration]
ORCHESTRATOR_PROVIDER = "openai"
ORCHESTRATOR_MODEL = "gpt-4o"
ORCHESTRATOR_TEMPERATURE = "0.3"

[generation]
GENERATOR_PROVIDER = "openai"
GENERATOR_MODEL = "gpt-4o"
GENERATOR_TEMPERATURE = "0.3"
GENERATOR_MAX_TOKENS = "4096"

[evaluation]
EVALUATOR_PROVIDER = "openai"
EVALUATOR_MODEL = "gpt-4o"
EVALUATOR_TEMPERATURE = "0.1"

[logging]
LLM_DEBUG_LEVEL = "NONE"
LLM_LOG_TO_FILE = "true"
LLM_LOG_DIR = "outputs"
```

---

## Error Handling & Resilience

### Agent-Level Error Handling

**Orchestrator**:
- Fallback to simple rule-based routing if LLM parsing fails
- Validates state before making decisions
- Handles missing fields gracefully

**Generator**:
- Validates sketch files exist before generation
- Handles LLM response parsing errors
- Saves HTML even if metadata extraction fails

**Evaluator**:
- Continues evaluation even if individual viewport fails
- Uses default scores (0.0) for failed evaluations
- Logs warnings for non-critical failures

**Reviewer**:
- Falls back to simple feedback if LLM parsing fails
- Handles missing evaluation results
- Provides default selector ("body") if none found

**Editor**:
- Multiple fallback strategies for missing selectors
- Creates missing HTML elements (e.g., body tag) when needed
- Preserves HTML structure on edit failures
- Returns empty updates if edit cannot be applied

### Tool-Level Error Handling

**HTML Manipulation**:
- Handles missing selectors with fallback strategies
- Creates missing HTML elements when appropriate
- Preserves document structure on errors

**Screenshot Capture**:
- Continues with other viewports if one fails
- Returns `None` for failed screenshots
- Logs errors without crashing workflow

**Evaluation**:
- Returns zero scores for failed evaluations
- Handles missing ground truth gracefully
- Continues with available metrics

### State Validation

- `validate_state()`: Checks required fields before operations
- Type checking via Pydantic models
- Optional fields allow incremental state building

### Logging

**LLM Logger** (`LoggedLLM`):
- Logs all LLM calls to JSONL files
- Tracks token usage and costs
- Component-level logging for debugging
- Configurable log levels and file output

**Log Format**:
```json
{
  "timestamp": "2024-01-01T12:00:00",
  "component": "generator",
  "provider": "openai",
  "model": "gpt-4o",
  "sample_id": "sample_001",
  "messages": [...],
  "response": "...",
  "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
  "metadata": {...}
}
```

---

## Future Enhancements

### Planned Features

1. **Perceptual Evaluation**: Implement CLIP-based visual similarity metrics
2. **LLM Judge**: Complete LLM-as-a-judge implementations for layout accuracy and responsiveness
3. **Ground Truth Support**: Enhanced ground truth comparison in evaluation tools
4. **Parallel Evaluation**: Run viewport evaluations in parallel for speed
5. **Advanced Edit Resolution**: Handle edit conflicts and merge strategies
6. **Visual Diff**: Visualize HTML changes between iterations
7. **Multi-Thread Orchestration**: Support multiple concurrent generation threads
8. **Custom Tool Registration**: Allow users to register custom tools for agents
9. **Workflow Visualization**: Real-time graph visualization in Streamlit
10. **Export Capabilities**: Export workflows as reusable templates

### Known Limitations

- Perceptual and LLM judge evaluations are placeholders
- No support for JavaScript-generated content
- Limited error recovery for malformed HTML
- Sequential evaluation (not parallelized)
- No undo/rollback for edits
- Fixed iteration limit (5) and score threshold (0.7)

---

## Conclusion

ResponsiveGen's multi-agent architecture provides a robust, extensible framework for generating and refining responsive HTML from wireframe sketches. The system's modular design, stateful workflow, and tool-based approach enable iterative improvement while maintaining flexibility for future enhancements.

The orchestration system demonstrates how specialized agents can collaborate effectively through a shared state, with intelligent routing and comprehensive evaluation driving continuous refinement toward high-quality outputs.


