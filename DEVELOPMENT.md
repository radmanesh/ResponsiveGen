# Development Guide

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Poetry (for dependency management)
- API keys for OpenAI or Anthropic

### Initial Setup

```bash
# Clone repository (if from git)
git clone <repository-url>
cd ResponsiveGen

# Install dependencies
make install

# Configure environment
cp .env.example .env
# Edit .env and add your API keys

# Install Playwright browsers
make playwright
```

## Project Architecture

### Core Components

1. **Models Layer** (`responsive_gen/models.py`)
   - Pydantic models for type safety
   - Data validation and serialization
   - Business logic (IoU calculation, score computation)

2. **I/O Layer** (`responsive_gen/io/`)
   - Sketch loading and validation
   - Artifact persistence
   - Image processing utilities

3. **Pipeline Layer** (`responsive_gen/pipeline/`)
   - LLM orchestration with LangChain
   - Prompt engineering
   - Response parsing

4. **Rendering Layer** (`responsive_gen/rendering/`)
   - Browser automation with Playwright
   - Screenshot capture
   - Visual comparison tools

5. **Evaluation Layer** (`responsive_gen/evaluation/`)
   - Metric calculation (IoU, perceptual, LLM judge)
   - Composite scoring
   - Result aggregation

### Design Patterns

- **Separation of Concerns**: Each module has a clear responsibility
- **Dependency Injection**: Pass dependencies explicitly (e.g., SketchLoader to Generator)
- **Configuration via Environment**: API keys and settings from `.env`
- **Type Safety**: Pydantic models ensure data validity
- **Async/Sync Wrappers**: Playwright async operations wrapped for ease of use

## Development Workflow

### Running the Application

```bash
# Web interface (recommended for development)
make run-app

# CLI interface
python cli.py generate --help
```

### Code Quality

```bash
# Format code with black and ruff
make format

# Lint code
make lint

# Run tests (when implemented)
make test
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest --cov=responsive_gen tests/
```

## Implementing Evaluation Metrics

The evaluation framework is scaffolded but requires implementation. Here's how to proceed:

### 1. Component Detection

**File**: `responsive_gen/evaluation/layout_metrics.py`
**Class**: `ComponentDetector`

```python
def detect_components(self, screenshot_path, device_type):
    # TODO: Implement using:
    # - YOLO/Faster R-CNN for object detection
    # - Fine-tuned on webpage elements
    # - OCR (Tesseract/EasyOCR) for text
    # - Custom heuristics for UI patterns

    # Example approach:
    # 1. Load image
    # 2. Run detection model
    # 3. Post-process bounding boxes
    # 4. Classify component types
    # 5. Return LayoutAnalysis
    pass
```

**Resources**:
- Train on WebUI dataset or similar
- Use transfer learning from COCO
- Combine with rule-based heuristics

### 2. IoU Calculation

**File**: `responsive_gen/evaluation/layout_metrics.py`
**Class**: `IoUCalculator`

```python
def calculate_component_iou(self, generated_components, ground_truth_components):
    # TODO: Implement using:
    # - Hungarian algorithm (scipy.optimize.linear_sum_assignment)
    # - Match generated to ground truth components
    # - Calculate IoU for each match
    # - Average across all matches

    # Example:
    # 1. Build cost matrix (1 - IoU for all pairs)
    # 2. Find optimal matching
    # 3. Calculate average IoU of matched pairs
    # 4. Penalize unmatched components
    pass
```

### 3. Perceptual Similarity

**File**: `responsive_gen/evaluation/layout_metrics.py`
**Class**: `PerceptualSimilarity`

```python
def calculate_clip_similarity(self, generated_screenshot, ground_truth_screenshot):
    # TODO: Implement using:
    # - Load CLIP model (openai/clip-vit-base-patch32)
    # - Extract embeddings for both images
    # - Calculate cosine similarity

    # Example:
    # from transformers import CLIPProcessor, CLIPModel
    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # ... encode and compare
    pass
```

### 4. LLM Judge

**File**: `responsive_gen/evaluation/llm_judge.py`
**Class**: `LLMJudge`

```python
def evaluate_layout_accuracy(self, wireframe_path, screenshot_path, device_type):
    # TODO: Implement using:
    # - Load and encode images as base64
    # - Construct multi-modal prompt
    # - Call LLM with images
    # - Parse JSON response

    # Example:
    # 1. Load images
    # 2. Create prompt with template
    # 3. messages = [system, HumanMessage(content=[text, image1, image2])]
    # 4. response = self.llm.invoke(messages)
    # 5. Parse JSON and extract score + reasoning
    pass
```

## Adding New Features

### Adding a New Metric

1. Define model in `responsive_gen/models.py`:
```python
class MyNewMetric(BaseModel):
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

2. Implement calculator in `responsive_gen/evaluation/`:
```python
class MyMetricCalculator:
    def calculate(self, inputs) -> MyNewMetric:
        # Implementation
        pass
```

3. Integrate into `ResponsiveMeter`:
```python
# In responsive_meter.py
def evaluate(self, ...):
    my_metric = MyMetricCalculator().calculate(...)
    # Add to composite score
```

### Adding a New LLM Provider

1. Add to `responsive_gen/pipeline/generation.py`:
```python
elif self.provider == "my_provider":
    from langchain_my_provider import ChatMyProvider
    self.llm = ChatMyProvider(...)
```

2. Update prompt construction for provider-specific format

3. Add to CLI and Streamlit dropdown options

## Debugging Tips

### Generation Issues

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Inspect prompts before sending
print(generator._create_prompt(triplet_data))

# Save intermediate outputs
html_path.write_text(response.content)
```

### Rendering Issues

```python
# Run browser in headed mode to see what's happening
renderer = HTMLRenderer(headless=False)

# Increase wait time if content loads slowly
renderer.render_viewport(..., wait_time=5000)

# Check console errors
# Add page.on("console", lambda msg: print(msg.text))
```

### API Errors

```python
# Check environment variables
import os
print(os.getenv("OPENAI_API_KEY"))

# Test API connection
from langchain_openai import ChatOpenAI
llm = ChatOpenAI()
llm.invoke("test")

# Monitor rate limits and costs
print(f"Tokens: {generated.prompt_tokens + generated.completion_tokens}")
```

## Performance Optimization

### Generation Speed

- Use faster models (gpt-4o vs gpt-4-vision-preview)
- Reduce max_tokens if HTML is consistently shorter
- Cache sketch embeddings if generating multiple times

### Rendering Speed

- Render viewports in parallel (use asyncio.gather)
- Reduce wait_time for simple pages
- Use smaller viewport heights for faster screenshots

### Evaluation Speed

- Batch component detection across multiple samples
- Cache CLIP embeddings
- Parallelize per-device evaluations

## Contributing

### Code Style

- Follow PEP 8
- Use type hints everywhere
- Document all public functions
- Keep functions focused and small

### Commit Messages

```
type(scope): brief description

Longer explanation if needed

- Bullet points for details
```

Types: feat, fix, docs, style, refactor, test, chore

### Pull Request Process

1. Create feature branch
2. Implement changes with tests
3. Run `make format && make lint`
4. Update documentation
5. Submit PR with clear description

## Resources

### Documentation

- [LangChain Docs](https://python.langchain.com/docs/)
- [Playwright Python](https://playwright.dev/python/docs/intro)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Pydantic Docs](https://docs.pydantic.dev/)

### Related Papers

- Sketch2Code (Li et al., 2024)
- Design2Code
- Screenshot-to-Code projects

### Datasets

- WebUI dataset for component detection
- Rico dataset (mobile UI)
- Common Crawl for webpage examples

---

**Happy coding!** 🚀

For questions or issues, see README.md or open an issue.

