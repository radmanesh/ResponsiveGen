# Responsive Web Generation from Wireframe Sketch Triplets

This project extends the *Sketch2Code* framework to a new benchmark task: **responsive webpage generation from multi-viewport sketches**. It investigates whether Large Language Models (LLMs) can synthesize **one responsive webpage** from **three related sketches**—mobile, tablet, and desktop.

## Overview

The pipeline:
1. **Ingests** three wireframe sketches of a single webpage (mobile 375×688, tablet 768×1024, desktop 1280×800)
2. **Generates** a single static HTML document with inline CSS that renders responsively across all three viewports
3. **Evaluates** the generated page using IoU-based layout similarity, perceptual metrics, and LLM-as-a-judge rubrics

## Project Structure

```
ResponsiveGen/
├── responsive_gen/          # Main package
│   ├── io/                  # Sketch loading utilities
│   ├── models.py           # Data models and schemas
│   ├── pipeline/           # Generation pipeline
│   ├── rendering/          # HTML rendering & screenshot capture
│   └── evaluation/         # Metrics and judging
├── streamlit_app.py        # Web interface
├── cli.py                  # Command-line interface
├── pyproject.toml          # Dependencies (Poetry)
├── .env.example            # Environment variable template
└── Makefile                # Common commands
```

## Installation

### Prerequisites
- Python 3.10+
- Mamba/Conda, Poetry, or pip

### Setup

**Option 1: Using Mamba/Conda**

```bash
# Create and activate environment
mamba create -n responsive python=3.10
mamba activate responsive

# Install dependencies
pip install -r requirements.txt
playwright install chromium

# Copy environment template and configure
cp .env.example .env
# Edit .env with your API keys (OpenAI, Anthropic, etc.)
```

**Option 2: Using Poetry**

```bash
# Install dependencies
make install

# Copy environment template and configure
cp .env.example .env
# Edit .env with your API keys (OpenAI, Anthropic, etc.)
```

**Option 3: Using pip + venv**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
playwright install chromium

# Copy environment template and configure
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### Streamlit Web Interface (Recommended)

```bash
make run-app
```

Navigate to `http://localhost:8501` and:
1. Upload three wireframe sketches (mobile, tablet, desktop)
2. Configure LLM provider and generation parameters
3. Click "Generate Responsive HTML"
4. View generated HTML, screenshots, and evaluation metrics

### Command-Line Interface

```bash
# Generate responsive HTML from sketch triplet
python cli.py generate \
  --mobile path/to/mobile.png \
  --tablet path/to/tablet.png \
  --desktop path/to/desktop.png \
  --output outputs/sample_001

# Render existing HTML at all viewports
python cli.py render \
  --html outputs/sample_001/generated.html \
  --output outputs/sample_001/screenshots

# Evaluate generated HTML against ground truth
python cli.py evaluate \
  --generated outputs/sample_001/generated.html \
  --ground-truth path/to/ground_truth/ \
  --output outputs/sample_001/metrics.json
```

## Evaluation Framework

### Objective Metrics
- **IoU-Based Layout Similarity**: Per-component overlap for each viewport (text, images, nav, buttons, forms, etc.)
- **Perceptual Similarity**: Block similarity, text region similarity, CLIP-based visual correspondence

### LLM-as-a-Judge Rubrics
1. **Layout Accuracy**: Component presence, ordering, structural fidelity per device
2. **Visual Hierarchy & Spacing**: Grouping, alignment, proportional scale
3. **Cross-Device Responsive Consistency**: Correct stacking/reflow, breakpoint alignment

### Responsive Meter (Composite Score)
```
ResponsiveMeter = w1·Avg(IoU) + w2·CrossDeviceConsistency + w3·LLMJudge + w4·Perceptual
```

## Research Motivation

This project creates the first **multi-sketch → responsive-HTML** benchmark aligned with Sketch2Code methodology, addressing:
- Multi-sketch conditioning for responsive design
- One-shot responsive generation
- Cross-device evaluation metrics
- Responsiveness-aware qualitative evaluation

## Input Format

Wireframe sketches should follow standard wireframing conventions:
- Boxes with "X" for images
- Wavy lines for text
- Rectangles for cards, buttons, sections
- No color or detailed text (low-fidelity abstraction)

Each triplet represents the same webpage at three breakpoints:
- **Mobile**: 375px width
- **Tablet**: 768px width
- **Desktop**: 1280px width

## Output

Generated artifacts for each sample:
- `generated.html` - Single responsive HTML file with inline CSS
- `screenshots/` - Screenshots at mobile, tablet, desktop viewports
- `metrics.json` - Evaluation results
- `generation_log.json` - Metadata and timestamps

## License

MIT License - See LICENSE file for details

## Citation

If you use this work, please cite:

```bibtex
@article{li2024sketch2code,
  title={Sketch2Code: Evaluating Vision-Language Models for Interactive Web Design Prototyping},
  author={Li et al.},
  year={2024}
}
```

