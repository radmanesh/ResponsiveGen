# Quick Start Guide

## Installation

### Option 1: Using Mamba/Conda

```bash
# Create and activate mamba environment
mamba create -n responsive python=3.10
mamba activate responsive

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium
```

### Option 2: Using Poetry

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
make install

# This will also install Playwright browsers
```

### Option 3: Using pip + venv

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium
```

## Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys
# Required: OPENAI_API_KEY or ANTHROPIC_API_KEY
nano .env
```

## Running the Web Interface (Recommended)

```bash
# Launch Streamlit app
make run-app

# Open browser to http://localhost:8501
```

### Using the Web Interface

1. **Upload Sketches**: Drag and drop three wireframe images (mobile, tablet, desktop)
2. **Configure Settings**: Adjust LLM provider, model, and evaluation weights in sidebar
3. **Generate**: Click "Generate Responsive HTML" button
4. **View Results**: Navigate to "Results" tab to see screenshots and HTML
5. **Evaluate**: Go to "Evaluation" tab to run metrics (placeholder implementation)

## Command-Line Interface

### Generate responsive HTML

```bash
python cli.py generate \
  --mobile examples/mobile.png \
  --tablet examples/tablet.png \
  --desktop examples/desktop.png \
  --output outputs/my_sample \
  --provider openai \
  --model gpt-4-vision-preview
```

### Render existing HTML

```bash
python cli.py render \
  --html outputs/my_sample/generated.html \
  --output outputs/my_sample/screenshots \
  --sample-id my_sample
```

### Evaluate (stub implementation)

```bash
python cli.py evaluate \
  --generated outputs/my_sample/generated.html \
  --ground-truth path/to/ground_truth/ \
  --output outputs/my_sample/metrics.json
```

## Example Workflow

```bash
# 1. Prepare wireframe sketches (PNG/JPG format)
# - mobile.png (375px width recommended)
# - tablet.png (768px width recommended)
# - desktop.png (1280px width recommended)

# 2. Generate responsive HTML
python cli.py generate \
  -m sketches/mobile.png \
  -t sketches/tablet.png \
  -d sketches/desktop.png \
  -o outputs/sample_001

# 3. Check outputs
ls -R outputs/sample_001/
# outputs/sample_001/
# ├── generated.html
# ├── generation_log.json
# └── screenshots/
#     ├── mobile.png
#     ├── tablet.png
#     └── desktop.png

# 4. Open generated HTML in browser
open outputs/sample_001/generated.html

# 5. (Optional) Evaluate against ground truth
python cli.py evaluate \
  --generated outputs/sample_001/generated.html \
  --ground-truth data/ground_truth/sample_001 \
  --output outputs/sample_001/metrics.json
```

## Evaluation

### Preparing Ground Truth Data

Create ground truth HTML files for each viewport:

```bash
# Structure
data/ground_truth/sample_001/
├── mobile.html      # Reference HTML for mobile (375px)
├── tablet.html      # Reference HTML for tablet (768px)
└── desktop.html     # Reference HTML for desktop (1280px)
```

See `data/README.md` for detailed requirements.

### Running Evaluation

```bash
# Evaluate generated HTML against ground truth
python cli.py evaluate \
  --generated outputs/sample_001/generated.html \
  --ground-truth data/ground_truth/sample_001 \
  --output outputs/sample_001/metrics.json

# View results
cat outputs/sample_001/metrics.json
```

### Evaluation Metrics

The evaluation pipeline computes:

1. **IoU Metrics** (Intersection over Union)
   - Per-viewport layout similarity
   - Component-level matching (text, images, nav, buttons, etc.)
   - Uses Playwright for component extraction and Shapely for geometric calculations

2. **Perceptual Metrics** (TODO)
   - CLIP-based visual similarity
   - Block-level structural comparison
   - Text region alignment

3. **LLM Judge** (TODO)
   - Qualitative assessment of layout accuracy
   - Visual hierarchy evaluation
   - Cross-device consistency scoring

4. **Responsive Meter**
   - Composite score combining all metrics
   - Configurable weights for each dimension

## Development

```bash
# Format code
make format

# Lint code
make lint

# Run tests (when implemented)
make test

# Clean outputs and caches
make clean
```

## Troubleshooting

### Issue: "Playwright browser not found"

```bash
# Install Playwright browsers
make playwright
# or
poetry run playwright install chromium
```

### Issue: "API key not found"

- Ensure `.env` file exists and contains valid API keys
- Check that environment variables are set: `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`

### Issue: "Import errors"

```bash
# Reinstall dependencies
make clean
make install
```

### Issue: "Screenshot rendering fails"

- Check that Playwright browsers are installed (`make playwright`)
- Try running with `--headed` flag to see browser window
- Increase `--render-wait` time if pages load slowly

## Next Steps

1. **Create wireframe sketches** following conventions (see README.md)
2. **Run generation pipeline** with your sketches
3. **Implement evaluation metrics** (currently placeholder stubs)
4. **Build dataset** of wireframe/webpage pairs for benchmarking

For more details, see README.md and project documentation.

