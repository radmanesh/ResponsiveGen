# Data Directory Structure

This directory contains data files for evaluation and testing of the Responsive Web Generation pipeline.

## Directory Structure

```
data/
├── eval/              # Evaluation datasets
├── ground_truth/      # Ground truth HTML and screenshots
└── examples/          # Example wireframe sketches and outputs
```

## Ground Truth Data

For proper evaluation, place ground truth files in `ground_truth/` with the following structure:

```
ground_truth/
├── <sample_id>/
│   ├── mobile.html      # Ground truth HTML for mobile viewport
│   ├── tablet.html      # Ground truth HTML for tablet viewport
│   ├── desktop.html     # Ground truth HTML for desktop viewport
│   ├── mobile.png       # Optional: Screenshot of mobile view
│   ├── tablet.png       # Optional: Screenshot of tablet view
│   └── desktop.png      # Optional: Screenshot of desktop view
```

### Ground Truth HTML Requirements

Each ground truth HTML file should:
- Be a complete, static HTML file
- Render correctly at its target viewport size:
  - **mobile.html**: 375px width
  - **tablet.html**: 768px width
  - **desktop.html**: 1280px width
- Include proper viewport meta tags
- Use semantic HTML5 elements
- Be self-contained (inline CSS or embedded stylesheets)

## Evaluation Data (`eval/`)

Place HTML files and associated data for batch evaluation in this directory.

Supported formats:
- Individual HTML files: `sample_001.html`, `sample_002.html`, etc.
- With screenshots: `sample_001.png`, `sample_002.png`
- Metadata: `res_dict.json` (optional, for batch processing)

### Metadata Format (`res_dict.json`)

```json
[
  {
    "id": "sample_001",
    "results": [
      {
        "filename": "path/to/generated.html",
        "timestamp": "2024-01-01T00:00:00"
      }
    ]
  }
]
```

## Example Data (`examples/`)

Place example wireframe sketches and sample outputs here for testing and demonstration.

Recommended structure:
```
examples/
├── sample_001/
│   ├── mobile.png       # Mobile wireframe sketch
│   ├── tablet.png       # Tablet wireframe sketch
│   ├── desktop.png      # Desktop wireframe sketch
│   └── generated.html   # Example generated output
```

## Wireframe Sketch Guidelines

Wireframe sketches should follow standard wireframing conventions:

1. **Images**: Boxes with "X" or placeholder icons
2. **Text**: Wavy lines or Lorem ipsum placeholder
3. **Buttons**: Clearly defined rectangles with labels
4. **Navigation**: Horizontal or vertical bar with menu items
5. **Cards**: Grouped rectangular sections
6. **Forms**: Input fields and labels

### Viewport Sizes

- **Mobile**: 375 × 688px (iPhone standard)
- **Tablet**: 768 × 1024px (iPad standard)
- **Desktop**: 1280 × 800px (HD standard)

## Usage in Evaluation

### Command-Line Evaluation

```bash
python cli.py evaluate \
  --generated outputs/sample_001/generated.html \
  --ground-truth data/ground_truth/sample_001 \
  --output outputs/sample_001/metrics.json
```

### Python API

```python
from responsive_gen.evaluation.responsive_meter import ResponsiveMeter
from responsive_gen.io.sketch_loader import SketchLoader
from pathlib import Path

# Load wireframes
loader = SketchLoader()
triplet = loader.load_triplet(
    "data/examples/sample_001/mobile.png",
    "data/examples/sample_001/tablet.png",
    "data/examples/sample_001/desktop.png"
)

# Evaluate
meter = ResponsiveMeter()
result = meter.evaluate(
    wireframe_triplet=triplet,
    rendered_output=rendered,
    ground_truth_dir=Path("data/ground_truth/sample_001")
)

print(f"Composite Score: {result.composite_score}")
print(f"Average IoU: {result.iou_metrics.average_iou}")
```

## Notes

- All paths in evaluation scripts are relative to the project root
- Ground truth files are optional but required for objective metrics
- Screenshots are automatically generated if not provided
- Wireframe sketches can be hand-drawn or created with tools (Figma, Sketch, etc.)

