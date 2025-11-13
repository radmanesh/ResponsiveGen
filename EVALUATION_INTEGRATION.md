# Evaluation Integration Summary

## Overview

Successfully integrated working evaluation modules into ResponsiveGen, replacing scaffolded stubs with functional implementations for IoU-based layout similarity metrics.

## What Was Integrated

### 1. Core Evaluation Modules

#### `responsive_gen/evaluation/html_utils.py`
- **Visual Component Extraction**: Uses Playwright (sync API) to extract components from HTML
- **Component Types**: text_block, image, video, nav_bar, button, form_table, divider
- **Text Block Merging**: Intelligently merges adjacent text blocks
- **IoU Calculation**: Shapely-based geometric IoU computation
- **Normalization**: Converts absolute coordinates to relative positions

**Key Functions:**
- `extract_visual_components(url, save_path)` - Extract and optionally visualize components
- `compute_weighted_iou_shapely(elementsA, elementsB)` - Compute weighted IoU across component types
- `take_and_save_screenshot(url, output_file)` - Screenshot utility

#### `responsive_gen/evaluation/layout_similarity.py`
- **Layout Similarity**: Computes similarity between predicted and reference HTML
- **Multi-Viewport Support**: Handles mobile, tablet, desktop comparisons
- **Per-Component Scoring**: Returns IoU scores per component type

**Key Functions:**
- `layout_similarity(input_list, debug)` - Main similarity computation
- `compute_layout_similarity_multi_viewport(...)` - Multi-viewport evaluation

#### `responsive_gen/evaluation/layout_metrics.py` (Updated)
- **Integration Layer**: Bridges new utilities with existing ResponsiveGen models
- **Component Detection**: Converts html_utils output to LayoutAnalysis format
- **IoU Calculation**: Wraps layout_similarity for ResponsiveMeter use

### 2. Dependencies Added

```
shapely>=2.0.0          # Geometric calculations for IoU
beautifulsoup4>=4.12.0  # HTML parsing
lxml>=5.1.0             # XML/HTML processing
tqdm>=4.66.0            # Progress bars
```

### 3. Data Structure

Created `data/` directory with subdirectories:
- `data/eval/` - Evaluation datasets
- `data/ground_truth/` - Reference HTML files for comparison
- `data/examples/` - Example wireframes

Created `data/README.md` with comprehensive documentation on:
- Ground truth file structure
- HTML requirements
- Evaluation usage examples

### 4. CLI Integration

Updated `cli.py` evaluate command with full functionality:
- Loads generated HTML and screenshots
- Validates ground truth directory structure
- Runs ResponsiveMeter evaluation
- Saves detailed metrics to JSON
- Displays summary statistics

### 5. Documentation Updates

#### README.md
- Added evaluation framework details
- Documented IoU-based metrics as functional
- Added usage examples with ground truth preparation

#### QUICKSTART.md
- Added evaluation workflow section
- Documented ground truth preparation
- Listed all evaluation metrics and their status
- Provided complete evaluation examples

## How It Works

### Component Extraction Pipeline

```
HTML File → Playwright → Bounding Boxes → Normalization → Component Dict
                ↓
        Query Selectors (text, images, nav, etc.)
                ↓
        Filter Visible Elements
                ↓
        Merge Adjacent Text Blocks
                ↓
        Normalize to Relative Coordinates
```

### IoU Calculation Pipeline

```
Generated HTML ──┐
                 ├→ extract_visual_components() → Component Lists
Ground Truth HTML┘                                       ↓
                                          compute_weighted_iou_shapely()
                                                         ↓
                                              Per-Component IoU Scores
                                                         ↓
                                                  Weighted Average
```

### Evaluation Workflow

```
1. Generate HTML (streamlit_app.py or cli.py generate)
   ↓
2. Render Screenshots (automatic in generation)
   ↓
3. Prepare Ground Truth (data/ground_truth/<sample_id>/)
   ├── mobile.html
   ├── tablet.html
   └── desktop.html
   ↓
4. Run Evaluation (cli.py evaluate)
   ↓
5. Get Metrics
   ├── IoU per viewport (functional ✅)
   ├── Perceptual metrics (TODO)
   └── LLM Judge (TODO)
```

## Status of Evaluation Components

### ✅ Fully Functional
- IoU-based layout similarity
- Component extraction (text, images, nav, buttons, forms, dividers)
- Weighted IoU computation using Shapely
- Multi-viewport evaluation (mobile, tablet, desktop)
- Per-component type scoring
- Ground truth comparison
- CLI evaluation command
- Results persistence (JSON)

### 🚧 TODO (Placeholders Remain)
- **CLIP-based perceptual similarity** - Requires loading CLIP model
- **Block-level similarity** - Grid-based feature comparison
- **Text region similarity** - OCR and position matching
- **LLM Judge** - Image encoding and multi-modal prompting
  - Layout accuracy assessment
  - Visual hierarchy evaluation
  - Cross-device consistency scoring

## Usage Examples

### 1. Basic Evaluation

```bash
# Generate HTML
python cli.py generate \
  -m sketches/mobile.png \
  -t sketches/tablet.png \
  -d sketches/desktop.png \
  -o outputs/sample_001

# Prepare ground truth
mkdir -p data/ground_truth/sample_001
cp reference_mobile.html data/ground_truth/sample_001/mobile.html
cp reference_tablet.html data/ground_truth/sample_001/tablet.html
cp reference_desktop.html data/ground_truth/sample_001/desktop.html

# Evaluate
python cli.py evaluate \
  --generated outputs/sample_001/generated.html \
  --ground-truth data/ground_truth/sample_001 \
  --output outputs/sample_001/metrics.json
```

### 2. Python API

```python
from responsive_gen.evaluation.layout_similarity import layout_similarity
from responsive_gen.evaluation.html_utils import extract_visual_components

# Extract components
generated_components = extract_visual_components("outputs/sample_001/generated.html")
ground_truth_components = extract_visual_components("data/ground_truth/sample_001/mobile.html")

# Compute similarity
from responsive_gen.evaluation.html_utils import compute_weighted_iou_shapely
iou_score, multi_score = compute_weighted_iou_shapely(
    generated_components,
    ground_truth_components
)

print(f"IoU Score: {iou_score:.4f}")
print(f"Per-Component Scores: {multi_score}")
```

### 3. Using ResponsiveMeter

```python
from responsive_gen.evaluation.responsive_meter import ResponsiveMeter
from pathlib import Path

meter = ResponsiveMeter(
    w1=0.35,  # IoU weight
    w2=0.25,  # Consistency weight
    w3=0.25,  # LLM judge weight
    w4=0.15   # Perceptual weight
)

result = meter.evaluate(
    wireframe_triplet=triplet,
    rendered_output=rendered,
    ground_truth_dir=Path("data/ground_truth/sample_001")
)

print(f"Composite Score: {result.composite_score:.4f}")
```

## Implementation Notes

### Why Sync Playwright?
The original `html_utils.py` uses synchronous Playwright API (`playwright.sync_api`), while the rendering module uses async API. Both are valid; they're kept separate for compatibility with existing code.

### Ground Truth Requirements
- Must be complete HTML files (not just screenshots)
- Should render correctly at target viewport sizes
- Separate files for each viewport enable proper responsive evaluation
- File naming convention: `mobile.html`, `tablet.html`, `desktop.html`

### Shapely for IoU
Using Shapely geometric library provides:
- Accurate polygon-based IoU (handles overlaps correctly)
- Union of multiple components
- Efficient spatial operations
- Robust handling of edge cases

## Next Steps for Full Implementation

### High Priority
1. **CLIP Integration** - Add perceptual similarity
   ```python
   from transformers import CLIPProcessor, CLIPModel
   # Load model and compute embeddings
   ```

2. **LLM Judge Image Encoding** - Enable multi-modal evaluation
   ```python
   # Encode wireframes and screenshots as base64
   # Include in LLM prompts
   # Parse structured responses
   ```

### Medium Priority
3. **Block Similarity** - Grid-based structural comparison
4. **Text Region Similarity** - OCR + position matching
5. **Positional Alignment** - Key element position analysis

### Enhancement Opportunities
- Parallel evaluation for multiple samples
- Caching of component extractions
- Visualization of component matches
- Detailed per-component reports
- Integration with responsive-metrics.py checks

## Testing

```bash
# Install updated dependencies
pip install -r requirements.txt
playwright install chromium

# Test component extraction
python -c "from responsive_gen.evaluation.html_utils import extract_visual_components; \
           print(extract_visual_components('path/to/test.html'))"

# Test layout similarity
python -c "from responsive_gen.evaluation.layout_similarity import layout_similarity; \
           print(layout_similarity((['test.html'], 'reference.html')))"

# Test full evaluation
python cli.py evaluate --generated outputs/test/generated.html \
                       --ground-truth data/ground_truth/test \
                       --output outputs/test/metrics.json
```

## Files Modified

1. `requirements.txt` - Added shapely, beautifulsoup4, lxml, tqdm
2. `pyproject.toml` - Added same dependencies
3. `responsive_gen/evaluation/html_utils.py` - NEW
4. `responsive_gen/evaluation/layout_similarity.py` - NEW
5. `responsive_gen/evaluation/layout_metrics.py` - UPDATED
6. `responsive_gen/evaluation/responsive_meter.py` - UPDATED
7. `cli.py` - UPDATED (evaluate command)
8. `README.md` - UPDATED (evaluation section)
9. `QUICKSTART.md` - UPDATED (evaluation workflow)
10. `data/README.md` - NEW (ground truth documentation)
11. `data/` structure - CREATED

## Conclusion

The evaluation system now has a fully functional IoU-based layout similarity pipeline that:
- ✅ Extracts visual components from HTML using Playwright
- ✅ Computes geometric IoU using Shapely
- ✅ Supports multi-viewport evaluation
- ✅ Provides per-component scoring
- ✅ Integrates with CLI and Python API
- ✅ Saves detailed metrics to JSON

The foundation is solid for adding the remaining evaluation components (CLIP, LLM Judge, etc.) when ready.

