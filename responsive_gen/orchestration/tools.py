"""
Tools for LangGraph orchestration.

Wraps existing ResponsiveGen functionality as LangChain tools for use by agents.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup
from langchain_core.tools import tool

from responsive_gen.evaluation.html_utils import (
    compute_weighted_iou_shapely,
    extract_visual_components,
    take_and_save_screenshot,
)
from responsive_gen.evaluation.layout_similarity import layout_similarity
from responsive_gen.evaluation.responsive_meter import ResponsiveMeter
from responsive_gen.io.sketch_loader import ArtifactManager, SketchLoader
from responsive_gen.models import DeviceType, RenderedOutput
from responsive_gen.pipeline.generation import ResponsiveGenerator
from responsive_gen.rendering.html_renderer import HTMLRenderer


@tool
def generate_html(sample_id: str, sketch_triplet_path: str) -> Dict[str, str]:
    """
    Generate responsive HTML from wireframe sketch triplet.

    Args:
        sample_id: Unique identifier for this generation
        sketch_triplet_path: Path to directory containing mobile.png, tablet.png, desktop.png

    Returns:
        Dictionary with 'html' (HTML content) and 'html_path' (file path)
    """
    sketch_dir = Path(sketch_triplet_path)
    mobile_path = sketch_dir / "mobile.png"
    tablet_path = sketch_dir / "tablet.png"
    desktop_path = sketch_dir / "desktop.png"

    if not all([mobile_path.exists(), tablet_path.exists(), desktop_path.exists()]):
        raise FileNotFoundError(f"Missing sketch files in {sketch_triplet_path}")

    # Load sketches
    loader = SketchLoader()
    triplet = loader.load_triplet(
        mobile_path, tablet_path, desktop_path, sample_id=sample_id
    )

    # Generate HTML
    generator = ResponsiveGenerator()
    generated, html_path = generator.generate_and_save(
        triplet, output_dir=Path("outputs"), sketch_loader=loader
    )

    return {
        "html": generated.html_content,
        "html_path": str(html_path),
        "sample_id": sample_id
    }


@tool
def read_html(html_source: str, selector: Optional[str] = None) -> str:
    """
    Read HTML content, optionally extracting a fragment by CSS selector.

    Args:
        html_source: Either raw HTML string or path to HTML file
        selector: Optional CSS selector (e.g., "#hero", ".nav", "main > section:nth-child(2)")

    Returns:
        Full HTML if no selector, otherwise the HTML fragment for that selector
    """
    # Determine if html_source is a file path or HTML string
    if os.path.exists(html_source):
        with open(html_source, 'r', encoding='utf-8') as f:
            html_content = f.read()
    else:
        html_content = html_source

    soup = BeautifulSoup(html_content, 'html.parser')

    if selector is None:
        return str(soup)

    # Find element by selector
    element = soup.select_one(selector)
    if element is None:
        # Try fallback strategies for common selectors
        if selector == "body":
            # If body not found, try to get the body tag directly or return html
            body = soup.find('body')
            if body:
                return str(body)
            # If still not found, return the full HTML document
            return str(soup)
        elif selector == "html":
            # Return the full document
            return str(soup)
        else:
            # For other selectors, return error message
            return f"<!-- Selector '{selector}' not found -->"

    return str(element)


@tool
def modify_html(html_source: str, selector: str, new_fragment: str) -> str:
    """
    Replace HTML element matching selector with new fragment.

    Args:
        html_source: Either raw HTML string or path to HTML file
        selector: CSS selector for element to replace
        new_fragment: New HTML fragment to insert

    Returns:
        Updated full HTML string
    """
    # Determine if html_source is a file path or HTML string
    if os.path.exists(html_source):
        with open(html_source, 'r', encoding='utf-8') as f:
            html_content = f.read()
    else:
        html_content = html_source

    soup = BeautifulSoup(html_content, 'html.parser')

    # Find element to replace
    element = soup.select_one(selector)
    if element is None:
        # Try fallback strategies for common selectors
        if selector == "body":
            # If body not found, try to get the body tag directly
            body = soup.find('body')
            if body:
                element = body
            else:
                # If still no body, try to append to html or create body
                html_tag = soup.find('html')
                if html_tag:
                    # Create a new body tag and append to html
                    from bs4 import Tag
                    new_body = soup.new_tag('body')
                    html_tag.append(new_body)
                    element = new_body
                else:
                    raise ValueError(f"Selector '{selector}' not found in HTML and unable to create body tag")
        elif selector == "html":
            # For html selector, we modify the html tag itself
            element = soup.find('html')
            if element is None:
                raise ValueError(f"Selector '{selector}' not found in HTML")
        else:
            raise ValueError(f"Selector '{selector}' not found in HTML")

    # Parse new fragment and replace
    new_soup = BeautifulSoup(new_fragment, 'html.parser')
    element.replace_with(new_soup)

    return str(soup)


@tool
def take_screenshot(html_source: str, view: str) -> str:
    """
    Take screenshot of HTML at specified viewport.

    Args:
        html_source: Either raw HTML string or path to HTML file
        view: Viewport type - "mobile", "tablet", or "desktop"

    Returns:
        Path to saved screenshot
    """
    # Handle HTML string by saving to temp file
    if not os.path.exists(html_source):
        # It's an HTML string, save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_source)
            html_path = f.name
    else:
        html_path = html_source

    # Map view to viewport config
    from responsive_gen.models import ViewportConfig
    if view == "mobile":
        viewport = ViewportConfig.mobile()
    elif view == "tablet":
        viewport = ViewportConfig.tablet()
    elif view == "desktop":
        viewport = ViewportConfig.desktop()
    else:
        raise ValueError(f"Invalid view: {view}. Must be 'mobile', 'tablet', or 'desktop'")

    # Render screenshot
    renderer = HTMLRenderer(headless=True)
    output_dir = Path("outputs") / "screenshots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{view}.png"

    renderer.render_viewport(html_path, viewport, output_path)

    return str(output_path)


@tool
def run_iou_evaluation(sample_id: str, html_path: str, view: str, ground_truth_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run IoU-based layout evaluation for a specific viewport.

    Args:
        sample_id: Sample identifier
        html_path: Path to generated HTML file
        view: Viewport type - "mobile", "tablet", or "desktop"
        ground_truth_path: Optional path to ground truth HTML for comparison

    Returns:
        Dictionary with IoU scores and component breakdown
    """
    if ground_truth_path and os.path.exists(ground_truth_path):
        # Compare against ground truth
        scores, multi_scores = layout_similarity(
            ([html_path], ground_truth_path),
            debug=False
        )
        iou_score = scores[0] if scores else 0.0
        multi_score = multi_scores[0] if multi_scores else {}
    else:
        # Extract components only (no comparison)
        components = extract_visual_components(html_path)
        iou_score = 0.0
        multi_score = {}
        for comp_type, comp_list in components.items():
            multi_score[comp_type] = (0.0, len(comp_list))

    return {
        "view": view,
        "iou_score": iou_score,
        "per_component_iou": multi_score,
        "has_ground_truth": ground_truth_path is not None and os.path.exists(ground_truth_path)
    }


@tool
def run_perceptual_evaluation(sample_id: str, html_path: str, view: str) -> Dict[str, Any]:
    """
    Run perceptual similarity evaluation (CLIP-based).

    Args:
        sample_id: Sample identifier
        html_path: Path to HTML file
        view: Viewport type

    Returns:
        Dictionary with perceptual similarity scores
    """
    # TODO: Implement CLIP-based evaluation
    return {
        "view": view,
        "clip_similarity": 0.0,
        "block_similarity": 0.0,
        "text_region_similarity": 0.0,
        "status": "not_implemented"
    }


@tool
def llm_judge_layout(html: str, screenshots: Dict[str, str]) -> float:
    """
    LLM-based layout accuracy assessment.

    Args:
        html: HTML content
        screenshots: Dictionary mapping viewport to screenshot path

    Returns:
        Layout accuracy score (0.0 to 1.0)
    """
    # TODO: Implement LLM judge
    return 0.0


@tool
def llm_judge_responsiveness(html: str, screenshots: Dict[str, str]) -> float:
    """
    LLM-based cross-device responsive consistency assessment.

    Args:
        html: HTML content
        screenshots: Dictionary mapping viewport to screenshot path

    Returns:
        Responsive consistency score (0.0 to 1.0)
    """
    # TODO: Implement LLM judge
    return 0.0


@tool
def compute_responsive_meter(
    iou_metrics: Dict[str, float],
    perceptual_metrics: Optional[Dict[str, float]] = None,
    llm_judge_metrics: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute composite ResponsiveMeter score from all metrics.

    Args:
        iou_metrics: Dictionary with 'mobile', 'tablet', 'desktop' IoU scores
        perceptual_metrics: Optional perceptual similarity scores
        llm_judge_metrics: Optional LLM judge scores

    Returns:
        Composite ResponsiveMeter score (0.0 to 1.0)
    """
    # Calculate average IoU
    avg_iou = (
        iou_metrics.get("mobile", 0.0) +
        iou_metrics.get("tablet", 0.0) +
        iou_metrics.get("desktop", 0.0)
    ) / 3.0

    # Default weights
    w1 = 0.35  # IoU
    w2 = 0.25  # Consistency
    w3 = 0.25  # LLM judge
    w4 = 0.15  # Perceptual

    # Get other scores (default to 0.0 if not provided)
    consistency = llm_judge_metrics.get("consistency", 0.0) if llm_judge_metrics else 0.0
    llm_score = llm_judge_metrics.get("layout_accuracy", 0.0) if llm_judge_metrics else 0.0
    perceptual = perceptual_metrics.get("clip_similarity", 0.0) if perceptual_metrics else 0.0

    # Compute composite score
    composite = (
        w1 * avg_iou +
        w2 * consistency +
        w3 * llm_score +
        w4 * perceptual
    )

    return composite


# Tool list for easy import
RESPONSIVE_TOOLS = [
    generate_html,
    read_html,
    modify_html,
    take_screenshot,
    run_iou_evaluation,
    run_perceptual_evaluation,
    llm_judge_layout,
    llm_judge_responsiveness,
    compute_responsive_meter,
]

