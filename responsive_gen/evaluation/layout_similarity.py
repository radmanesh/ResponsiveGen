"""
Layout similarity computation using weighted IoU.

This module provides functions to compute layout similarity between
predicted and reference HTML files using visual component extraction
and weighted Intersection over Union (IoU) metrics.
"""

from typing import List, Tuple, Dict, Optional

from responsive_gen.evaluation.html_utils import (
    extract_visual_components,
    compute_weighted_iou_shapely
)


def layout_similarity(
    input_list: Tuple[List[str], str],
    debug: bool = False
) -> Tuple[List[float], List[Dict[str, Tuple[float, float]]]]:
    """
    Compute layout similarity between predicted and reference HTML files.

    Args:
        input_list: Tuple of (list of predicted HTML paths, reference HTML path)
        debug: If True, save annotated screenshots

    Returns:
        Tuple of (similarity_scores, multi_scale_scores)
        - similarity_scores: List of overall IoU scores for each prediction
        - multi_scale_scores: List of per-component scores and weights
    """
    predict_html_list, original_html = input_list[0], input_list[1]
    results = []
    multi_scores = []

    # Extract reference layout
    reference_save_path = None
    if debug:
        reference_save_path = original_html.replace(".html", "_ref.png")

    reference_layout = extract_visual_components(original_html, reference_save_path)

    # Compare each predicted layout to reference
    for predict_html in predict_html_list:
        predict_save_path = predict_html.replace(".html", "_pred.png") if debug else None
        predict_layout = extract_visual_components(predict_html, predict_save_path)

        iou_score, multi_score = compute_weighted_iou_shapely(predict_layout, reference_layout)
        results.append(iou_score)
        multi_scores.append(multi_score)

    return results, multi_scores


def compute_layout_similarity_multi_viewport(
    generated_html: str,
    ground_truth_mobile: str,
    ground_truth_tablet: str,
    ground_truth_desktop: str,
    debug: bool = False
) -> Dict[str, float]:
    """
    Compute layout similarity for a responsive webpage across multiple viewports.

    This function is designed for the ResponsiveGen pipeline where a single
    generated HTML file should match different ground truth layouts at different
    viewport sizes.

    Args:
        generated_html: Path to generated responsive HTML file
        ground_truth_mobile: Path to ground truth mobile HTML
        ground_truth_tablet: Path to ground truth tablet HTML
        ground_truth_desktop: Path to ground truth desktop HTML
        debug: If True, save annotated screenshots

    Returns:
        Dictionary with IoU scores for each viewport and average:
        {
            'mobile_iou': float,
            'tablet_iou': float,
            'desktop_iou': float,
            'average_iou': float
        }
    """
    # Note: This is a simplified version. For true multi-viewport comparison,
    # you'd need to render the generated HTML at each viewport size and
    # extract components from those rendered versions.

    results = {}

    # Compute similarity for each viewport
    mobile_scores, _ = layout_similarity(
        ([generated_html], ground_truth_mobile),
        debug=debug
    )
    results['mobile_iou'] = mobile_scores[0]

    tablet_scores, _ = layout_similarity(
        ([generated_html], ground_truth_tablet),
        debug=debug
    )
    results['tablet_iou'] = tablet_scores[0]

    desktop_scores, _ = layout_similarity(
        ([generated_html], ground_truth_desktop),
        debug=debug
    )
    results['desktop_iou'] = desktop_scores[0]

    # Compute average
    results['average_iou'] = (
        results['mobile_iou'] +
        results['tablet_iou'] +
        results['desktop_iou']
    ) / 3.0

    return results

