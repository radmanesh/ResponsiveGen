"""
Layout similarity and perceptual metrics for evaluation using working implementations.

This module integrates the html_utils and layout_similarity modules to provide
complete IoU-based layout similarity metrics.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image

from responsive_gen.evaluation.html_utils import (
    extract_visual_components,
    compute_weighted_iou_shapely
)
from responsive_gen.evaluation.layout_similarity import layout_similarity
from responsive_gen.models import (
    ComponentType,
    DeviceType,
    IoUMetrics,
    LayoutAnalysis,
    LayoutComponent,
    PerceptualMetrics,
    RenderedOutput,
)


class ComponentDetector:
    """Detects visual components in rendered webpages using Playwright."""

    def __init__(self):
        """Initialize component detector."""
        pass

    def detect_components(
        self,
        screenshot_path: Union[str, Path],
        device_type: DeviceType
    ) -> LayoutAnalysis:
        """
        Detect visual components in a screenshot.

        This is a wrapper around extract_visual_components that returns
        our standard LayoutAnalysis format.

        Args:
            screenshot_path: Path to screenshot (or HTML file)
            device_type: Device type for context

        Returns:
            LayoutAnalysis with detected components
        """
        # Extract components using html_utils
        component_dict = extract_visual_components(str(screenshot_path))

        # Convert to our LayoutComponent format
        components = []

        # Map component types from html_utils to our ComponentType enum
        type_mapping = {
            'text_block': ComponentType.TEXT,
            'image': ComponentType.IMAGE,
            'video': ComponentType.IMAGE,  # Treat videos as images
            'nav_bar': ComponentType.NAVIGATION,
            'button': ComponentType.BUTTON,
            'form_table': ComponentType.FORM,
            'divider': ComponentType.DIVIDER,
        }

        for comp_type, comp_list in component_dict.items():
            our_type = type_mapping.get(comp_type, ComponentType.TERTIARY)
            for comp in comp_list:
                from responsive_gen.models import BoundingBox
                bbox = BoundingBox(
                    x=comp['box']['x'],
                    y=comp['box']['y'],
                    width=comp['box']['width'],
                    height=comp['box']['height']
                )
                components.append(LayoutComponent(
                    component_type=our_type,
                    bounding_box=bbox,
                    confidence=1.0,
                    metadata={'text_content': comp.get('text_content')}
                ))

        return LayoutAnalysis(
            device_type=device_type,
            components=components
        )


class IoUCalculator:
    """Calculates IoU-based layout similarity metrics."""

    def __init__(self):
        """Initialize IoU calculator."""
        self.component_detector = ComponentDetector()

    def calculate_layout_iou(
        self,
        generated_html: Union[str, Path],
        ground_truth_html: Union[str, Path],
        device_type: DeviceType
    ) -> Dict[ComponentType, float]:
        """
        Calculate per-component-type IoU for a device layout.

        Args:
            generated_html: Path to generated HTML file
            ground_truth_html: Path to ground truth HTML file
            device_type: Device type

        Returns:
            Dictionary mapping component types to IoU scores
        """
        # Extract components from both HTML files
        generated_components = extract_visual_components(str(generated_html))
        ground_truth_components = extract_visual_components(str(ground_truth_html))

        # Compute weighted IoU which includes per-component scores
        _, multi_score = compute_weighted_iou_shapely(generated_components, ground_truth_components)

        # Map to our ComponentType enum
        type_mapping = {
            'text_block': ComponentType.TEXT,
            'image': ComponentType.IMAGE,
            'video': ComponentType.IMAGE,
            'nav_bar': ComponentType.NAVIGATION,
            'button': ComponentType.BUTTON,
            'form_table': ComponentType.FORM,
            'divider': ComponentType.DIVIDER,
        }

        per_component_iou = {}
        for comp_type, (iou, weight) in multi_score.items():
            our_type = type_mapping.get(comp_type, ComponentType.TERTIARY)
            per_component_iou[our_type] = iou

        return per_component_iou

    def calculate_iou_metrics(
        self,
        generated_output: RenderedOutput,
        ground_truth_dir: Path
    ) -> IoUMetrics:
        """
        Calculate comprehensive IoU metrics across all viewports.

        Args:
            generated_output: Rendered output with screenshots
            ground_truth_dir: Directory with ground truth HTML files

        Returns:
            IoUMetrics object with scores
        """
        # Assume ground truth files are named: mobile.html, tablet.html, desktop.html
        gt_mobile = ground_truth_dir / "mobile.html"
        gt_tablet = ground_truth_dir / "tablet.html"
        gt_desktop = ground_truth_dir / "desktop.html"

        # For now, we'll use layout_similarity function
        # Note: This requires the original HTML, not screenshots
        # In a full implementation, you'd render screenshots at each viewport

        # Placeholder: return zero metrics if ground truth not found
        if not all([gt_mobile.exists(), gt_tablet.exists(), gt_desktop.exists()]):
            return IoUMetrics(
                mobile_iou=0.0,
                tablet_iou=0.0,
                desktop_iou=0.0,
                average_iou=0.0,
                per_component_iou={}
            )

        # Use layout_similarity to compute IoU
        # This is a simplified version - full implementation would render at each viewport
        from responsive_gen.evaluation.layout_similarity import (
            compute_layout_similarity_multi_viewport
        )

        # Get the generated HTML path from rendered output
        # This is a workaround - ideally we'd have the HTML path in RenderedOutput
        sample_id = generated_output.sample_id
        # Assume HTML is in outputs/{sample_id}/generated.html
        generated_html = Path("outputs") / sample_id / "generated.html"

        if not generated_html.exists():
            return IoUMetrics(
                mobile_iou=0.0,
                tablet_iou=0.0,
                desktop_iou=0.0,
                average_iou=0.0,
                per_component_iou={}
            )

        results = compute_layout_similarity_multi_viewport(
            str(generated_html),
            str(gt_mobile),
            str(gt_tablet),
            str(gt_desktop),
            debug=False
        )

        return IoUMetrics(
            mobile_iou=results['mobile_iou'],
            tablet_iou=results['tablet_iou'],
            desktop_iou=results['desktop_iou'],
            average_iou=results['average_iou'],
            per_component_iou={}  # TODO: Add per-component breakdown
        )


class PerceptualSimilarity:
    """Calculates perceptual similarity metrics."""

    def __init__(self):
        """Initialize perceptual similarity calculator."""
        # TODO: Load CLIP model for visual similarity
        self.clip_model = None

    def calculate_clip_similarity(
        self,
        generated_screenshot: Path,
        ground_truth_screenshot: Path
    ) -> float:
        """
        Calculate CLIP-based visual similarity.

        Args:
            generated_screenshot: Screenshot of generated page
            ground_truth_screenshot: Screenshot of ground truth page

        Returns:
            Similarity score [0, 1]
        """
        # TODO: Implement using CLIP
        # For now, return placeholder
        return 0.0

    def calculate_block_similarity(
        self,
        generated_screenshot: Path,
        ground_truth_screenshot: Path
    ) -> float:
        """
        Calculate block-level structural similarity.

        Args:
            generated_screenshot: Screenshot of generated page
            ground_truth_screenshot: Screenshot of ground truth page

        Returns:
            Block similarity score [0, 1]
        """
        # TODO: Implement block-based comparison
        return 0.0

    def calculate_text_region_similarity(
        self,
        generated_screenshot: Path,
        ground_truth_screenshot: Path
    ) -> float:
        """
        Calculate similarity of text region placement.

        Args:
            generated_screenshot: Screenshot of generated page
            ground_truth_screenshot: Screenshot of ground truth page

        Returns:
            Text region similarity score [0, 1]
        """
        # TODO: Implement text region comparison
        return 0.0

    def calculate_positional_alignment(
        self,
        generated_screenshot: Path,
        ground_truth_screenshot: Path
    ) -> float:
        """
        Calculate positional alignment of key elements.

        Args:
            generated_screenshot: Screenshot of generated page
            ground_truth_screenshot: Screenshot of ground truth page

        Returns:
            Positional alignment score [0, 1]
        """
        # TODO: Implement positional alignment
        return 0.0

    def calculate_perceptual_metrics(
        self,
        generated_output: RenderedOutput,
        ground_truth_dir: Path
    ) -> PerceptualMetrics:
        """
        Calculate comprehensive perceptual similarity metrics.

        Args:
            generated_output: Rendered output with screenshots
            ground_truth_dir: Directory with ground truth screenshots

        Returns:
            PerceptualMetrics object
        """
        # TODO: Implement full perceptual pipeline
        # For now, return placeholder metrics
        return PerceptualMetrics(
            clip_similarity_mobile=0.0,
            clip_similarity_tablet=0.0,
            clip_similarity_desktop=0.0,
            block_similarity=0.0,
            text_region_similarity=0.0,
            positional_alignment=0.0
        )


class MetricsAggregator:
    """Aggregates and combines multiple metrics."""

    def __init__(self):
        """Initialize metrics aggregator."""
        self.iou_calculator = IoUCalculator()
        self.perceptual_calculator = PerceptualSimilarity()

    def calculate_all_objective_metrics(
        self,
        generated_output: RenderedOutput,
        ground_truth_dir: Path
    ) -> tuple[IoUMetrics, PerceptualMetrics]:
        """
        Calculate all objective metrics.

        Args:
            generated_output: Rendered output with screenshots
            ground_truth_dir: Directory with ground truth screenshots

        Returns:
            Tuple of (IoUMetrics, PerceptualMetrics)
        """
        iou_metrics = self.iou_calculator.calculate_iou_metrics(
            generated_output,
            ground_truth_dir
        )

        perceptual_metrics = self.perceptual_calculator.calculate_perceptual_metrics(
            generated_output,
            ground_truth_dir
        )

        return iou_metrics, perceptual_metrics
