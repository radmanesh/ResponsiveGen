"""
Tests for data models.
"""

import pytest
from pathlib import Path

from responsive_gen.models import (
    BoundingBox,
    DeviceType,
    ViewportConfig,
)


def test_viewport_config_mobile():
    """Test mobile viewport configuration."""
    viewport = ViewportConfig.mobile()
    assert viewport.device_type == DeviceType.MOBILE
    assert viewport.width == 375
    assert viewport.height == 688


def test_viewport_config_tablet():
    """Test tablet viewport configuration."""
    viewport = ViewportConfig.tablet()
    assert viewport.device_type == DeviceType.TABLET
    assert viewport.width == 768
    assert viewport.height == 1024


def test_viewport_config_desktop():
    """Test desktop viewport configuration."""
    viewport = ViewportConfig.desktop()
    assert viewport.device_type == DeviceType.DESKTOP
    assert viewport.width == 1280
    assert viewport.height == 800


def test_bounding_box_area():
    """Test bounding box area calculation."""
    bbox = BoundingBox(x=0, y=0, width=100, height=50)
    assert bbox.area() == 5000


def test_bounding_box_iou_no_overlap():
    """Test IoU calculation with no overlap."""
    bbox1 = BoundingBox(x=0, y=0, width=10, height=10)
    bbox2 = BoundingBox(x=20, y=20, width=10, height=10)
    assert bbox1.iou(bbox2) == 0.0


def test_bounding_box_iou_perfect_overlap():
    """Test IoU calculation with perfect overlap."""
    bbox1 = BoundingBox(x=0, y=0, width=10, height=10)
    bbox2 = BoundingBox(x=0, y=0, width=10, height=10)
    assert bbox1.iou(bbox2) == 1.0


def test_bounding_box_iou_partial_overlap():
    """Test IoU calculation with partial overlap."""
    bbox1 = BoundingBox(x=0, y=0, width=10, height=10)
    bbox2 = BoundingBox(x=5, y=5, width=10, height=10)

    # Intersection: 5x5 = 25
    # Union: 100 + 100 - 25 = 175
    # IoU: 25/175 ≈ 0.143
    iou = bbox1.iou(bbox2)
    assert 0.14 < iou < 0.15


# TODO: Add more tests for other models
# - SketchInput
# - SketchTriplet
# - GeneratedHTML
# - RenderedOutput
# - EvaluationResult

