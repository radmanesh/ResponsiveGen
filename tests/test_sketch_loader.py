"""
Tests for sketch loader.
"""

import base64
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from responsive_gen.io.sketch_loader import SketchLoader
from responsive_gen.models import DeviceType


@pytest.fixture
def sample_wireframes(tmp_path):
    """Create sample wireframe images for testing."""
    sizes = {
        "mobile": (375, 687),
        "tablet": (1024, 1366),
        "desktop": (1280, 1080),
    }

    paths = {}
    for name, size in sizes.items():
        image = Image.new("RGB", size, color="white")
        file_path = tmp_path / f"{name}.png"
        image.save(file_path)
        paths[name] = file_path

    return paths


def test_load_image(sample_wireframes):
    loader = SketchLoader()

    image = loader.load_image(sample_wireframes["mobile"])

    assert image.mode == "RGB"
    assert image.size == (375, 687)


def test_load_triplet(sample_wireframes):
    loader = SketchLoader()

    triplet = loader.load_triplet(
        sample_wireframes["mobile"],
        sample_wireframes["tablet"],
        sample_wireframes["desktop"],
        sample_id="sample-123",
        designer="alice"
    )

    assert triplet.sample_id == "sample-123"
    assert triplet.metadata == {"designer": "alice"}
    assert triplet.mobile.device_type == DeviceType.MOBILE
    assert triplet.mobile.viewport.width == 375
    assert triplet.mobile.viewport.height == 687
    assert triplet.tablet.viewport.width == 1024
    assert triplet.tablet.viewport.height == 1366
    assert triplet.desktop.viewport.width == 1280
    assert triplet.desktop.viewport.height == 1080


def test_validate_sketches(sample_wireframes):
    loader = SketchLoader()
    triplet = loader.load_triplet(
        sample_wireframes["mobile"],
        sample_wireframes["tablet"],
        sample_wireframes["desktop"],
    )

    is_valid, error = loader.validate_sketches(triplet)
    assert is_valid
    assert error is None

    # Remove desktop sketch to trigger validation failure
    Path(sample_wireframes["desktop"]).unlink()
    is_valid, error = loader.validate_sketches(triplet)
    assert not is_valid
    assert "desktop" in error.lower()


def test_prepare_for_llm(sample_wireframes):
    loader = SketchLoader()
    triplet = loader.load_triplet(
        sample_wireframes["mobile"],
        sample_wireframes["tablet"],
        sample_wireframes["desktop"],
        sample_id="wireframe"
    )

    payload = loader.prepare_for_llm(triplet)

    assert payload["sample_id"] == "wireframe"
    assert set(payload["images"]) == {"mobile", "tablet", "desktop"}
    assert payload["viewports"]["mobile"]["width"] == 375

    # Ensure normalization output can be decoded and matches viewport
    decoded = base64.b64decode(payload["images"]["tablet"])
    image = Image.open(BytesIO(decoded))
    assert image.size == (1024, 1366)
