"""
Tests for sketch loader.
"""

import pytest
from pathlib import Path
from PIL import Image

# TODO: Implement tests for SketchLoader
# These require sample wireframe images in tests/fixtures/

# @pytest.fixture
# def sample_wireframes(tmp_path):
#     """Create sample wireframe images for testing."""
#     mobile = Image.new('RGB', (375, 688), color='white')
#     tablet = Image.new('RGB', (768, 1024), color='white')
#     desktop = Image.new('RGB', (1280, 800), color='white')
#
#     mobile.save(tmp_path / "mobile.png")
#     tablet.save(tmp_path / "tablet.png")
#     desktop.save(tmp_path / "desktop.png")
#
#     return tmp_path

# def test_load_image(sample_wireframes):
#     """Test image loading."""
#     pass

# def test_load_triplet(sample_wireframes):
#     """Test loading sketch triplet."""
#     pass

# def test_validate_sketches(sample_wireframes):
#     """Test sketch validation."""
#     pass

# def test_prepare_for_llm(sample_wireframes):
#     """Test LLM input preparation."""
#     pass

