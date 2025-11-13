"""
Utilities for loading and normalizing wireframe sketches.
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image

from responsive_gen.models import (
    DeviceType,
    SketchInput,
    SketchTriplet,
    ViewportConfig,
)


class SketchLoader:
    """Loads and normalizes wireframe sketch images."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize sketch loader.

        Args:
            cache_dir: Optional directory for caching processed images.
        """
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load an image from disk.

        Args:
            image_path: Path to the image file.

        Returns:
            PIL Image object.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        return Image.open(image_path).convert("RGB")

    def normalize_sketch(
        self,
        image: Image.Image,
        target_viewport: ViewportConfig,
        maintain_aspect: bool = True
    ) -> Image.Image:
        """
        Normalize sketch to target viewport dimensions.

        Args:
            image: Input PIL Image.
            target_viewport: Target viewport configuration.
            maintain_aspect: Whether to maintain aspect ratio.

        Returns:
            Normalized PIL Image.
        """
        target_width = target_viewport.width
        target_height = target_viewport.height

        if maintain_aspect:
            # Resize maintaining aspect ratio, fit within target dimensions
            image.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
        else:
            # Resize to exact dimensions
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)

        return image

    def image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """
        Convert PIL Image to base64 string.

        Args:
            image: PIL Image object.
            format: Image format (PNG, JPEG, etc.).

        Returns:
            Base64-encoded string.
        """
        buffer = BytesIO()
        image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def load_sketch_input(
        self,
        image_path: Union[str, Path],
        device_type: DeviceType,
        viewport_height: Optional[int] = None
    ) -> SketchInput:
        """
        Load a single sketch input for a specific device type.

        Args:
            image_path: Path to the wireframe sketch.
            device_type: Device type (mobile, tablet, desktop).
            viewport_height: Optional custom viewport height.

        Returns:
            SketchInput object.
        """
        image_path = Path(image_path)

        # Create viewport config
        if device_type == DeviceType.MOBILE:
            viewport = ViewportConfig.mobile(viewport_height or 688)
        elif device_type == DeviceType.TABLET:
            viewport = ViewportConfig.tablet(viewport_height or 1024)
        elif device_type == DeviceType.DESKTOP:
            viewport = ViewportConfig.desktop(viewport_height or 800)
        else:
            raise ValueError(f"Unknown device type: {device_type}")

        return SketchInput(
            device_type=device_type,
            image_path=image_path,
            viewport=viewport
        )

    def load_triplet(
        self,
        mobile_path: Union[str, Path],
        tablet_path: Union[str, Path],
        desktop_path: Union[str, Path],
        sample_id: Optional[str] = None,
        **metadata
    ) -> SketchTriplet:
        """
        Load a complete sketch triplet.

        Args:
            mobile_path: Path to mobile wireframe.
            tablet_path: Path to tablet wireframe.
            desktop_path: Path to desktop wireframe.
            sample_id: Optional sample identifier.
            **metadata: Additional metadata to attach.

        Returns:
            SketchTriplet object.
        """
        if sample_id is None:
            # Generate sample ID from mobile filename
            sample_id = Path(mobile_path).stem

        mobile = self.load_sketch_input(mobile_path, DeviceType.MOBILE)
        tablet = self.load_sketch_input(tablet_path, DeviceType.TABLET)
        desktop = self.load_sketch_input(desktop_path, DeviceType.DESKTOP)

        return SketchTriplet(
            sample_id=sample_id,
            mobile=mobile,
            tablet=tablet,
            desktop=desktop,
            metadata=metadata
        )

    def validate_sketches(self, triplet: SketchTriplet) -> Tuple[bool, Optional[str]]:
        """
        Validate that all sketch images exist and are readable.

        Args:
            triplet: SketchTriplet to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            for device_type in [DeviceType.MOBILE, DeviceType.TABLET, DeviceType.DESKTOP]:
                sketch = triplet.get_sketch(device_type)
                if not sketch.image_path.exists():
                    return False, f"{device_type.value} sketch not found: {sketch.image_path}"

                # Try to load the image
                self.load_image(sketch.image_path)

            return True, None
        except Exception as e:
            return False, str(e)

    def prepare_for_llm(
        self,
        triplet: SketchTriplet,
        normalize: bool = True
    ) -> dict:
        """
        Prepare sketch triplet for LLM input (base64 encoded images).

        Args:
            triplet: SketchTriplet to prepare.
            normalize: Whether to normalize images to viewport sizes.

        Returns:
            Dictionary with base64-encoded images and metadata.
        """
        result = {
            "sample_id": triplet.sample_id,
            "images": {},
            "viewports": {}
        }

        for device_type in [DeviceType.MOBILE, DeviceType.TABLET, DeviceType.DESKTOP]:
            sketch = triplet.get_sketch(device_type)
            image = self.load_image(sketch.image_path)

            if normalize:
                image = self.normalize_sketch(image, sketch.viewport)

            result["images"][device_type.value] = self.image_to_base64(image)
            result["viewports"][device_type.value] = {
                "width": sketch.viewport.width,
                "height": sketch.viewport.height
            }

        return result


class ArtifactManager:
    """Manages reading and writing pipeline artifacts."""

    def __init__(self, output_dir: Union[str, Path] = "outputs"):
        """
        Initialize artifact manager.

        Args:
            output_dir: Root directory for pipeline outputs.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_sample_directory(self, sample_id: str) -> Path:
        """
        Create output directory for a sample.

        Args:
            sample_id: Sample identifier.

        Returns:
            Path to sample directory.
        """
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (sample_dir / "screenshots").mkdir(exist_ok=True)
        (sample_dir / "logs").mkdir(exist_ok=True)

        return sample_dir

    def save_html(self, sample_id: str, html_content: str, filename: str = "generated.html") -> Path:
        """
        Save generated HTML to disk.

        Args:
            sample_id: Sample identifier.
            html_content: HTML content to save.
            filename: Output filename.

        Returns:
            Path to saved HTML file.
        """
        sample_dir = self.create_sample_directory(sample_id)
        html_path = sample_dir / filename
        html_path.write_text(html_content, encoding="utf-8")
        return html_path

    def save_screenshot(
        self,
        sample_id: str,
        screenshot: Image.Image,
        device_type: DeviceType
    ) -> Path:
        """
        Save rendered screenshot.

        Args:
            sample_id: Sample identifier.
            screenshot: PIL Image of screenshot.
            device_type: Device type for naming.

        Returns:
            Path to saved screenshot.
        """
        sample_dir = self.create_sample_directory(sample_id)
        screenshot_path = sample_dir / "screenshots" / f"{device_type.value}.png"
        screenshot.save(screenshot_path, "PNG")
        return screenshot_path

    def load_html(self, sample_id: str, filename: str = "generated.html") -> str:
        """
        Load generated HTML from disk.

        Args:
            sample_id: Sample identifier.
            filename: HTML filename.

        Returns:
            HTML content.
        """
        html_path = self.output_dir / sample_id / filename
        if not html_path.exists():
            raise FileNotFoundError(f"HTML file not found: {html_path}")
        return html_path.read_text(encoding="utf-8")

