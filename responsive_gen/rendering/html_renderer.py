"""
HTML rendering and screenshot capture using Playwright.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

from PIL import Image
from playwright.async_api import async_playwright

from responsive_gen.io.sketch_loader import ArtifactManager
from responsive_gen.models import (
    DeviceType,
    GeneratedHTML,
    RenderedOutput,
    ViewportConfig,
)


class HTMLRenderer:
    """Renders HTML files and captures screenshots at different viewports."""

    def __init__(
        self,
        headless: bool = True,
        browser_type: str = "chromium"
    ):
        """
        Initialize the renderer.

        Args:
            headless: Whether to run browser in headless mode.
            browser_type: Browser to use (chromium, firefox, webkit).
        """
        self.headless = headless
        self.browser_type = browser_type

    async def _render_viewport_async(
        self,
        html_path: Path,
        viewport: ViewportConfig,
        output_path: Path,
        wait_time: int = 1000
    ):
        """
        Render HTML at a specific viewport and capture screenshot (async).

        Args:
            html_path: Path to HTML file.
            viewport: Viewport configuration.
            output_path: Where to save screenshot.
            wait_time: Time to wait for rendering (ms).
        """
        async with async_playwright() as p:
            # Launch browser
            if self.browser_type == "chromium":
                browser = await p.chromium.launch(headless=self.headless)
            elif self.browser_type == "firefox":
                browser = await p.firefox.launch(headless=self.headless)
            elif self.browser_type == "webkit":
                browser = await p.webkit.launch(headless=self.headless)
            else:
                raise ValueError(f"Unsupported browser: {self.browser_type}")

            # Create context with viewport
            context = await browser.new_context(
                viewport={
                    "width": viewport.width,
                    "height": viewport.height
                },
                device_scale_factor=1
            )

            # Create page and navigate
            page = await context.new_page()

            # Load HTML file
            html_url = f"file://{html_path.absolute()}"
            await page.goto(html_url, wait_until="networkidle")

            # Wait for rendering
            await page.wait_for_timeout(wait_time)

            # Capture full page screenshot
            await page.screenshot(
                path=str(output_path),
                full_page=True
            )

            # Clean up
            await context.close()
            await browser.close()

    def render_viewport(
        self,
        html_path: Union[str, Path],
        viewport: ViewportConfig,
        output_path: Union[str, Path],
        wait_time: int = 1000
    ) -> Path:
        """
        Render HTML at a specific viewport and capture screenshot (sync wrapper).

        Args:
            html_path: Path to HTML file.
            viewport: Viewport configuration.
            output_path: Where to save screenshot.
            wait_time: Time to wait for rendering (ms).

        Returns:
            Path to saved screenshot.
        """
        html_path = Path(html_path)
        output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Run async rendering
        asyncio.run(self._render_viewport_async(html_path, viewport, output_path, wait_time))

        return output_path

    async def _render_all_viewports_async(
        self,
        html_path: Path,
        output_dir: Path,
        sample_id: str,
        wait_time: int = 1000
    ) -> RenderedOutput:
        """
        Render HTML at all three viewports (async).

        Args:
            html_path: Path to HTML file.
            output_dir: Output directory for screenshots.
            sample_id: Sample identifier.
            wait_time: Time to wait for rendering (ms).

        Returns:
            RenderedOutput object.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        viewports = {
            DeviceType.MOBILE: ViewportConfig.mobile(),
            DeviceType.TABLET: ViewportConfig.tablet(),
            DeviceType.DESKTOP: ViewportConfig.desktop()
        }

        screenshot_paths = {}

        async with async_playwright() as p:
            # Launch browser once
            if self.browser_type == "chromium":
                browser = await p.chromium.launch(headless=self.headless)
            elif self.browser_type == "firefox":
                browser = await p.firefox.launch(headless=self.headless)
            elif self.browser_type == "webkit":
                browser = await p.webkit.launch(headless=self.headless)
            else:
                raise ValueError(f"Unsupported browser: {self.browser_type}")

            html_url = f"file://{html_path.absolute()}"

            # Render each viewport
            for device_type, viewport in viewports.items():
                # Create context with viewport
                context = await browser.new_context(
                    viewport={
                        "width": viewport.width,
                        "height": viewport.height
                    },
                    device_scale_factor=1
                )

                page = await context.new_page()
                await page.goto(html_url, wait_until="networkidle")
                await page.wait_for_timeout(wait_time)

                # Save screenshot
                screenshot_path = output_dir / f"{device_type.value}.png"
                await page.screenshot(
                    path=str(screenshot_path),
                    full_page=True
                )

                screenshot_paths[device_type] = screenshot_path

                await context.close()

            await browser.close()

        return RenderedOutput(
            sample_id=sample_id,
            mobile_screenshot=screenshot_paths.get(DeviceType.MOBILE),
            tablet_screenshot=screenshot_paths.get(DeviceType.TABLET),
            desktop_screenshot=screenshot_paths.get(DeviceType.DESKTOP),
            rendering_timestamp=datetime.now()
        )

    def render_all_viewports(
        self,
        html_path: Union[str, Path],
        output_dir: Union[str, Path],
        sample_id: str,
        wait_time: int = 1000
    ) -> RenderedOutput:
        """
        Render HTML at all three viewports (sync wrapper).

        Args:
            html_path: Path to HTML file.
            output_dir: Output directory for screenshots.
            sample_id: Sample identifier.
            wait_time: Time to wait for rendering (ms).

        Returns:
            RenderedOutput object with screenshot paths.
        """
        html_path = Path(html_path)
        output_dir = Path(output_dir)

        return asyncio.run(
            self._render_all_viewports_async(html_path, output_dir, sample_id, wait_time)
        )

    def render_generated_html(
        self,
        generated_html: GeneratedHTML,
        artifact_manager: ArtifactManager,
        wait_time: int = 1000
    ) -> RenderedOutput:
        """
        Render a GeneratedHTML object and save screenshots.

        Args:
            generated_html: GeneratedHTML object.
            artifact_manager: ArtifactManager for saving.
            wait_time: Time to wait for rendering (ms).

        Returns:
            RenderedOutput object.
        """
        # Save HTML first
        html_path = artifact_manager.save_html(
            generated_html.sample_id,
            generated_html.html_content
        )

        # Render at all viewports
        screenshots_dir = html_path.parent / "screenshots"
        return self.render_all_viewports(
            html_path,
            screenshots_dir,
            generated_html.sample_id,
            wait_time
        )


class ScreenshotComparator:
    """Utilities for comparing screenshots."""

    @staticmethod
    def load_screenshot(path: Union[str, Path]) -> Image.Image:
        """
        Load a screenshot image.

        Args:
            path: Path to screenshot.

        Returns:
            PIL Image.
        """
        return Image.open(path).convert("RGB")

    @staticmethod
    def calculate_dimensions_match(
        screenshot: Image.Image,
        expected_width: int,
        tolerance: float = 0.05
    ) -> bool:
        """
        Check if screenshot width matches expected viewport width.

        Args:
            screenshot: Screenshot image.
            expected_width: Expected width.
            tolerance: Allowed relative difference.

        Returns:
            True if within tolerance.
        """
        actual_width = screenshot.width
        diff = abs(actual_width - expected_width) / expected_width
        return diff <= tolerance

    @staticmethod
    def save_side_by_side(
        screenshot1: Image.Image,
        screenshot2: Image.Image,
        output_path: Union[str, Path],
        labels: Optional[tuple] = None
    ):
        """
        Save two screenshots side by side for comparison.

        Args:
            screenshot1: First screenshot.
            screenshot2: Second screenshot.
            output_path: Where to save comparison.
            labels: Optional tuple of (label1, label2).
        """
        from PIL import ImageDraw, ImageFont

        # Calculate dimensions
        max_height = max(screenshot1.height, screenshot2.height)
        total_width = screenshot1.width + screenshot2.width + 30  # 30px gap

        # Create new image
        comparison = Image.new("RGB", (total_width, max_height), color="white")

        # Paste images
        comparison.paste(screenshot1, (0, 0))
        comparison.paste(screenshot2, (screenshot1.width + 30, 0))

        # Add labels if provided
        if labels:
            draw = ImageDraw.Draw(comparison)
            try:
                font = ImageFont.truetype("Arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            draw.text((10, 10), labels[0], fill="black", font=font)
            draw.text((screenshot1.width + 40, 10), labels[1], fill="black", font=font)

        comparison.save(output_path)

