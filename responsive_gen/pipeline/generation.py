"""
LangChain-based generation pipeline for responsive HTML from wireframe sketches.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from langchain.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from responsive_gen.io.sketch_loader import ArtifactManager, SketchLoader
from responsive_gen.models import (
    DeviceType,
    GeneratedHTML,
    SketchTriplet,
)


SYSTEM_PROMPT = """You are an expert frontend developer specializing in responsive web design.
Your task is to generate a single, static HTML file with inline CSS that faithfully reproduces
the layouts shown in three wireframe sketches representing mobile, tablet, and desktop viewports.

Key requirements:
1. Generate ONE complete HTML file with ALL CSS inline in a <style> tag
2. Use CSS media queries to implement responsive behavior at these exact breakpoints:
   - Mobile: max-width: 767px (375px target)
   - Tablet: min-width: 768px and max-width: 1279px (768px target)
   - Desktop: min-width: 1280px (1280px target)
3. NO JavaScript - this must be a static page
4. Faithfully reproduce the structure, layout, and component placement from each sketch
5. Use semantic HTML5 elements
6. Ensure proper stacking/reflow at breakpoints
7. Use placeholder content for text and images

Wireframe conventions:
- Boxes with "X" represent images
- Wavy lines represent text blocks
- Rectangles represent cards, buttons, or containers
- Pay attention to relative positioning and proportions

Generate clean, well-structured HTML that looks professional and matches the wireframes."""


PROMPT_TEMPLATE = """I will provide you with three wireframe sketches of the same webpage at different device sizes:

1. **Mobile** (375px width): {mobile_description}
2. **Tablet** (768px width): {tablet_description}
3. **Desktop** (1280px width): {desktop_description}

Please analyze these wireframes and generate a complete, responsive HTML file that reproduces
these layouts at their respective breakpoints.

The HTML should:
- Start with <!DOCTYPE html> and include proper meta tags for responsive design
- Include all CSS inline within a <style> tag
- Use CSS Grid, Flexbox, or other modern layout techniques
- Implement proper responsive behavior with media queries
- Use placeholder text (Lorem ipsum) and placeholder images (use colored div placeholders)
- Be production-ready and well-formatted

Generate ONLY the HTML code, no explanations."""


class ResponseParser:
    """Parses LLM responses to extract HTML content."""

    @staticmethod
    def extract_html(response_text: str) -> str:
        """
        Extract HTML content from LLM response.

        Args:
            response_text: Raw LLM response.

        Returns:
            Extracted HTML content.
        """
        # Remove markdown code fences if present
        if "```html" in response_text:
            parts = response_text.split("```html")
            if len(parts) > 1:
                html_part = parts[1].split("```")[0]
                return html_part.strip()
        elif "```" in response_text:
            parts = response_text.split("```")
            if len(parts) >= 3:
                return parts[1].strip()

        # If no code fences, return the full response
        return response_text.strip()

    @staticmethod
    def extract_css(html_content: str) -> str:
        """
        Extract inline CSS from HTML content.

        Args:
            html_content: HTML content with inline CSS.

        Returns:
            Extracted CSS content.
        """
        import re

        # Extract content between <style> tags
        style_pattern = r"<style[^>]*>(.*?)</style>"
        matches = re.findall(style_pattern, html_content, re.DOTALL | re.IGNORECASE)

        if matches:
            return "\n\n".join(matches)

        return ""


class ResponsiveGenerator:
    """Generates responsive HTML from wireframe sketch triplets."""

    def __init__(
        self,
        provider: str = "openai",
        model_name: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        api_key: Optional[str] = None
    ):
        """
        Initialize the generator.

        Args:
            provider: LLM provider (openai or anthropic).
            model_name: Model name (optional, uses defaults).
            temperature: Generation temperature.
            max_tokens: Maximum tokens to generate.
            api_key: API key (optional, uses environment variable).
        """
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.parser = ResponseParser()

        # Initialize LLM
        if self.provider == "openai":
            self.model_name = model_name or "gpt-4-vision-preview"
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key or os.getenv("OPENAI_API_KEY")
            )
        elif self.provider == "anthropic":
            self.model_name = model_name or "claude-3-opus-20240229"
            self.llm = ChatAnthropic(
                model=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _create_prompt(
        self,
        triplet_data: Dict[str, Any],
        mobile_desc: str = "See attached image",
        tablet_desc: str = "See attached image",
        desktop_desc: str = "See attached image"
    ) -> List:
        """
        Create prompt messages for the LLM.

        Args:
            triplet_data: Prepared triplet data with base64 images.
            mobile_desc: Description for mobile layout.
            tablet_desc: Description for tablet layout.
            desktop_desc: Description for desktop layout.

        Returns:
            List of messages for the LLM.
        """
        # Build messages with images
        messages = [
            SystemMessage(content=SYSTEM_PROMPT)
        ]

        # Add images and prompt
        if self.provider == "openai":
            # OpenAI format with vision
            content = [
                {
                    "type": "text",
                    "text": PROMPT_TEMPLATE.format(
                        mobile_description=mobile_desc,
                        tablet_description=tablet_desc,
                        desktop_description=desktop_desc
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{triplet_data['images']['mobile']}"
                    }
                },
                {
                    "type": "text",
                    "text": "Mobile wireframe shown above. Tablet wireframe:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{triplet_data['images']['tablet']}"
                    }
                },
                {
                    "type": "text",
                    "text": "Tablet wireframe shown above. Desktop wireframe:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{triplet_data['images']['desktop']}"
                    }
                },
            ]
            messages.append(HumanMessage(content=content))
        else:
            # Anthropic format
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": triplet_data['images']['mobile']
                    }
                },
                {
                    "type": "text",
                    "text": "Mobile wireframe (375px width)"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": triplet_data['images']['tablet']
                    }
                },
                {
                    "type": "text",
                    "text": "Tablet wireframe (768px width)"
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": triplet_data['images']['desktop']
                    }
                },
                {
                    "type": "text",
                    "text": "Desktop wireframe (1280px width)"
                },
                {
                    "type": "text",
                    "text": PROMPT_TEMPLATE.format(
                        mobile_description=mobile_desc,
                        tablet_description=tablet_desc,
                        desktop_description=desktop_desc
                    )
                }
            ]
            messages.append(HumanMessage(content=content))

        return messages

    def generate(
        self,
        sketch_triplet: SketchTriplet,
        sketch_loader: Optional[SketchLoader] = None
    ) -> GeneratedHTML:
        """
        Generate responsive HTML from a sketch triplet.

        Args:
            sketch_triplet: Input sketch triplet.
            sketch_loader: SketchLoader instance (creates one if not provided).

        Returns:
            GeneratedHTML object.
        """
        if sketch_loader is None:
            sketch_loader = SketchLoader()

        # Prepare triplet data
        triplet_data = sketch_loader.prepare_for_llm(sketch_triplet)

        # Create prompt
        messages = self._create_prompt(triplet_data)

        # Generate response
        response = self.llm.invoke(messages)

        # Extract HTML
        html_content = self.parser.extract_html(response.content)
        css_content = self.parser.extract_css(html_content)

        # Extract token usage if available
        prompt_tokens = None
        completion_tokens = None
        if hasattr(response, "usage_metadata"):
            prompt_tokens = response.usage_metadata.get("input_tokens")
            completion_tokens = response.usage_metadata.get("output_tokens")
        elif hasattr(response, "response_metadata"):
            usage = response.response_metadata.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")

        return GeneratedHTML(
            sample_id=sketch_triplet.sample_id,
            html_content=html_content,
            css_content=css_content,
            generation_timestamp=datetime.now(),
            model_name=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            generation_metadata={
                "provider": self.provider,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        )

    def generate_and_save(
        self,
        sketch_triplet: SketchTriplet,
        output_dir: Path = Path("outputs"),
        sketch_loader: Optional[SketchLoader] = None
    ) -> tuple[GeneratedHTML, Path]:
        """
        Generate responsive HTML and save to disk.

        Args:
            sketch_triplet: Input sketch triplet.
            output_dir: Output directory root.
            sketch_loader: SketchLoader instance.

        Returns:
            Tuple of (GeneratedHTML, html_file_path).
        """
        # Generate HTML
        generated = self.generate(sketch_triplet, sketch_loader)

        # Save to disk
        artifact_manager = ArtifactManager(output_dir)
        html_path = artifact_manager.save_html(
            sketch_triplet.sample_id,
            generated.html_content
        )

        # Also save metadata
        metadata_path = html_path.parent / "generation_log.json"
        metadata = {
            "sample_id": generated.sample_id,
            "timestamp": generated.generation_timestamp.isoformat(),
            "model": generated.model_name,
            "prompt_tokens": generated.prompt_tokens,
            "completion_tokens": generated.completion_tokens,
            "metadata": generated.generation_metadata
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return generated, html_path

