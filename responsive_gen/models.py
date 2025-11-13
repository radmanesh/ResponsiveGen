"""
Data models and schemas for responsive webpage generation pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field


class DeviceType(str, Enum):
    """Device viewport types."""
    MOBILE = "mobile"
    TABLET = "tablet"
    DESKTOP = "desktop"


class ViewportConfig(BaseModel):
    """Viewport configuration for a specific device type."""
    device_type: DeviceType
    width: int
    height: int

    @classmethod
    def mobile(cls, height: int = 688) -> "ViewportConfig":
        return cls(device_type=DeviceType.MOBILE, width=375, height=height)

    @classmethod
    def tablet(cls, height: int = 1024) -> "ViewportConfig":
        return cls(device_type=DeviceType.TABLET, width=768, height=height)

    @classmethod
    def desktop(cls, height: int = 800) -> "ViewportConfig":
        return cls(device_type=DeviceType.DESKTOP, width=1280, height=height)


class SketchInput(BaseModel):
    """Input wireframe sketch for a specific viewport."""
    device_type: DeviceType
    image_path: Path
    viewport: ViewportConfig

    class Config:
        arbitrary_types_allowed = True


class SketchTriplet(BaseModel):
    """Collection of three wireframe sketches representing responsive breakpoints."""
    sample_id: str
    mobile: SketchInput
    tablet: SketchInput
    desktop: SketchInput
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_sketch(self, device_type: DeviceType) -> SketchInput:
        """Get sketch for specific device type."""
        if device_type == DeviceType.MOBILE:
            return self.mobile
        elif device_type == DeviceType.TABLET:
            return self.tablet
        elif device_type == DeviceType.DESKTOP:
            return self.desktop
        else:
            raise ValueError(f"Unknown device type: {device_type}")

    class Config:
        arbitrary_types_allowed = True


class GeneratedHTML(BaseModel):
    """Generated responsive HTML output."""
    sample_id: str
    html_content: str
    css_content: str  # Extracted inline CSS if needed
    generation_timestamp: datetime = Field(default_factory=datetime.now)
    model_name: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class RenderedOutput(BaseModel):
    """Screenshots of rendered HTML at different viewports."""
    sample_id: str
    mobile_screenshot: Optional[Path] = None
    tablet_screenshot: Optional[Path] = None
    desktop_screenshot: Optional[Path] = None
    rendering_timestamp: datetime = Field(default_factory=datetime.now)
    rendering_metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class ComponentType(str, Enum):
    """Visual component types for layout analysis."""
    TEXT = "text"
    IMAGE = "image"
    NAVIGATION = "navigation"
    BUTTON = "button"
    FORM = "form"
    TABLE = "table"
    DIVIDER = "divider"
    CARD = "card"
    TERTIARY = "tertiary"  # Other minor components


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x: float
    y: float
    width: float
    height: float

    def area(self) -> float:
        """Calculate bounding box area."""
        return self.width * self.height

    def iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection over Union with another bounding box."""
        # Calculate intersection coordinates
        x_left = max(self.x, other.x)
        y_top = max(self.y, other.y)
        x_right = min(self.x + self.width, other.x + other.width)
        y_bottom = min(self.y + self.height, other.y + other.height)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        union_area = self.area() + other.area() - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0


class LayoutComponent(BaseModel):
    """A visual component detected in a layout."""
    component_type: ComponentType
    bounding_box: BoundingBox
    confidence: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LayoutAnalysis(BaseModel):
    """Layout analysis result for a single viewport."""
    device_type: DeviceType
    components: List[LayoutComponent] = Field(default_factory=list)
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict)


class IoUMetrics(BaseModel):
    """IoU-based layout similarity metrics."""
    mobile_iou: float
    tablet_iou: float
    desktop_iou: float
    average_iou: float
    per_component_iou: Dict[ComponentType, float] = Field(default_factory=dict)


class PerceptualMetrics(BaseModel):
    """Perceptual similarity metrics."""
    clip_similarity_mobile: float = 0.0
    clip_similarity_tablet: float = 0.0
    clip_similarity_desktop: float = 0.0
    block_similarity: float = 0.0
    text_region_similarity: float = 0.0
    positional_alignment: float = 0.0


class LLMJudgeScore(BaseModel):
    """LLM-as-a-judge evaluation scores."""
    layout_accuracy_mobile: float = 0.0
    layout_accuracy_tablet: float = 0.0
    layout_accuracy_desktop: float = 0.0
    visual_hierarchy_score: float = 0.0
    cross_device_consistency: float = 0.0
    overall_score: float = 0.0
    feedback: str = ""


class ResponsiveMeterScore(BaseModel):
    """Composite responsive webpage quality score."""
    iou_metrics: IoUMetrics
    perceptual_metrics: PerceptualMetrics
    llm_judge_score: LLMJudgeScore

    # Weights
    w1: float = 0.35  # IoU weight
    w2: float = 0.25  # Cross-device consistency weight
    w3: float = 0.25  # LLM judge weight
    w4: float = 0.15  # Perceptual weight

    composite_score: float = 0.0

    def calculate_composite(self) -> float:
        """Calculate weighted composite score."""
        self.composite_score = (
            self.w1 * self.iou_metrics.average_iou +
            self.w2 * self.llm_judge_score.cross_device_consistency +
            self.w3 * self.llm_judge_score.overall_score +
            self.w4 * (
                (self.perceptual_metrics.clip_similarity_mobile +
                 self.perceptual_metrics.clip_similarity_tablet +
                 self.perceptual_metrics.clip_similarity_desktop) / 3.0
            )
        )
        return self.composite_score


class EvaluationResult(BaseModel):
    """Complete evaluation result for a generated webpage."""
    sample_id: str
    responsive_meter: ResponsiveMeterScore
    evaluation_timestamp: datetime = Field(default_factory=datetime.now)
    evaluation_metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class PipelineArtifact(BaseModel):
    """Complete pipeline output artifact."""
    sample_id: str
    sketch_triplet: SketchTriplet
    generated_html: GeneratedHTML
    rendered_output: RenderedOutput
    evaluation_result: Optional[EvaluationResult] = None
    output_directory: Path

    class Config:
        arbitrary_types_allowed = True

