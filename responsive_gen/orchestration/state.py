"""
State management for LangGraph orchestration.

Defines ResponsiveState as a TypedDict with messages and custom fields
for HTML content, evaluation results, edit history, and workflow control.
"""

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from responsive_gen.models import SketchTriplet


class ResponsiveState(TypedDict, total=False):
    """
    State for responsive HTML generation orchestration.

    Includes messages for chat memory and custom fields for HTML content,
    evaluation results, edit history, and workflow control.

    All fields are optional (total=False) to allow incremental state updates.
    """

    # Messages for chat memory
    messages: Annotated[List[BaseMessage], add_messages]

    # HTML content
    html: Optional[str]
    html_path: Optional[str]

    # Evaluation results
    eval_results: Dict[str, Any]
    responsive_score: Optional[float]

    # Edit history
    edit_history: List[Dict[str, Any]]

    # Workflow control
    active_view: Optional[str]  # "mobile" | "tablet" | "desktop"
    focus_selector: Optional[str]  # CSS selector being edited
    iteration: int
    next_step: Optional[str]  # "GENERATE" | "EVALUATE" | "REVIEW_EDIT" | "EDIT" | "FINISH"

    # Sample metadata
    sample_id: Optional[str]
    sketch_triplet: Optional[SketchTriplet]

    # Screenshot paths
    screenshots: Dict[str, Optional[str]]  # {"mobile": path, "tablet": path, "desktop": path}

    # Feedback and suggestions
    feedback: Optional[Dict[str, Any]]  # From reviewer agent
    edit_target: Optional[str]  # HTML fragment to replace current selector



