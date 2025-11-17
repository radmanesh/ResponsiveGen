"""
Agent node functions for LangGraph orchestration.

Each agent is a node function that takes state, performs actions using tools,
and returns updated state.
"""

import json
import os
from typing import Any, Dict, Literal, Optional

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from responsive_gen.orchestration.state import ResponsiveState
from responsive_gen.orchestration.tools import (
    RESPONSIVE_TOOLS,
    compute_responsive_meter,
    generate_html,
    llm_judge_layout,
    llm_judge_responsiveness,
    modify_html,
    read_html,
    run_iou_evaluation,
    run_perceptual_evaluation,
    take_screenshot,
)
from responsive_gen.utils.llm_logger import LoggedLLM


# Initialize LLM for agents
def get_agent_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    component: str = "orchestrator",
    sample_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Get LLM instance for agents (orchestrator, reviewer, editor).

    Args:
        model_name: Model to use. If None, reads from ORCHESTRATOR_MODEL env var.
        temperature: Generation temperature. If None, reads from ORCHESTRATOR_TEMPERATURE env var.
        component: Component name for logging (default: "orchestrator")
        sample_id: Optional sample ID for tracking
        metadata: Optional additional metadata for logging

    Returns:
        LoggedLLM wrapper around ChatOpenAI or ChatAnthropic instance
    """
    # Load environment variables
    load_dotenv()

    # Get configuration from env vars if not explicitly provided
    provider = os.getenv("ORCHESTRATOR_PROVIDER", "openai").lower()
    model = model_name or os.getenv("ORCHESTRATOR_MODEL", "gpt-4o")
    temp = temperature if temperature is not None else float(os.getenv("ORCHESTRATOR_TEMPERATURE", "0.3"))

    llm_instance = None
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        llm_instance = ChatOpenAI(model=model, temperature=temp, api_key=api_key)
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        llm_instance = ChatAnthropic(model=model, temperature=temp, api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Wrap with logging
    return LoggedLLM(
        llm_instance=llm_instance,
        component=component,
        provider=provider,
        model=model,
        sample_id=sample_id,
        metadata=metadata or {},
    )


ORCHESTRATOR_SYSTEM_PROMPT = """You are the Orchestrator Agent for responsive HTML generation.

Your role is to decide the next step in the workflow based on the current state.

Available steps:
- GENERATE: Generate initial HTML from wireframe sketches (if html is None)
- EVALUATE: Run evaluation metrics on current HTML
- REVIEW_EDIT: Analyze issues and prepare feedback for editing
- EDIT: Apply edits to fix issues
- FINISH: Stop when score is acceptable or max iterations reached

Current state context:
- iteration: Current iteration number
- html: Whether HTML exists
- responsive_score: Current quality score (0.0 to 1.0)
- eval_results: Latest evaluation results
- active_view: Viewport currently being focused on
- focus_selector: CSS selector being edited

Decision rules:
1. If html is None → GENERATE
2. If no eval_results or iteration == 0 → EVALUATE
3. If responsive_score < 0.7 and iteration < 5 → REVIEW_EDIT
4. If feedback exists and edit_target is None → EDIT
5. If responsive_score >= 0.7 or iteration >= 5 → FINISH

Respond with JSON:
{
  "next_step": "GENERATE" | "EVALUATE" | "REVIEW_EDIT" | "EDIT" | "FINISH",
  "reasoning": "brief explanation",
  "active_view": "mobile" | "tablet" | "desktop" | null,
  "focus_selector": "CSS selector" | null
}"""


GENERATOR_SYSTEM_PROMPT = """You are the Generator Agent.

Your role is to generate responsive HTML from wireframe sketches.

You have access to the generate_html tool which takes:
- sample_id: Unique identifier
- sketch_triplet_path: Path to directory with mobile.png, tablet.png, desktop.png

After generating, update the state with the HTML content and path."""


EVALUATOR_SYSTEM_PROMPT = """You are the Evaluator Agent.

Your role is to run comprehensive evaluation metrics on the generated HTML.

Available tools:
- take_screenshot: Capture screenshots at different viewports
- run_iou_evaluation: Compute IoU-based layout similarity
- run_perceptual_evaluation: Compute perceptual similarity (stub)
- llm_judge_layout: LLM-based layout assessment (stub)
- llm_judge_responsiveness: Cross-device consistency check (stub)
- compute_responsive_meter: Aggregate all metrics into composite score

Workflow:
1. Take screenshots for mobile, tablet, desktop
2. Run IoU evaluation for each viewport
3. (Optional) Run perceptual and LLM judge evaluations
4. Compute composite ResponsiveMeter score
5. Update state with all results"""


REVIEWER_SYSTEM_PROMPT = """You are the Reviewer Agent.

Your role is to analyze evaluation results and identify specific issues to fix.

You have access to:
- read_html: Inspect HTML fragments by CSS selector
- eval_results: Latest evaluation metrics
- responsive_score: Overall quality score

Workflow:
1. Analyze eval_results to find worst-performing viewport/component
2. Use read_html to inspect problematic HTML sections
3. Identify specific issues (e.g., "Hero text overlaps image", "Button too close to margin")
4. Provide actionable suggestions
5. Set active_view and focus_selector for the Editor

Respond with structured feedback:
{
  "view": "mobile" | "tablet" | "desktop",
  "selector": "CSS selector",
  "issues": ["issue1", "issue2"],
  "suggestions": ["suggestion1", "suggestion2"]
}"""


EDITOR_SYSTEM_PROMPT = """You are the Editor Agent.

Your role is to apply targeted HTML edits to fix identified issues.

You have access to:
- read_html: Read current HTML fragment
- modify_html: Replace HTML fragment with improved version

Workflow:
1. Read current HTML fragment using focus_selector
2. Analyze feedback and suggestions
3. Generate improved HTML fragment that addresses issues
4. Use modify_html to apply the change
5. Update state with new HTML

Important:
- Keep IDs and class names consistent
- Maintain semantic HTML structure
- Only modify the targeted selector
- Ensure responsive behavior is preserved"""


def orchestrator_node(state: ResponsiveState) -> Dict[str, Any]:
    """Orchestrator agent that routes workflow."""
    sample_id = state.get('sample_id')
    iteration = state.get('iteration', 0)
    llm = get_agent_llm(
        component="orchestrator",
        sample_id=sample_id,
        metadata={"iteration": iteration, "agent_type": "orchestrator"}
    )
    llm_with_tools = llm.bind_tools([])  # No tools needed for routing

    # Build context message
    context = f"""Current State:
- iteration: {state.get('iteration', 0)}
- html exists: {state.get('html') is not None}
- responsive_score: {state.get('responsive_score')}
- active_view: {state.get('active_view')}
- focus_selector: {state.get('focus_selector')}
- has_feedback: {state.get('feedback') is not None}
- has_edit_target: {state.get('edit_target') is not None}
"""

    messages = [
        SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
        HumanMessage(content=context)
    ]

    response = llm_with_tools.invoke(messages)

    # Parse response
    try:
        # Try to extract JSON from response
        content = response.content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "{" in content:
            json_str = content[content.index("{"):content.rindex("}")+1]
        else:
            json_str = content

        decision = json.loads(json_str)
    except:
        # Fallback to simple logic
        iteration = state.get('iteration', 0)
        score = state.get('responsive_score')
        max_iterations = 5  # Default max iterations

        if state.get('html') is None:
            decision = {"next_step": "GENERATE"}
        elif score is None or iteration == 0:
            decision = {"next_step": "EVALUATE"}
        elif iteration >= max_iterations:
            decision = {"next_step": "FINISH"}
        elif score and score >= 0.7:
            decision = {"next_step": "FINISH"}
        elif score and score < 0.7 and iteration < max_iterations:
            if state.get('feedback') and not state.get('edit_target'):
                decision = {"next_step": "EDIT"}
            else:
                decision = {"next_step": "REVIEW_EDIT"}
        else:
            decision = {"next_step": "FINISH"}

    # Return state updates
    updates = {
        "next_step": decision.get("next_step", "FINISH")
    }
    if "active_view" in decision:
        updates["active_view"] = decision["active_view"]
    if "focus_selector" in decision:
        updates["focus_selector"] = decision["focus_selector"]

    return updates


def generator_node(state: ResponsiveState) -> Dict[str, Any]:
    """Generator agent that creates initial HTML."""
    if not state.get('sample_id') or not state.get('sketch_triplet'):
        raise ValueError("sample_id and sketch_triplet required for generation")

    try:
        # Get sketch triplet path
        sketch_triplet = state['sketch_triplet']
        sketch_dir = sketch_triplet.mobile.image_path.parent

        # Generate HTML
        result = generate_html.invoke({
            "sample_id": state['sample_id'],
            "sketch_triplet_path": str(sketch_dir)
        })

        # Return state updates
        return {
            "html": result["html"],
            "html_path": result["html_path"],
            "iteration": 0,
            "edit_history": []
        }
    except Exception as e:
        print(f"Error in generator_node: {e}")
        raise


def evaluator_node(state: ResponsiveState) -> Dict[str, Any]:
    """Evaluator agent that runs all metrics."""
    if not state.get('html'):
        raise ValueError("HTML required for evaluation")

    # Take screenshots for all viewports
    screenshots = {}
    for view in ["mobile", "tablet", "desktop"]:
        try:
            screenshot_path = take_screenshot.invoke({
                "html_source": state.get('html'),
                "view": view
            })
            screenshots[view] = screenshot_path
        except Exception as e:
            print(f"Warning: Failed to take {view} screenshot: {e}")
            screenshots[view] = None

    # Prepare updates
    updates = {
        "screenshots": screenshots
    }

    # Run IoU evaluation for each viewport
    iou_metrics = {}
    eval_results = {}

    for view in ["mobile", "tablet", "desktop"]:
        try:
            result = run_iou_evaluation.invoke({
                "sample_id": state.get('sample_id') or "unknown",
                "html_path": state.get('html_path') or state.get('html'),
                "view": view,
                "ground_truth_path": None  # TODO: Add ground truth support
            })
            iou_metrics[view] = result["iou_score"]
            eval_results[view] = result
        except Exception as e:
            print(f"Warning: Failed IoU evaluation for {view}: {e}")
            iou_metrics[view] = 0.0
            eval_results[view] = {"iou_score": 0.0}

    # Compute composite score
    try:
        composite_score = compute_responsive_meter.invoke({
            "iou_metrics": iou_metrics,
            "perceptual_metrics": None,
            "llm_judge_metrics": None
        })
        updates["responsive_score"] = composite_score
    except Exception as e:
        print(f"Warning: Failed to compute composite score: {e}")
        updates["responsive_score"] = sum(iou_metrics.values()) / 3.0

    # Update eval results and iteration
    updates["eval_results"] = eval_results
    updates["iteration"] = state.get("iteration", 0) + 1

    return updates


def reviewer_node(state: ResponsiveState) -> Dict[str, Any]:
    """Reviewer agent that analyzes issues and provides feedback."""
    if not state.get('eval_results'):
        raise ValueError("Evaluation results required for review")

    sample_id = state.get('sample_id')
    iteration = state.get('iteration', 0)
    llm = get_agent_llm(
        component="reviewer",
        sample_id=sample_id,
        metadata={"iteration": iteration, "agent_type": "reviewer"}
    )
    llm_with_tools = llm.bind_tools([read_html])

    # Find worst-performing viewport
    worst_view = "mobile"
    worst_score = 1.0
    for view, results in state.get('eval_results', {}).items():
        score = results.get("iou_score", 0.0)
        if score < worst_score:
            worst_score = score
            worst_view = view

    updates = {
        "active_view": worst_view
    }

    # Build context
    context = f"""Evaluation Results:
{json.dumps(state.get('eval_results', {}), indent=2)}

Current HTML exists: {state.get('html') is not None}
Worst performing viewport: {worst_view} (score: {worst_score:.3f})
Responsive score: {state.get('responsive_score')}
"""

    messages = [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(content=context)
    ]

    response = llm_with_tools.invoke(messages)

    # Parse feedback
    try:
        content = response.content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "{" in content:
            json_str = content[content.index("{"):content.rindex("}")+1]
        else:
            json_str = content

        feedback = json.loads(json_str)
    except:
        # Fallback feedback
        feedback = {
            "view": worst_view,
            "selector": "body",
            "issues": [f"Low IoU score ({worst_score:.3f}) in {worst_view} viewport"],
            "suggestions": ["Review layout structure", "Check responsive breakpoints"]
        }

    updates["feedback"] = feedback
    updates["focus_selector"] = feedback.get("selector", "body")

    # Add to edit history
    edit_history = state.get("edit_history", [])
    edit_history.append({
        "iteration": state.get("iteration", 0),
        "type": "review",
        "feedback": feedback
    })
    updates["edit_history"] = edit_history

    return updates


def editor_node(state: ResponsiveState) -> Dict[str, Any]:
    """Editor agent that applies targeted HTML edits."""
    if not state.get('html') or not state.get('focus_selector') or not state.get('feedback'):
        raise ValueError("HTML, focus_selector, and feedback required for editing")

    sample_id = state.get('sample_id')
    iteration = state.get('iteration', 0)
    focus_selector = state.get('focus_selector')
    llm = get_agent_llm(
        component="editor",
        sample_id=sample_id,
        metadata={
            "iteration": iteration,
            "agent_type": "editor",
            "focus_selector": focus_selector
        }
    )
    llm_with_tools = llm.bind_tools([read_html, modify_html])

    # Read current HTML fragment
    try:
        current_fragment = read_html.invoke({
            "html_source": state.get('html'),
            "selector": state.get('focus_selector')
        })

        # If selector not found, try reading full HTML as fallback
        if current_fragment.startswith("<!-- Selector"):
            print(f"Warning: Selector '{state.get('focus_selector')}' not found, using full HTML")
            current_fragment = read_html.invoke({
                "html_source": state.get('html'),
                "selector": None  # Get full HTML
            })
    except Exception as e:
        print(f"Warning: Failed to read HTML fragment: {e}")
        # Try to get full HTML as fallback
        try:
            current_fragment = read_html.invoke({
                "html_source": state.get('html'),
                "selector": None
            })
        except:
            return {}

    # Build context
    context = f"""Current HTML Fragment:
{current_fragment}

Feedback:
{json.dumps(state.get('feedback', {}), indent=2)}

Task: Generate an improved version of the HTML fragment that addresses the issues and follows the suggestions.
Keep IDs, classes, and structure consistent. Only modify what's necessary."""

    messages = [
        SystemMessage(content=EDITOR_SYSTEM_PROMPT),
        HumanMessage(content=context)
    ]

    response = llm_with_tools.invoke(messages)

    # Extract new fragment from response
    try:
        content = response.content
        # Try to extract HTML from code blocks
        if "```html" in content:
            new_fragment = content.split("```html")[1].split("```")[0].strip()
        elif "```" in content:
            new_fragment = content.split("```")[1].split("```")[0].strip()
        else:
            new_fragment = content.strip()
    except:
        print("Warning: Failed to extract new HTML fragment from response")
        return {}

    # Apply edit
    try:
        updated_html = modify_html.invoke({
            "html_source": state["html"],
            "selector": state.get("focus_selector", "body"),
            "new_fragment": new_fragment
        })

        # Add to edit history
        edit_history = state.get("edit_history", [])
        edit_history.append({
            "iteration": state.get("iteration", 0),
            "type": "edit",
            "selector": state.get("focus_selector", "body"),
            "change_summary": f"Modified {state.get('focus_selector', 'body')} based on feedback"
        })

        return {
            "html": updated_html,
            "edit_target": new_fragment,
            "edit_history": edit_history
        }
    except ValueError as ve:
        # If selector not found, try fallback strategies
        selector = state.get("focus_selector", "body")
        if "not found" in str(ve).lower():
            print(f"Warning: Selector '{selector}' not found, attempting fallback strategies")
            # If the selector was 'body' and not found, try to append new content to html or replace entire html
            if selector == "body":
                soup = BeautifulSoup(state["html"], 'html.parser')
                # Try to find or create body tag
                body = soup.find('body')
                if not body:
                    html_tag = soup.find('html')
                    if html_tag:
                        body = soup.new_tag('body')
                        html_tag.append(body)
                if body:
                    # Replace body content with new fragment
                    new_soup = BeautifulSoup(new_fragment, 'html.parser')
                    body.clear()
                    for child in new_soup.children:
                        if hasattr(child, 'name') or (hasattr(child, 'strip') and child.strip()):
                            body.append(child)
                    updated_html = str(soup)

                    # Add to edit history
                    edit_history = state.get("edit_history", [])
                    edit_history.append({
                        "iteration": state.get("iteration", 0),
                        "type": "edit",
                        "selector": selector,
                        "change_summary": f"Modified {selector} based on feedback (created body tag)"
                    })

                    return {
                        "html": updated_html,
                        "edit_target": new_fragment,
                        "edit_history": edit_history
                    }
        # If all fallbacks fail, return empty dict
        print(f"Warning: Failed to apply edit: {ve}")
        return {}
    except Exception as e:
        print(f"Warning: Failed to apply edit: {e}")
        return {}

