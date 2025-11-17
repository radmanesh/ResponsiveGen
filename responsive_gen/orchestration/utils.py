"""
Utility functions for orchestration system.
"""

from typing import Any, Dict, Optional

from responsive_gen.orchestration.state import ResponsiveState


def get_state_summary(state: ResponsiveState) -> str:
    """
    Get a human-readable summary of the current state.

    Args:
        state: Current ResponsiveState

    Returns:
        Formatted string summary
    """
    summary = []
    summary.append(f"Iteration: {state.get('iteration', 0)}")
    summary.append(f"HTML exists: {state.get('html') is not None}")
    summary.append(f"Responsive score: {state.get('responsive_score', 'N/A')}")
    summary.append(f"Next step: {state.get('next_step', 'N/A')}")
    summary.append(f"Active view: {state.get('active_view', 'N/A')}")
    summary.append(f"Focus selector: {state.get('focus_selector', 'N/A')}")

    if state.get('eval_results'):
        summary.append(f"Evaluation results: {len(state['eval_results'])} viewports")

    if state.get('edit_history'):
        summary.append(f"Edit history: {len(state['edit_history'])} edits")

    return "\n".join(summary)


def validate_state(state: ResponsiveState) -> tuple[bool, Optional[str]]:
    """
    Validate that state has required fields for current operation.

    Args:
        state: State to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not state.get('sample_id'):
        return False, "sample_id is required"

    return True, None


def merge_state_updates(current: ResponsiveState, updates: Dict[str, Any]) -> ResponsiveState:
    """
    Merge state updates into current state.

    Args:
        current: Current state
        updates: Updates to apply

    Returns:
        Updated state
    """
    merged = dict(current)
    merged.update(updates)
    return merged

