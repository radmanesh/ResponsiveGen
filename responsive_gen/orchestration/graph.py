"""
LangGraph construction for responsive HTML generation orchestration.

Builds the state graph with all nodes, edges, and conditional routing.
"""

from typing import Literal

from langgraph.graph import END, StateGraph

from responsive_gen.orchestration.agents import (
    editor_node,
    evaluator_node,
    generator_node,
    orchestrator_node,
    reviewer_node,
)
from responsive_gen.orchestration.state import ResponsiveState


def route_orchestrator(state: ResponsiveState) -> Literal["generator", "evaluator", "reviewer", "editor", END]:
    """
    Route function for orchestrator conditional edges.

    Determines next node based on orchestrator's next_step decision.
    """
    next_step = state.get("next_step")

    if next_step == "GENERATE":
        return "generator"
    elif next_step == "EVALUATE":
        return "evaluator"
    elif next_step == "REVIEW_EDIT":
        return "reviewer"
    elif next_step == "EDIT":
        return "editor"
    elif next_step == "FINISH":
        return END
    else:
        # Default: evaluate if we have HTML, otherwise generate
        if state.get("html"):
            return "evaluator"
        else:
            return "generator"


def create_responsive_graph(checkpointer=None):
    """
    Create and compile the responsive HTML generation LangGraph.

    Args:
        checkpointer: Optional checkpointer for state persistence (e.g., SqliteSaver)

    Returns:
        Compiled LangGraph application
    """
    # Create state graph
    graph = StateGraph(ResponsiveState)

    # Add nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("generator", generator_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("editor", editor_node)

    # Set entry point
    graph.set_entry_point("orchestrator")

    # Add conditional edges from orchestrator
    graph.add_conditional_edges(
        "orchestrator",
        route_orchestrator,
        {
            "generator": "generator",
            "evaluator": "evaluator",
            "reviewer": "reviewer",
            "editor": "editor",
            END: END,
        }
    )

    # Add linear chains
    graph.add_edge("generator", "evaluator")  # After generation, always evaluate
    graph.add_edge("reviewer", "editor")  # After review, always edit
    graph.add_edge("editor", "evaluator")  # After edit, re-evaluate

    # Loop back to orchestrator after evaluation (orchestrator decides next step)
    graph.add_edge("evaluator", "orchestrator")

    # Note: The graph will loop until orchestrator returns FINISH

    # Compile graph
    if checkpointer:
        app = graph.compile(checkpointer=checkpointer)
    else:
        app = graph.compile()

    return app


def create_responsive_graph_with_checkpointing(checkpoint_dir: str = ".checkpoints", silent: bool = False):
    """
    Create graph with SqliteSaver checkpointing for persistence.

    Args:
        checkpoint_dir: Directory to store checkpoint database
        silent: If True, suppress warning when SqliteSaver is not available

    Returns:
        Compiled LangGraph application with checkpointing
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        import os
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpointer = SqliteSaver.from_conn_string(f"{checkpoint_dir}/checkpoints.db")
        return create_responsive_graph(checkpointer=checkpointer)
    except ImportError:
        if not silent:
            print("Warning: SqliteSaver not available. Using graph without checkpointing.")
        return create_responsive_graph()


# Convenience function that uses checkpointing by default
def get_responsive_app(checkpoint_dir: str = ".checkpoints", use_checkpointing: bool = True, silent_checkpoint_warning: bool = True):
    """
    Get compiled responsive graph application.

    Args:
        checkpoint_dir: Directory for checkpoint database
        use_checkpointing: Whether to enable checkpointing
        silent_checkpoint_warning: If True, suppress warning when SqliteSaver is not available

    Returns:
        Compiled LangGraph application
    """
    if use_checkpointing:
        return create_responsive_graph_with_checkpointing(checkpoint_dir, silent=silent_checkpoint_warning)
    else:
        return create_responsive_graph()

