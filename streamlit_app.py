"""
Streamlit web interface for Responsive Web Generation pipeline.

Provides drag-and-drop interface for uploading wireframe sketches and
visualizing generated responsive HTML with evaluation metrics.
"""

import os
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from responsive_gen.evaluation.responsive_meter import ResponsiveMeter
from responsive_gen.io.sketch_loader import ArtifactManager, SketchLoader
from responsive_gen.models import EvaluationResult
from responsive_gen.models import DeviceType, ViewportConfig
from responsive_gen.orchestration.graph import get_responsive_app
from responsive_gen.orchestration.state import ResponsiveState
from responsive_gen.pipeline.generation import ResponsiveGenerator
from responsive_gen.rendering.html_renderer import HTMLRenderer
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Responsive Web Generation",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .device-label {
        font-weight: 600;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "generated_html" not in st.session_state:
    st.session_state.generated_html = None
if "rendered_output" not in st.session_state:
    st.session_state.rendered_output = None
if "sample_id" not in st.session_state:
    st.session_state.sample_id = None
if "sketch_triplet" not in st.session_state:
    st.session_state.sketch_triplet = None
if "evaluation_result" not in st.session_state:
    st.session_state.evaluation_result = None
if "orchestration_thread_id" not in st.session_state:
    st.session_state.orchestration_thread_id = None
if "orchestration_app" not in st.session_state:
    st.session_state.orchestration_app = None


def main():
    """Main application entry point."""

    # Header
    st.markdown('<div class="main-header">🖼️ Responsive Web Generation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Generate responsive HTML from wireframe sketch triplets</div>',
        unsafe_allow_html=True
    )

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # LLM Settings
        st.subheader("LLM Provider")
        provider = st.selectbox(
            "Provider",
            ["openai", "anthropic"],
            index=0,
            help="Select the LLM provider for generation"
        )

        if provider == "openai":
            model_options = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
            default_model = "gpt-4o"
        else:
            model_options = ["claude-3-opus-20240229", "claude-3-sonnet-20240229"]
            default_model = "claude-3-opus-20240229"

        model_name = st.selectbox("Model", model_options, index=0)

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Lower = more deterministic, Higher = more creative"
        )

        max_tokens = st.number_input(
            "Max Tokens",
            min_value=1024,
            max_value=8192,
            value=4096,
            step=512
        )

        st.divider()

        # Evaluation Settings
        st.subheader("Evaluation Weights")
        w1 = st.slider("IoU Weight", 0.0, 1.0, 0.35, 0.05)
        w2 = st.slider("Consistency Weight", 0.0, 1.0, 0.25, 0.05)
        w3 = st.slider("LLM Judge Weight", 0.0, 1.0, 0.25, 0.05)
        w4 = st.slider("Perceptual Weight", 0.0, 1.0, 0.15, 0.05)

        st.divider()

        # Rendering Settings
        st.subheader("Rendering")
        render_wait = st.number_input(
            "Wait Time (ms)",
            min_value=500,
            max_value=5000,
            value=1000,
            step=500,
            help="Time to wait for page rendering"
        )

        st.divider()

        # Output Settings
        output_dir = st.text_input(
            "Output Directory",
            value="outputs",
            help="Directory to save generated files"
        )

    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Upload & Generate",
        "🖥️ Results",
        "📊 Evaluation",
        "🤖 Orchestration",
        "ℹ️ About"
    ])

    with tab1:
        upload_and_generate_tab(
            provider, model_name, temperature, max_tokens,
            output_dir, render_wait
        )

    with tab2:
        results_tab()

    with tab3:
        evaluation_tab(w1, w2, w3, w4, provider, model_name)

    with tab4:
        orchestration_tab(provider, model_name)

    with tab5:
        about_tab()


def upload_and_generate_tab(
    provider, model_name, temperature, max_tokens,
    output_dir, render_wait
):
    """Upload sketches and generate responsive HTML."""

    st.header("Upload Wireframe Sketches")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="device-label">📱 Mobile (375px)</div>', unsafe_allow_html=True)
        mobile_file = st.file_uploader(
            "Upload mobile wireframe",
            type=["png", "jpg", "jpeg"],
            key="mobile_upload",
            label_visibility="collapsed"
        )
        if mobile_file:
            st.image(mobile_file, caption="Mobile Wireframe", width='stretch')

    with col2:
        st.markdown('<div class="device-label">💻 Tablet (768px)</div>', unsafe_allow_html=True)
        tablet_file = st.file_uploader(
            "Upload tablet wireframe",
            type=["png", "jpg", "jpeg"],
            key="tablet_upload",
            label_visibility="collapsed"
        )
        if tablet_file:
            st.image(tablet_file, caption="Tablet Wireframe", width='stretch')

    with col3:
        st.markdown('<div class="device-label">🖥️ Desktop (1280px)</div>', unsafe_allow_html=True)
        desktop_file = st.file_uploader(
            "Upload desktop wireframe",
            type=["png", "jpg", "jpeg"],
            key="desktop_upload",
            label_visibility="collapsed"
        )
        if desktop_file:
            st.image(desktop_file, caption="Desktop Wireframe", width='stretch')

    st.divider()

    # Sample ID input
    sample_id = st.text_input(
        "Sample ID (optional)",
        value="",
        placeholder=f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Identifier for this generation run"
    )

    # Generate button
    if st.button("🚀 Generate Responsive HTML", type="primary"):
        if not all([mobile_file, tablet_file, desktop_file]):
            st.error("❌ Please upload all three wireframe sketches (mobile, tablet, desktop)")
            return

        # Generate sample ID if not provided
        if not sample_id:
            sample_id = f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        st.session_state.sample_id = sample_id

        try:
            with st.spinner("🔄 Generating responsive HTML..."):
                # Save uploaded files temporarily
                with TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)

                    mobile_path = temp_path / "mobile.png"
                    tablet_path = temp_path / "tablet.png"
                    desktop_path = temp_path / "desktop.png"

                    mobile_path.write_bytes(mobile_file.getvalue())
                    tablet_path.write_bytes(tablet_file.getvalue())
                    desktop_path.write_bytes(desktop_file.getvalue())

                    # Load sketch triplet
                    sketch_loader = SketchLoader()
                    triplet = sketch_loader.load_triplet(
                        mobile_path, tablet_path, desktop_path,
                        sample_id=sample_id
                    )

                    # Validate sketches
                    is_valid, error_msg = sketch_loader.validate_sketches(triplet)
                    if not is_valid:
                        st.error(f"❌ Sketch validation failed: {error_msg}")
                        return

                    # Generate HTML
                    generator = ResponsiveGenerator(
                        provider=provider,
                        model_name=model_name,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )

                    generated, html_path = generator.generate_and_save(
                        triplet,
                        output_dir=Path(output_dir),
                        sketch_loader=sketch_loader
                    )

                    st.session_state.generated_html = generated
                    st.session_state.sketch_triplet = triplet

                    st.success(f"✅ HTML generated successfully!")
                    st.info(f"📁 Saved to: {html_path}")

                    # Render screenshots
                    with st.spinner("📸 Rendering screenshots..."):
                        renderer = HTMLRenderer(headless=True)
                        artifact_manager = ArtifactManager(output_dir)

                        rendered = renderer.render_generated_html(
                            generated,
                            artifact_manager,
                            wait_time=render_wait
                        )

                        st.session_state.rendered_output = rendered
                        st.success("✅ Screenshots captured!")

            # Show quick preview
            st.divider()
            st.subheader("Quick Preview")

            if st.session_state.rendered_output:
                preview_col1, preview_col2, preview_col3 = st.columns(3)

                with preview_col1:
                    if st.session_state.rendered_output.mobile_screenshot:
                        st.image(
                            str(st.session_state.rendered_output.mobile_screenshot),
                            caption="Mobile",
                            width='stretch'
                        )

                with preview_col2:
                    if st.session_state.rendered_output.tablet_screenshot:
                        st.image(
                            str(st.session_state.rendered_output.tablet_screenshot),
                            caption="Tablet",
                            width='stretch'
                        )

                with preview_col3:
                    if st.session_state.rendered_output.desktop_screenshot:
                        st.image(
                            str(st.session_state.rendered_output.desktop_screenshot),
                            caption="Desktop",
                            width='stretch'
                        )

        except Exception as e:
            st.error(f"❌ Error during generation: {str(e)}")
            st.exception(e)


def results_tab():
    """Display generation results."""

    st.header("Generated Results")

    if not st.session_state.generated_html:
        st.info("👆 Upload wireframes and generate HTML in the 'Upload & Generate' tab first.")
        return

    generated = st.session_state.generated_html
    rendered = st.session_state.rendered_output

    # Metadata
    st.subheader("📋 Generation Metadata")
    meta_col1, meta_col2, meta_col3 = st.columns(3)

    with meta_col1:
        st.metric("Sample ID", generated.sample_id)
        st.metric("Model", generated.model_name)

    with meta_col2:
        st.metric("Timestamp", generated.generation_timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        if generated.prompt_tokens:
            st.metric("Prompt Tokens", f"{generated.prompt_tokens:,}")

    with meta_col3:
        if generated.completion_tokens:
            st.metric("Completion Tokens", f"{generated.completion_tokens:,}")
            total_tokens = generated.prompt_tokens + generated.completion_tokens
            st.metric("Total Tokens", f"{total_tokens:,}")

    st.divider()

    # Screenshots
    st.subheader("📸 Rendered Screenshots")

    if rendered:
        screenshot_col1, screenshot_col2, screenshot_col3 = st.columns(3)

        with screenshot_col1:
            st.markdown("**📱 Mobile (375px)**")
            if rendered.mobile_screenshot:
                st.image(str(rendered.mobile_screenshot), width='stretch')

        with screenshot_col2:
            st.markdown("**💻 Tablet (768px)**")
            if rendered.tablet_screenshot:
                st.image(str(rendered.tablet_screenshot), width='stretch')

        with screenshot_col3:
            st.markdown("**🖥️ Desktop (1280px)**")
            if rendered.desktop_screenshot:
                st.image(str(rendered.desktop_screenshot), width='stretch')

    st.divider()

    # HTML Source Code
    st.subheader("💻 Generated HTML Source")

    with st.expander("View HTML Source Code", expanded=False):
        st.code(generated.html_content, language="html", line_numbers=True)

    # Download button
    st.download_button(
        label="⬇️ Download HTML",
        data=generated.html_content,
        file_name=f"{generated.sample_id}.html",
        mime="text/html",
        key="download_html_results_tab"
    )


def evaluation_tab(w1, w2, w3, w4, provider, model_name):
    """Display evaluation metrics."""

    st.header("📊 Evaluation Metrics")

    # Check prerequisites
    if not st.session_state.generated_html or not st.session_state.rendered_output:
        st.info("👆 Generate HTML first to see evaluation metrics.")
        return

    # Check if sketch triplet exists, try to reload if missing
    sketch_triplet = st.session_state.sketch_triplet
    if sketch_triplet is None and st.session_state.sample_id:
        # Try to reload from saved sketches
        try:
            sample_dir = Path("outputs") / st.session_state.sample_id / "sketches"
            if all([
                (sample_dir / "mobile.png").exists(),
                (sample_dir / "tablet.png").exists(),
                (sample_dir / "desktop.png").exists()
            ]):
                loader = SketchLoader()
                sketch_triplet = loader.load_triplet(
                    sample_dir / "mobile.png",
                    sample_dir / "tablet.png",
                    sample_dir / "desktop.png",
                    sample_id=st.session_state.sample_id
                )
                st.session_state.sketch_triplet = sketch_triplet
                st.success("✅ Loaded wireframe sketches from saved files.")
        except Exception as e:
            st.warning(f"⚠️ Could not reload wireframe sketches: {e}")

    if sketch_triplet is None:
        st.warning(
            "⚠️ **Warning:** Wireframe sketches not available. "
            "Evaluation will run but IoU metrics require ground truth comparison. "
            "LLM judge evaluation may have limited accuracy without original wireframes."
        )

    # Ground Truth Upload Section
    st.subheader("📁 Ground Truth (Optional)")
    st.caption(
        "Upload a single HTML file as ground truth. It will be rendered at three viewport sizes "
        "(mobile 375px, tablet 768px, desktop 1280px) with full height for evaluation comparison. "
        "If not provided, evaluation will still run with LLM judge using wireframes."
    )

    ground_truth_option = st.radio(
        "Ground Truth Input Method",
        ["File Path", "Upload HTML File"],
        horizontal=True,
        help="Choose to provide a file path or upload a single HTML file"
    )

    ground_truth_file = None
    ground_truth_dir = None

    if ground_truth_option == "File Path":
        gt_path = st.text_input(
            "Ground Truth HTML File Path",
            value="",
            placeholder="e.g., data/58.html or outputs/sample_001/ground_truth.html",
            help="Path to a single HTML file (will be rendered at mobile, tablet, and desktop viewports)"
        )
        if gt_path:
            gt_path_obj = Path(gt_path)
            if gt_path_obj.exists():
                if gt_path_obj.is_file() and gt_path_obj.suffix == ".html":
                    ground_truth_file = gt_path_obj
                    st.success(f"✅ Found HTML file: {gt_path}")
                elif gt_path_obj.is_dir():
                    # Check if directory contains mobile.html, tablet.html, desktop.html (backward compatibility)
                    gt_mobile = gt_path_obj / "mobile.html"
                    gt_tablet = gt_path_obj / "tablet.html"
                    gt_desktop = gt_path_obj / "desktop.html"
                    if all([gt_mobile.exists(), gt_tablet.exists(), gt_desktop.exists()]):
                        ground_truth_dir = gt_path_obj
                        st.success(f"✅ Found directory with three HTML files (mobile.html, tablet.html, desktop.html)")
                    else:
                        # Check for single HTML file in directory
                        html_files = list(gt_path_obj.glob("*.html"))
                        if len(html_files) == 1:
                            ground_truth_file = html_files[0]
                            st.success(f"✅ Found HTML file in directory: {html_files[0].name}")
                        else:
                            st.warning(f"⚠️ Directory does not contain expected files. Expected: mobile.html, tablet.html, desktop.html OR a single HTML file")
                else:
                    st.warning(f"⚠️ Path exists but is not an HTML file or directory: {gt_path}")
            else:
                st.warning(f"⚠️ Path not found: {gt_path}")
    else:
        # Upload single HTML file
        gt_html_file = st.file_uploader(
            "Upload Ground Truth HTML File",
            type=["html"],
            key="gt_html_file",
            help="Upload a single HTML file that will be rendered at mobile, tablet, and desktop viewports"
        )

        if gt_html_file:
            # Save to persistent location in outputs directory
            sample_id = st.session_state.sample_id or "ground_truth"
            artifact_manager = ArtifactManager(output_dir="outputs")
            sample_dir = artifact_manager.create_sample_directory(sample_id)
            gt_dir = sample_dir / "ground_truth"
            gt_dir.mkdir(parents=True, exist_ok=True)

            gt_file_path = gt_dir / "ground_truth.html"
            gt_file_path.write_bytes(gt_html_file.getvalue())

            ground_truth_file = gt_file_path
            st.success(f"✅ Ground truth HTML file saved to {gt_file_path}")

    st.divider()

    # Evaluate button
    if st.button("🔍 Run Evaluation", type="primary"):
        if sketch_triplet is None:
            st.error("❌ Cannot run evaluation: Wireframe sketches are required but not available.")
            return

        with st.spinner("🔄 Running comprehensive evaluation (this may take a minute)..."):
            try:
                # Initialize ResponsiveMeter with weights from sidebar
                meter = ResponsiveMeter(
                    w1=w1,
                    w2=w2,
                    w3=w3,
                    w4=w4,
                    judge_provider=provider,
                    judge_model=model_name
                )

                # Perform evaluation
                rendered_output = st.session_state.rendered_output

                # Use evaluate_and_save which handles both with and without ground truth
                sample_id = st.session_state.sample_id or rendered_output.sample_id
                output_path = Path("outputs") / sample_id / "metrics.json"

                # Handle ground truth: single file or directory
                gt_for_eval = None
                if ground_truth_file and ground_truth_file.exists():
                    # Single HTML file - pass as file path
                    gt_for_eval = ground_truth_file
                elif ground_truth_dir and isinstance(ground_truth_dir, Path) and ground_truth_dir.exists():
                    # Directory with mobile.html, tablet.html, desktop.html (backward compatibility)
                    gt_for_eval = ground_truth_dir

                evaluation_result = meter.evaluate_and_save(
                    wireframe_triplet=sketch_triplet,
                    rendered_output=rendered_output,
                    output_path=output_path,
                    ground_truth_dir=gt_for_eval
                )

                # Store result in session state
                st.session_state.evaluation_result = evaluation_result

                st.success("✅ Evaluation complete!")
                if gt_for_eval:
                    if isinstance(gt_for_eval, Path) and gt_for_eval.is_file():
                        st.info(f"📊 Evaluation included ground truth comparison from single HTML file: {gt_for_eval}")
                    else:
                        st.info(f"📊 Evaluation included ground truth comparison from: {gt_for_eval}")
                else:
                    st.info("📊 Evaluation ran without ground truth (IoU/perceptual metrics are placeholders). LLM judge used wireframe comparison.")

            except Exception as e:
                st.error(f"❌ Evaluation error: {str(e)}")
                st.exception(e)

    # Display results
    if st.session_state.evaluation_result:
        result = st.session_state.evaluation_result
        meter_score = result.responsive_meter

        # Composite Score
        st.subheader("🎯 Responsive Meter Score")
        score_col1, score_col2, score_col3, score_col4 = st.columns(4)

        with score_col1:
            st.metric("Composite Score", f"{meter_score.composite_score:.4f}")

        with score_col2:
            st.metric("Avg IoU", f"{meter_score.iou_metrics.average_iou:.4f}")

        with score_col3:
            perceptual_avg = (
                meter_score.perceptual_metrics.clip_similarity_mobile +
                meter_score.perceptual_metrics.clip_similarity_tablet +
                meter_score.perceptual_metrics.clip_similarity_desktop
            ) / 3.0 if (meter_score.perceptual_metrics.clip_similarity_mobile > 0 or
                       meter_score.perceptual_metrics.clip_similarity_tablet > 0 or
                       meter_score.perceptual_metrics.clip_similarity_desktop > 0) else 0.0
            st.metric("Perceptual", f"{perceptual_avg:.4f}")

        with score_col4:
            st.metric("LLM Judge", f"{meter_score.llm_judge_score.overall_score:.4f}")

        st.divider()

        # Detailed metrics in tabs
        metric_tab1, metric_tab2, metric_tab3 = st.tabs(["📊 IoU Metrics", "🎨 Perceptual Metrics", "🤖 LLM Judge"])

        with metric_tab1:
            st.subheader("IoU-Based Layout Similarity")

            iou_col1, iou_col2, iou_col3 = st.columns(3)
            with iou_col1:
                st.metric("📱 Mobile IoU", f"{meter_score.iou_metrics.mobile_iou:.4f}")
            with iou_col2:
                st.metric("💻 Tablet IoU", f"{meter_score.iou_metrics.tablet_iou:.4f}")
            with iou_col3:
                st.metric("🖥️ Desktop IoU", f"{meter_score.iou_metrics.desktop_iou:.4f}")

            if meter_score.iou_metrics.average_iou == 0.0:
                st.info("ℹ️ IoU metrics require ground truth HTML files for comparison. Upload ground truth files above to enable IoU evaluation.")

        with metric_tab2:
            st.subheader("Perceptual Similarity Metrics")

            st.metric("📱 Mobile CLIP", f"{meter_score.perceptual_metrics.clip_similarity_mobile:.4f}")
            st.metric("💻 Tablet CLIP", f"{meter_score.perceptual_metrics.clip_similarity_tablet:.4f}")
            st.metric("🖥️ Desktop CLIP", f"{meter_score.perceptual_metrics.clip_similarity_desktop:.4f}")
            st.metric("Block Similarity", f"{meter_score.perceptual_metrics.block_similarity:.4f}")
            st.metric("Text Region Similarity", f"{meter_score.perceptual_metrics.text_region_similarity:.4f}")
            st.metric("Positional Alignment", f"{meter_score.perceptual_metrics.positional_alignment:.4f}")

            if all([
                meter_score.perceptual_metrics.clip_similarity_mobile == 0.0,
                meter_score.perceptual_metrics.clip_similarity_tablet == 0.0,
                meter_score.perceptual_metrics.clip_similarity_desktop == 0.0
            ]):
                st.info("ℹ️ Perceptual metrics require ground truth for comparison. Upload ground truth files above to enable perceptual evaluation.")

        with metric_tab3:
            st.subheader("LLM Judge Qualitative Assessment")

            judge_col1, judge_col2 = st.columns(2)
            with judge_col1:
                st.metric("Layout Accuracy (Mobile)", f"{meter_score.llm_judge_score.layout_accuracy_mobile:.4f}")
                st.metric("Layout Accuracy (Tablet)", f"{meter_score.llm_judge_score.layout_accuracy_tablet:.4f}")
                st.metric("Layout Accuracy (Desktop)", f"{meter_score.llm_judge_score.layout_accuracy_desktop:.4f}")
            with judge_col2:
                st.metric("Visual Hierarchy", f"{meter_score.llm_judge_score.visual_hierarchy_score:.4f}")
                st.metric("Cross-Device Consistency", f"{meter_score.llm_judge_score.cross_device_consistency:.4f}")
                st.metric("Overall Score", f"{meter_score.llm_judge_score.overall_score:.4f}")

            st.subheader("Feedback")
            st.text_area(
                "LLM Judge Feedback",
                value=meter_score.llm_judge_score.feedback or "No feedback available.",
                height=200,
                disabled=True,
                key="llm_judge_feedback"
            )

        st.divider()

        # Download evaluation results
        if st.session_state.sample_id:
            sample_id = st.session_state.sample_id
            metrics_path = Path("outputs") / sample_id / "metrics.json"
            if metrics_path.exists():
                st.download_button(
                    "⬇️ Download Evaluation Results (JSON)",
                    data=metrics_path.read_text(encoding="utf-8"),
                    file_name=f"{sample_id}_evaluation.json",
                    mime="application/json",
                    key="download_evaluation_results"
                )


def orchestration_tab(provider, model_name):
    """LangGraph orchestration interface with chat."""
    st.header("🤖 Multi-Agent Orchestration")
    st.markdown("""
    **Iterative refinement workflow** with multiple specialized agents:
    - **Orchestrator**: Routes workflow intelligently
    - **Generator**: Creates initial HTML from wireframes
    - **Evaluator**: Runs comprehensive metrics
    - **Reviewer**: Analyzes issues and provides feedback
    - **Editor**: Applies targeted improvements
    """)

    # Initialize app
    if st.session_state.orchestration_app is None:
        with st.spinner("Initializing orchestration system..."):
            st.session_state.orchestration_app = get_responsive_app(
                checkpoint_dir=".checkpoints",
                use_checkpointing=True
            )

    # Thread management
    col1, col2 = st.columns([3, 1])
    with col1:
        thread_id = st.text_input(
            "Thread ID",
            value=st.session_state.orchestration_thread_id or f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            help="Unique identifier for this conversation thread"
        )
        st.session_state.orchestration_thread_id = thread_id
    with col2:
        if st.button("🆕 New Thread"):
            st.session_state.orchestration_thread_id = f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.rerun()

    st.divider()

    # Upload wireframes
    st.subheader("📤 Upload Wireframe Sketches")
    col1, col2, col3 = st.columns(3)

    with col1:
        mobile_file = st.file_uploader("Mobile", type=["png", "jpg", "jpeg"], key="orch_mobile")
    with col2:
        tablet_file = st.file_uploader("Tablet", type=["png", "jpg", "jpeg"], key="orch_tablet")
    with col3:
        desktop_file = st.file_uploader("Desktop", type=["png", "jpg", "jpeg"], key="orch_desktop")

    # Chat interface
    st.divider()
    st.subheader("💬 Chat with Orchestration System")

    # Display chat history
    if "orchestration_messages" not in st.session_state:
        st.session_state.orchestration_messages = []

    # Show chat history
    for msg in st.session_state.orchestration_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "metadata" in msg:
                with st.expander("Details"):
                    st.json(msg["metadata"])

    # Chat input
    user_input = st.chat_input("Enter your message or instruction...")

    if user_input:
        # Add user message
        st.session_state.orchestration_messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.write(user_input)

        # Prepare state
        config = {"configurable": {"thread_id": thread_id}}

        # Initialize or update state
        sample_id = st.session_state.sample_id or f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "sample_id": sample_id,
            "html": None,
            "html_path": None,
            "eval_results": {},
            "responsive_score": None,
            "edit_history": [],
            "active_view": None,
            "focus_selector": None,
            "iteration": 0,
            "next_step": None,
            "screenshots": {},
            "feedback": None,
            "edit_target": None,
        }

        # Save wireframes if uploaded to persistent location
        if all([mobile_file, tablet_file, desktop_file]):
            from responsive_gen.io.sketch_loader import SketchLoader, ArtifactManager

            # Create persistent directory for sketch files
            artifact_manager = ArtifactManager(output_dir="outputs")
            sample_dir = artifact_manager.create_sample_directory(sample_id)
            sketches_dir = sample_dir / "sketches"
            sketches_dir.mkdir(parents=True, exist_ok=True)

            # Copy files to persistent location
            mobile_path = sketches_dir / "mobile.png"
            tablet_path = sketches_dir / "tablet.png"
            desktop_path = sketches_dir / "desktop.png"

            mobile_path.write_bytes(mobile_file.getvalue())
            tablet_path.write_bytes(tablet_file.getvalue())
            desktop_path.write_bytes(desktop_file.getvalue())

            # Load triplet using persistent paths
            loader = SketchLoader()
            triplet = loader.load_triplet(
                mobile_path,
                tablet_path,
                desktop_path,
                sample_id=sample_id
            )
            initial_state["sketch_triplet"] = triplet

        # Run orchestration
        with st.chat_message("assistant"):
            with st.spinner("🤖 Agents are working..."):
                try:
                    # Invoke graph
                    result = st.session_state.orchestration_app.invoke(
                        initial_state,
                        config=config
                    )

                    # Display response
                    response_text = f"""
**Orchestration Complete!**

- **Iteration**: {result.get('iteration', 0)}
- **Responsive Score**: {result.get('responsive_score', 0.0):.4f}
- **Next Step**: {result.get('next_step', 'FINISH')}
- **Active View**: {result.get('active_view', 'N/A')}
"""

                    if result.get('html'):
                        response_text += f"\n✅ HTML generated ({len(result['html'])} characters)"

                    if result.get('eval_results'):
                        response_text += f"\n📊 Evaluation results available"

                    st.write(response_text)

                    # Show HTML preview if available
                    if result.get('html'):
                        with st.expander("📄 View Generated HTML"):
                            st.code(result['html'], language="html")

                        # Download button
                        st.download_button(
                            "⬇️ Download HTML",
                            data=result['html'],
                            file_name=f"{initial_state['sample_id']}.html",
                            mime="text/html",
                            key="download_html_orchestration_tab"
                        )

                    # Show metrics if available
                    if result.get('eval_results'):
                        with st.expander("📊 View Metrics"):
                            st.json(result['eval_results'])

                    # Add assistant response
                    st.session_state.orchestration_messages.append({
                        "role": "assistant",
                        "content": response_text,
                        "metadata": {
                            "iteration": result.get('iteration'),
                            "score": result.get('responsive_score'),
                            "next_step": result.get('next_step')
                        }
                    })

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.orchestration_messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())


def about_tab():
    """About the project."""

    st.header("ℹ️ About This Project")

    st.markdown("""
    ### Responsive Web Generation from Wireframe Sketch Triplets

    This project extends the *Sketch2Code* framework to a new benchmark task:
    **responsive webpage generation from multi-viewport sketches**.

    #### Pipeline Overview

    1. **Input**: Three wireframe sketches (mobile 375px, tablet 768px, desktop 1280px)
    2. **Generation**: Single-turn LLM generation of responsive HTML with inline CSS
    3. **Rendering**: Screenshots captured at all three viewport sizes
    4. **Evaluation**:
       - IoU-based layout similarity per viewport
       - Perceptual similarity metrics (CLIP, block similarity)
       - LLM-as-a-Judge qualitative assessment
       - Composite Responsive Meter score

    #### Wireframe Conventions

    - Boxes with "X" represent images
    - Wavy lines represent text blocks
    - Rectangles represent cards, buttons, or containers
    - No color or detailed text (low-fidelity abstraction)

    #### Evaluation Framework

    The **Responsive Meter** combines multiple evaluation dimensions:

    ```
    ResponsiveMeter = w1·Avg(IoU) + w2·CrossDeviceConsistency
                    + w3·LLMJudge + w4·Perceptual
    ```

    #### Technology Stack

    - **LangChain**: LLM orchestration
    - **OpenAI/Anthropic**: Vision-language models
    - **Playwright**: Headless browser rendering
    - **Streamlit**: Web interface
    - **Python 3.10+**

    #### Research Motivation

    This project creates the first **multi-sketch → responsive-HTML** benchmark
    aligned with Sketch2Code methodology, addressing:

    - Multi-sketch conditioning for responsive design
    - One-shot responsive generation
    - Cross-device evaluation metrics
    - Responsiveness-aware qualitative evaluation

    #### Status

    ✅ Complete: Core pipeline, generation, rendering
    🚧 In Progress: Evaluation metric implementations
    📋 Planned: Dataset creation, benchmark suite

    ---

    For more information, see the README.md file or project documentation.
    """)


if __name__ == "__main__":
    main()

