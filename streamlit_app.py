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
from responsive_gen.models import DeviceType, ViewportConfig
from responsive_gen.pipeline.generation import ResponsiveGenerator
from responsive_gen.rendering.html_renderer import HTMLRenderer

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
if "evaluation_result" not in st.session_state:
    st.session_state.evaluation_result = None


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
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Generate", "🖥️ Results", "📊 Evaluation", "ℹ️ About"])

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
            st.image(mobile_file, caption="Mobile Wireframe", use_container_width=True)

    with col2:
        st.markdown('<div class="device-label">💻 Tablet (768px)</div>', unsafe_allow_html=True)
        tablet_file = st.file_uploader(
            "Upload tablet wireframe",
            type=["png", "jpg", "jpeg"],
            key="tablet_upload",
            label_visibility="collapsed"
        )
        if tablet_file:
            st.image(tablet_file, caption="Tablet Wireframe", use_container_width=True)

    with col3:
        st.markdown('<div class="device-label">🖥️ Desktop (1280px)</div>', unsafe_allow_html=True)
        desktop_file = st.file_uploader(
            "Upload desktop wireframe",
            type=["png", "jpg", "jpeg"],
            key="desktop_upload",
            label_visibility="collapsed"
        )
        if desktop_file:
            st.image(desktop_file, caption="Desktop Wireframe", use_container_width=True)

    st.divider()

    # Sample ID input
    sample_id = st.text_input(
        "Sample ID (optional)",
        value="",
        placeholder=f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Identifier for this generation run"
    )

    # Generate button
    if st.button("🚀 Generate Responsive HTML", type="primary", use_container_width=True):
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
                            use_container_width=True
                        )

                with preview_col2:
                    if st.session_state.rendered_output.tablet_screenshot:
                        st.image(
                            str(st.session_state.rendered_output.tablet_screenshot),
                            caption="Tablet",
                            use_container_width=True
                        )

                with preview_col3:
                    if st.session_state.rendered_output.desktop_screenshot:
                        st.image(
                            str(st.session_state.rendered_output.desktop_screenshot),
                            caption="Desktop",
                            use_container_width=True
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
                st.image(str(rendered.mobile_screenshot), use_container_width=True)

        with screenshot_col2:
            st.markdown("**💻 Tablet (768px)**")
            if rendered.tablet_screenshot:
                st.image(str(rendered.tablet_screenshot), use_container_width=True)

        with screenshot_col3:
            st.markdown("**🖥️ Desktop (1280px)**")
            if rendered.desktop_screenshot:
                st.image(str(rendered.desktop_screenshot), use_container_width=True)

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
        mime="text/html"
    )


def evaluation_tab(w1, w2, w3, w4, provider, model_name):
    """Display evaluation metrics."""

    st.header("📊 Evaluation Metrics")

    if not st.session_state.generated_html or not st.session_state.rendered_output:
        st.info("👆 Generate HTML first to see evaluation metrics.")
        return

    st.warning(
        "⚠️ **Note:** Full evaluation metrics require implementation of component detection, "
        "IoU calculation, and LLM judge. Current metrics are placeholders."
    )

    # Evaluate button
    if st.button("🔍 Run Evaluation", type="primary"):
        with st.spinner("Evaluating..."):
            try:
                # Note: This requires the wireframe triplet to be stored or re-uploaded
                st.info("📝 Evaluation requires original wireframe sketches and optional ground truth.")
                st.info("🚧 Full evaluation pipeline is scaffolded but not yet implemented.")

                # Placeholder metrics display
                st.session_state.evaluation_result = {
                    "composite_score": 0.0,
                    "iou": {"mobile": 0.0, "tablet": 0.0, "desktop": 0.0, "average": 0.0},
                    "perceptual": {"clip_average": 0.0},
                    "llm_judge": {"overall": 0.0, "feedback": "Evaluation not yet implemented."}
                }

            except Exception as e:
                st.error(f"❌ Evaluation error: {str(e)}")

    if st.session_state.evaluation_result:
        result = st.session_state.evaluation_result

        # Composite Score
        st.subheader("🎯 Responsive Meter Score")
        score_col1, score_col2, score_col3, score_col4 = st.columns(4)

        with score_col1:
            st.metric("Composite Score", f"{result['composite_score']:.2f}")

        with score_col2:
            st.metric("Avg IoU", f"{result['iou']['average']:.2f}")

        with score_col3:
            st.metric("Perceptual", f"{result['perceptual']['clip_average']:.2f}")

        with score_col4:
            st.metric("LLM Judge", f"{result['llm_judge']['overall']:.2f}")

        st.divider()

        # Detailed metrics
        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.subheader("IoU Metrics")
            st.metric("📱 Mobile IoU", f"{result['iou']['mobile']:.2f}")
            st.metric("💻 Tablet IoU", f"{result['iou']['tablet']:.2f}")
            st.metric("🖥️ Desktop IoU", f"{result['iou']['desktop']:.2f}")

        with detail_col2:
            st.subheader("LLM Judge Feedback")
            st.text_area(
                "Feedback",
                value=result['llm_judge']['feedback'],
                height=150,
                disabled=True
            )


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

