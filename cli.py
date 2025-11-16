#!/usr/bin/env python3
"""
Command-line interface for Responsive Web Generation pipeline.
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from responsive_gen.evaluation.responsive_meter import ResponsiveMeter
from responsive_gen.io.sketch_loader import ArtifactManager, SketchLoader
from responsive_gen.models import DeviceType
from responsive_gen.pipeline.generation import ResponsiveGenerator
from responsive_gen.rendering.html_renderer import HTMLRenderer

# Load environment variables
load_dotenv()


def cmd_generate(args):
    """Generate responsive HTML from sketch triplet."""
    print("🚀 Generating responsive HTML...")

    # Validate input files
    mobile_path = Path(args.mobile)
    tablet_path = Path(args.tablet)
    desktop_path = Path(args.desktop)

    if not mobile_path.exists():
        print(f"❌ Error: Mobile sketch not found: {mobile_path}")
        return 1
    if not tablet_path.exists():
        print(f"❌ Error: Tablet sketch not found: {tablet_path}")
        return 1
    if not desktop_path.exists():
        print(f"❌ Error: Desktop sketch not found: {desktop_path}")
        return 1

    # Load sketch triplet
    sketch_loader = SketchLoader()
    sample_id = args.sample_id or mobile_path.stem

    print(f"📁 Sample ID: {sample_id}")
    print(f"📱 Mobile: {mobile_path}")
    print(f"💻 Tablet: {tablet_path}")
    print(f"🖥️  Desktop: {desktop_path}")

    triplet = sketch_loader.load_triplet(
        mobile_path, tablet_path, desktop_path,
        sample_id=sample_id
    )

    # Validate sketches
    is_valid, error_msg = sketch_loader.validate_sketches(triplet)
    if not is_valid:
        print(f"❌ Sketch validation failed: {error_msg}")
        return 1

    print("✅ Sketches validated")

    # Initialize generator
    generator = ResponsiveGenerator(
        provider=args.provider,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )

    print(f"🤖 Using {args.provider}/{args.model or 'default'}")

    # Generate HTML
    generated, html_path = generator.generate_and_save(
        triplet,
        output_dir=Path(args.output),
        sketch_loader=sketch_loader
    )

    print(f"✅ HTML generated successfully!")
    print(f"📄 HTML: {html_path}")
    print(f"📊 Tokens: {generated.prompt_tokens} prompt, {generated.completion_tokens} completion")

    # Optionally render screenshots
    if not args.no_render:
        print("\n📸 Rendering screenshots...")
        renderer = HTMLRenderer(headless=True)
        artifact_manager = ArtifactManager(args.output)

        rendered = renderer.render_generated_html(
            generated,
            artifact_manager,
            wait_time=args.render_wait
        )

        print(f"✅ Screenshots saved:")
        print(f"   📱 Mobile: {rendered.mobile_screenshot}")
        print(f"   💻 Tablet: {rendered.tablet_screenshot}")
        print(f"   🖥️  Desktop: {rendered.desktop_screenshot}")

    return 0


def cmd_render(args):
    """Render existing HTML at all viewports."""
    print("📸 Rendering HTML...")

    html_path = Path(args.html)
    if not html_path.exists():
        print(f"❌ Error: HTML file not found: {html_path}")
        return 1

    output_dir = Path(args.output)
    sample_id = args.sample_id or html_path.stem

    print(f"📄 HTML: {html_path}")
    print(f"📁 Sample ID: {sample_id}")
    print(f"💾 Output: {output_dir}")

    # Render
    renderer = HTMLRenderer(headless=not args.headed)
    rendered = renderer.render_all_viewports(
        html_path,
        output_dir,
        sample_id,
        wait_time=args.wait
    )

    print(f"✅ Screenshots saved:")
    print(f"   📱 Mobile: {rendered.mobile_screenshot}")
    print(f"   💻 Tablet: {rendered.tablet_screenshot}")
    print(f"   🖥️  Desktop: {rendered.desktop_screenshot}")

    return 0


def cmd_evaluate(args):
    """Evaluate generated HTML against ground truth."""
    print("📊 Evaluating generated webpage...")

    from responsive_gen.evaluation.responsive_meter import ResponsiveMeter
    from responsive_gen.models import RenderedOutput
    from datetime import datetime

    # Validate inputs
    generated_path = Path(args.generated)
    if not generated_path.exists():
        print(f"❌ Error: Generated HTML not found: {generated_path}")
        return 1

    # Extract sample_id from path
    sample_id = generated_path.parent.name
    print(f"📁 Sample ID: {sample_id}")
    print(f"📄 Generated HTML: {generated_path}")

    # Check for screenshots
    screenshots_dir = generated_path.parent / "screenshots"
    rendered_output = None

    if screenshots_dir.exists():
        print(f"📸 Found screenshots: {screenshots_dir}")
        rendered_output = RenderedOutput(
            sample_id=sample_id,
            mobile_screenshot=screenshots_dir / "mobile.png" if (screenshots_dir / "mobile.png").exists() else None,
            tablet_screenshot=screenshots_dir / "tablet.png" if (screenshots_dir / "tablet.png").exists() else None,
            desktop_screenshot=screenshots_dir / "desktop.png" if (screenshots_dir / "desktop.png").exists() else None,
            rendering_timestamp=datetime.now()
        )
    else:
        print("⚠️  No screenshots found. Evaluation may be limited.")
        rendered_output = RenderedOutput(
            sample_id=sample_id,
            rendering_timestamp=datetime.now()
        )

    # Check for ground truth
    ground_truth_dir = None
    if args.ground_truth:
        ground_truth_dir = Path(args.ground_truth)
        if not ground_truth_dir.exists():
            print(f"⚠️  Ground truth directory not found: {ground_truth_dir}")
            ground_truth_dir = None
        else:
            print(f"✅ Ground truth: {ground_truth_dir}")
            # Check for required files
            required_files = ['mobile.html', 'tablet.html', 'desktop.html']
            missing = [f for f in required_files if not (ground_truth_dir / f).exists()]
            if missing:
                print(f"⚠️  Missing ground truth files: {', '.join(missing)}")
    else:
        print("ℹ️  No ground truth provided. Using placeholder metrics.")

    # Load wireframes if available
    wireframes_dir = Path(args.wireframes) if args.wireframes else None
    wireframe_triplet = None

    if wireframes_dir and wireframes_dir.exists():
        from responsive_gen.io.sketch_loader import SketchLoader
        loader = SketchLoader()
        try:
            wireframe_triplet = loader.load_triplet(
                wireframes_dir / "mobile.png",
                wireframes_dir / "tablet.png",
                wireframes_dir / "desktop.png",
                sample_id=sample_id
            )
            print(f"✅ Loaded wireframes from: {wireframes_dir}")
        except Exception as e:
            print(f"⚠️  Could not load wireframes: {e}")

    # If no wireframe triplet, create a minimal one
    if not wireframe_triplet:
        from responsive_gen.models import SketchTriplet, SketchInput, ViewportConfig, DeviceType
        wireframe_triplet = SketchTriplet(
            sample_id=sample_id,
            mobile=SketchInput(device_type=DeviceType.MOBILE, image_path=Path(""), viewport=ViewportConfig.mobile()),
            tablet=SketchInput(device_type=DeviceType.TABLET, image_path=Path(""), viewport=ViewportConfig.tablet()),
            desktop=SketchInput(device_type=DeviceType.DESKTOP, image_path=Path(""), viewport=ViewportConfig.desktop())
        )

    # Run evaluation
    print("\n🔍 Running evaluation...")
    meter = ResponsiveMeter()

    try:
        result = meter.evaluate(
            wireframe_triplet=wireframe_triplet,
            rendered_output=rendered_output,
            ground_truth_dir=ground_truth_dir
        )

        # Save results
        output_path = Path(args.output)
        meter.evaluate_and_save(
            wireframe_triplet=wireframe_triplet,
            rendered_output=rendered_output,
            output_path=output_path,
            ground_truth_dir=ground_truth_dir
        )

        print(f"\n✅ Evaluation complete!")
        print(f"📊 Results saved to: {output_path}")
        print(f"\n📈 Summary:")
        print(f"   Composite Score: {result.composite_score:.4f}")
        print(f"   Average IoU: {result.iou_metrics.average_iou:.4f}")
        print(f"   Mobile IoU: {result.iou_metrics.mobile_iou:.4f}")
        print(f"   Tablet IoU: {result.iou_metrics.tablet_iou:.4f}")
        print(f"   Desktop IoU: {result.iou_metrics.desktop_iou:.4f}")

        return 0

    except Exception as e:
        print(f"\n❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_orchestrate(args):
    """Run multi-agent orchestration workflow."""
    print("🤖 Starting multi-agent orchestration workflow...")

    from responsive_gen.orchestration.graph import get_responsive_app
    from responsive_gen.io.sketch_loader import SketchLoader
    from langchain_core.messages import HumanMessage
    from datetime import datetime

    # Validate inputs
    mobile_path = Path(args.mobile)
    tablet_path = Path(args.tablet)
    desktop_path = Path(args.desktop)

    if not mobile_path.exists():
        print(f"❌ Error: Mobile sketch not found: {mobile_path}")
        return 1
    if not tablet_path.exists():
        print(f"❌ Error: Tablet sketch not found: {tablet_path}")
        return 1
    if not desktop_path.exists():
        print(f"❌ Error: Desktop sketch not found: {desktop_path}")
        return 1

    # Load sketches
    sketch_loader = SketchLoader()
    sample_id = args.sample_id or mobile_path.stem

    print(f"📁 Sample ID: {sample_id}")
    print(f"📱 Mobile: {mobile_path}")
    print(f"💻 Tablet: {tablet_path}")
    print(f"🖥️  Desktop: {desktop_path}")

    triplet = sketch_loader.load_triplet(
        mobile_path, tablet_path, desktop_path,
        sample_id=sample_id
    )

    # Initialize orchestration app
    print(f"\n🔧 Initializing orchestration system...")
    app = get_responsive_app(
        checkpoint_dir=args.checkpoint_dir,
        use_checkpointing=True
    )

    # Prepare initial state
    thread_id = args.thread_id or f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages": [HumanMessage(content=f"Generate responsive HTML from wireframe sketches for sample {sample_id}")],
        "sample_id": sample_id,
        "sketch_triplet": triplet,
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

    print(f"🧵 Thread ID: {thread_id}")
    print(f"🔄 Max iterations: {args.max_iterations}")
    print(f"🎯 Score threshold: {args.score_threshold}")
    print("\n" + "="*60)

    # Run orchestration
    # The graph will handle routing internally until FINISH or max iterations
    try:
        print("🚀 Starting orchestration workflow...\n")

        # Invoke graph - it will run until FINISH or we interrupt
        result = app.invoke(initial_state, config=config)

        # Display final progress
        print("\n" + "="*60)
        print("✅ Orchestration Complete")
        print("="*60)
        print(f"Final Step: {result.get('next_step', 'UNKNOWN')}")
        if result.get('responsive_score') is not None:
            print(f"📈 Responsive Score: {result.get('responsive_score'):.4f}")
        if result.get('iteration'):
            print(f"🔄 Iterations: {result.get('iteration')}")
        if result.get('active_view'):
            print(f"👁️  Active View: {result.get('active_view')}")
        if result.get('focus_selector'):
            print(f"🎯 Focus Selector: {result.get('focus_selector')}")

        # Final results
        if result:
            print("\n" + "="*60)
            print("📊 Final Results")
            print("="*60)
            print(f"✅ Iterations completed: {result.get('iteration', 0)}")
            print(f"📈 Final Score: {result.get('responsive_score', 0.0):.4f}")

            if result.get('html'):
                # Save HTML
                output_dir = Path(args.output) / sample_id
                output_dir.mkdir(parents=True, exist_ok=True)
                html_path = output_dir / "generated.html"
                html_path.write_text(result['html'], encoding='utf-8')
                print(f"📄 HTML saved: {html_path}")

            if result.get('eval_results'):
                # Save metrics
                import json
                metrics_path = output_dir / "metrics.json"
                metrics_path.write_text(
                    json.dumps(result['eval_results'], indent=2),
                    encoding='utf-8'
                )
                print(f"📊 Metrics saved: {metrics_path}")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Orchestration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Responsive Web Generation from Wireframe Sketches",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate responsive HTML from sketches")
    gen_parser.add_argument("--mobile", "-m", required=True, help="Path to mobile wireframe")
    gen_parser.add_argument("--tablet", "-t", required=True, help="Path to tablet wireframe")
    gen_parser.add_argument("--desktop", "-d", required=True, help="Path to desktop wireframe")
    gen_parser.add_argument("--output", "-o", default="outputs", help="Output directory")
    gen_parser.add_argument("--sample-id", "-s", help="Sample identifier (default: from filename)")
    gen_parser.add_argument("--provider", "-p", default="openai", choices=["openai", "anthropic"])
    gen_parser.add_argument("--model", help="Model name (default: provider default)")
    gen_parser.add_argument("--temperature", type=float, default=0.3, help="Generation temperature")
    gen_parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum tokens")
    gen_parser.add_argument("--no-render", action="store_true", help="Skip screenshot rendering")
    gen_parser.add_argument("--render-wait", type=int, default=1000, help="Render wait time (ms)")

    # Render command
    render_parser = subparsers.add_parser("render", help="Render existing HTML")
    render_parser.add_argument("--html", required=True, help="Path to HTML file")
    render_parser.add_argument("--output", "-o", required=True, help="Output directory")
    render_parser.add_argument("--sample-id", "-s", help="Sample identifier")
    render_parser.add_argument("--wait", "-w", type=int, default=1000, help="Wait time (ms)")
    render_parser.add_argument("--headed", action="store_true", help="Run browser in headed mode")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate generated HTML")
    eval_parser.add_argument("--generated", "-g", required=True, help="Path to generated HTML")
    eval_parser.add_argument("--ground-truth", help="Path to ground truth directory")
    eval_parser.add_argument("--wireframes", help="Path to wireframe directory")
    eval_parser.add_argument("--output", "-o", required=True, help="Output path for metrics JSON")

    # Orchestration command
    orch_parser = subparsers.add_parser("orchestrate", help="Run multi-agent orchestration workflow")
    orch_parser.add_argument("--mobile", "-m", required=True, help="Path to mobile wireframe")
    orch_parser.add_argument("--tablet", "-t", required=True, help="Path to tablet wireframe")
    orch_parser.add_argument("--desktop", "-d", required=True, help="Path to desktop wireframe")
    orch_parser.add_argument("--sample-id", "-s", help="Sample identifier")
    orch_parser.add_argument("--output", "-o", default="outputs", help="Output directory")
    orch_parser.add_argument("--thread-id", help="Thread ID for state persistence")
    orch_parser.add_argument("--max-iterations", type=int, default=5, help="Maximum iterations")
    orch_parser.add_argument("--score-threshold", type=float, default=0.7, help="Score threshold to stop")
    orch_parser.add_argument("--checkpoint-dir", default=".checkpoints", help="Checkpoint directory")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "generate":
            return cmd_generate(args)
        elif args.command == "render":
            return cmd_render(args)
        elif args.command == "evaluate":
            return cmd_evaluate(args)
        elif args.command == "orchestrate":
            return cmd_orchestrate(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

