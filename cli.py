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

    print("⚠️  Note: Full evaluation requires implemented metrics.")
    print("    Current implementation uses placeholder values.\n")

    # This is a stub - actual implementation requires:
    # 1. Loading wireframe triplet
    # 2. Loading rendered output
    # 3. Loading ground truth (if available)
    # 4. Running evaluation pipeline

    print("🚧 Evaluation pipeline scaffolded but not fully implemented.")
    print("   See responsive_gen/evaluation/ for metric implementations.")

    return 0


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
    render_parser.add_argument("--html", "-h", required=True, help="Path to HTML file")
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

