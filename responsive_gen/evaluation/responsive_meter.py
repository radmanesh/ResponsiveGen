"""
Responsive Meter: Composite scoring system for responsive webpage quality.

This module integrates IoU-based layout similarity, perceptual metrics,
and LLM-based qualitative assessment into a unified scoring system.
"""

from pathlib import Path
from typing import Optional

from responsive_gen.evaluation.layout_metrics import MetricsAggregator
from responsive_gen.evaluation.llm_judge import LLMJudge
from responsive_gen.models import (
    EvaluationResult,
    IoUMetrics,
    LLMJudgeScore,
    PerceptualMetrics,
    RenderedOutput,
    ResponsiveMeterScore,
    SketchTriplet,
)


class ResponsiveMeter:
    """
    Composite evaluation system for responsive webpage generation.

    Combines objective metrics (IoU, perceptual similarity) with LLM-based
    qualitative assessment to produce a unified score.
    """

    def __init__(
        self,
        w1: float = 0.35,  # IoU weight
        w2: float = 0.25,  # Cross-device consistency weight
        w3: float = 0.25,  # LLM judge weight
        w4: float = 0.15,  # Perceptual similarity weight
        judge_provider: Optional[str] = None,
        judge_model: Optional[str] = None
    ):
        """
        Initialize Responsive Meter.

        Args:
            w1: Weight for IoU metrics.
            w2: Weight for cross-device consistency.
            w3: Weight for LLM judge score.
            w4: Weight for perceptual similarity.
            judge_provider: LLM provider for judge. If None, reads from EVALUATOR_PROVIDER env var.
            judge_model: Model name for judge. If None, reads from EVALUATOR_MODEL env var.
        """
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4

        # Normalize weights
        total = w1 + w2 + w3 + w4
        self.w1, self.w2, self.w3, self.w4 = w1/total, w2/total, w3/total, w4/total

        self.metrics_aggregator = MetricsAggregator()
        # Pass None defaults to allow LLMJudge to read from env vars
        self.llm_judge = LLMJudge(provider=judge_provider, model_name=judge_model)

    def evaluate(
        self,
        wireframe_triplet: SketchTriplet,
        rendered_output: RenderedOutput,
        ground_truth_dir: Optional[Path] = None
    ) -> ResponsiveMeterScore:
        """
        Perform comprehensive evaluation and compute Responsive Meter score.

        Args:
            wireframe_triplet: Original wireframe sketches.
            rendered_output: Rendered screenshots of generated page.
            ground_truth_dir: Optional directory with ground truth HTML files.
                            Expected structure: ground_truth_dir/mobile.html,
                                              ground_truth_dir/tablet.html,
                                              ground_truth_dir/desktop.html

        Returns:
            ResponsiveMeterScore with all metrics and composite score.
        """
        # Calculate objective metrics (if ground truth available)
        if ground_truth_dir and Path(ground_truth_dir).exists():
            print(f"Computing metrics with ground truth from: {ground_truth_dir}")
            iou_metrics, perceptual_metrics = self.metrics_aggregator.calculate_all_objective_metrics(
                rendered_output,
                Path(ground_truth_dir)
            )
        else:
            print("No ground truth provided or directory not found. Using placeholder metrics.")
            # Placeholder metrics when no ground truth
            iou_metrics = IoUMetrics(
                mobile_iou=0.0,
                tablet_iou=0.0,
                desktop_iou=0.0,
                average_iou=0.0
            )
            perceptual_metrics = PerceptualMetrics()

        # Calculate LLM judge scores
        # Note: LLM judge still requires implementation of image encoding
        print("Computing LLM judge scores (placeholder)...")
        llm_judge_score = self.llm_judge.evaluate_comprehensive(
            wireframe_triplet,
            rendered_output
        )

        # Create ResponsiveMeterScore
        meter_score = ResponsiveMeterScore(
            iou_metrics=iou_metrics,
            perceptual_metrics=perceptual_metrics,
            llm_judge_score=llm_judge_score,
            w1=self.w1,
            w2=self.w2,
            w3=self.w3,
            w4=self.w4
        )

        # Calculate composite score
        meter_score.calculate_composite()

        print(f"Evaluation complete. Composite score: {meter_score.composite_score:.4f}")

        return meter_score

    def evaluate_and_save(
        self,
        wireframe_triplet: SketchTriplet,
        rendered_output: RenderedOutput,
        output_path: Path,
        ground_truth_dir: Optional[Path] = None
    ) -> EvaluationResult:
        """
        Evaluate and save results to disk.

        Args:
            wireframe_triplet: Original wireframe sketches.
            rendered_output: Rendered screenshots.
            output_path: Where to save evaluation results (JSON).
            ground_truth_dir: Optional ground truth directory.

        Returns:
            EvaluationResult object.
        """
        import json
        from datetime import datetime

        # Perform evaluation
        meter_score = self.evaluate(wireframe_triplet, rendered_output, ground_truth_dir)

        # Create evaluation result
        evaluation_result = EvaluationResult(
            sample_id=wireframe_triplet.sample_id,
            responsive_meter=meter_score,
            evaluation_timestamp=datetime.now(),
            evaluation_metadata={
                "weights": {
                    "w1_iou": self.w1,
                    "w2_consistency": self.w2,
                    "w3_llm_judge": self.w3,
                    "w4_perceptual": self.w4
                },
                "has_ground_truth": ground_truth_dir is not None
            }
        )

        # Save to disk
        output_path.parent.mkdir(parents=True, exist_ok=True)

        result_dict = {
            "sample_id": evaluation_result.sample_id,
            "timestamp": evaluation_result.evaluation_timestamp.isoformat(),
            "composite_score": meter_score.composite_score,
            "metrics": {
                "iou": {
                    "mobile": meter_score.iou_metrics.mobile_iou,
                    "tablet": meter_score.iou_metrics.tablet_iou,
                    "desktop": meter_score.iou_metrics.desktop_iou,
                    "average": meter_score.iou_metrics.average_iou
                },
                "perceptual": {
                    "clip_mobile": meter_score.perceptual_metrics.clip_similarity_mobile,
                    "clip_tablet": meter_score.perceptual_metrics.clip_similarity_tablet,
                    "clip_desktop": meter_score.perceptual_metrics.clip_similarity_desktop,
                    "block_similarity": meter_score.perceptual_metrics.block_similarity,
                    "text_region": meter_score.perceptual_metrics.text_region_similarity,
                    "positional": meter_score.perceptual_metrics.positional_alignment
                },
                "llm_judge": {
                    "layout_accuracy_mobile": meter_score.llm_judge_score.layout_accuracy_mobile,
                    "layout_accuracy_tablet": meter_score.llm_judge_score.layout_accuracy_tablet,
                    "layout_accuracy_desktop": meter_score.llm_judge_score.layout_accuracy_desktop,
                    "visual_hierarchy": meter_score.llm_judge_score.visual_hierarchy_score,
                    "cross_device_consistency": meter_score.llm_judge_score.cross_device_consistency,
                    "overall": meter_score.llm_judge_score.overall_score,
                    "feedback": meter_score.llm_judge_score.feedback
                }
            },
            "weights": evaluation_result.evaluation_metadata["weights"],
            "metadata": evaluation_result.evaluation_metadata
        }

        output_path.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")

        return evaluation_result

