from typing import Optional, Union

import numpy as np
import pandas as pd

from convolution_patterns.config.config import Config
from convolution_patterns.logger_manager import LoggerManager
from convolution_patterns.services.chart_render_service import ChartRenderService
from convolution_patterns.services.inference_model_service import InferenceModelService
from convolution_patterns.services.pattern_detector_service import (
    PatternDetectorService,
)
from convolution_patterns.services.sliding_inference_controller import (
    SlidingInferenceController,
)
from convolution_patterns.types import InferenceResult

logging = LoggerManager.get_logger(__name__)


class InferencePipeline:
    def __init__(self, model_path: Optional[str] = None):
        self.config = Config()
        self.model_path = model_path or self.config.inference_model_path
        self.print_stats = True

        # Initialize services
        self._initialize_services()

    def _initialize_services(self):
        """Initialize all required services for the inference pipeline."""
        logging.info("[InferencePipeline] Initializing services...")

        # Chart rendering service
        self.chart_render_service = ChartRenderService(
            img_size=self.config.image_size, line_thickness=self.config.line_thickness
        )

        # Model service
        if self.model_path is None:
            raise ValueError("Model path must not be None.")

        self.model_service = InferenceModelService.from_file(self.model_path)
        logging.info(f"[InferencePipeline] Loaded model from: {self.model_path}")

        # Sliding inference controller
        self.sliding_controller = SlidingInferenceController(
            render_service=self.chart_render_service,
            model_service=self.model_service,
            window_sizes=self.config.window_sizes,
            threshold=self.config.confidence_threshold,
        )

        # Pattern detector service (orchestrator)
        self.pattern_detector = PatternDetectorService(
            render_service=self.chart_render_service,
            sliding_controller=self.sliding_controller,
            model_service=self.model_service,
        )

    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate input dataframe has required columns."""
        required_columns = ["LC", "LP", "AHMA"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logging.error(
                f"[InferencePipeline] Missing required columns: {missing_columns}"
            )
            return False

        if len(df) < min(self.config.window_sizes):
            logging.warning(
                f"[InferencePipeline] DataFrame has {len(df)} rows, minimum required: {min(self.config.window_sizes)}"
            )
            return False

        return True

    def _log_inference_stats(self, df: pd.DataFrame, result: Optional[InferenceResult]):
        """Log inference statistics if enabled."""
        if not self.print_stats:
            return

        logging.info(f"[InferencePipeline] Input data shape: {df.shape}")
        logging.info(f"[InferencePipeline] Available bars: {len(df)}")
        logging.info(
            f"[InferencePipeline] Window sizes tested: {self.config.window_sizes}"
        )

        if result:
            logging.info(f"[InferencePipeline] Pattern detected!")
            logging.info(f"  - Label: {result.label}")
            logging.info(f"  - Confidence: {result.confidence:.4f}")
            logging.info(f"  - Window size: {result.window_size}")
            logging.info(
                f"  - Raw probabilities: {[f'{p:.4f}' for p in result.raw_probs]}"
            )
        else:
            logging.info("[InferencePipeline] No pattern detected above threshold")

    def run(self, df: pd.DataFrame) -> Optional[InferenceResult]:
        """
        Run the complete inference pipeline on input dataframe.

        Args:
            df: DataFrame with LC, LP, AHMA columns

        Returns:
            InferenceResult if pattern detected above threshold, None otherwise
        """
        logging.info("[InferencePipeline] Starting inference pipeline...")

        # Validate input
        if not self._validate_dataframe(df):
            raise ValueError("Invalid input dataframe")

        # Run pattern detection
        result = self.pattern_detector.detect(df)

        # Log results
        self._log_inference_stats(df, result)

        logging.info("[InferencePipeline] Inference pipeline completed")
        return result

    def run_batch(
        self, dataframes: list[pd.DataFrame]
    ) -> list[Optional[InferenceResult]]:
        """
        Run inference on multiple dataframes.

        Args:
            dataframes: List of DataFrames to process

        Returns:
            List of InferenceResults (None for no detection)
        """
        logging.info(
            f"[InferencePipeline] Starting batch inference on {len(dataframes)} dataframes..."
        )

        results = []
        for i, df in enumerate(dataframes):
            logging.info(
                f"[InferencePipeline] Processing dataframe {i+1}/{len(dataframes)}"
            )
            try:
                result = self.run(df)
                results.append(result)
            except Exception as e:
                logging.error(
                    f"[InferencePipeline] Error processing dataframe {i+1}: {e}"
                )
                results.append(None)

        logging.info(
            f"[InferencePipeline] Batch inference completed. Detected patterns: {sum(1 for r in results if r is not None)}/{len(results)}"
        )
        return results
