import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from convolution_patterns.config.config import Config
from convolution_patterns.exception import CustomException
from convolution_patterns.logger_manager import LoggerManager
from convolution_patterns.services.chart_render.chart_render_service import (
    ChartRenderService,
)

logging = LoggerManager.get_logger(__name__)


class RenderImagePipeline:
    """
    Pipeline for rendering chart images from indicator data using tiered right-anchored windows.

    This pipeline:
    1. Loads indicator data from input source
    2. Creates right-anchored windows of specified sizes (one per size)
    3. Renders chart images using the specified backend
    4. Saves images to output directory
    5. Generates a manifest file mapping images to metadata
    """

    def __init__(self):
        """
        Initialize the RenderImagePipeline.
        """
        self.config = Config()

        # Pipeline configuration
        self.input_path = self.config.render_input_path
        self.output_dir = self.config.render_output_dir
        self.manifest_path = self.config.render_manifest_path
        self.window_sizes = self.config.render_window_sizes
        self.backend = self.config.render_backend
        self.image_format = self.config.render_image_format
        self.image_size = self.config.image_size
        self.include_close = self.config.include_close
        self.line_width = self.config.render_line_width

        # Renderer will be set based on backend
        self.renderer = None

    def run(self) -> Dict[str, Any]:
        """
        Execute the complete render-images pipeline.

        Returns:
            Dict containing pipeline results and metadata
        """
        try:
            logging.info("🎨 Starting render-images pipeline...")

            # Validate inputs
            self._validate_inputs()

            # Setup output directory
            self._setup_output_directory()

            # Initialize renderer
            self._initialize_renderer()

            # Load indicator data
            logging.info("📊 Loading indicator data from: %s", self.input_path)
            data = self._load_indicator_data()

            # Convert to dict of arrays for efficient processing
            data_arrays = self._dataframe_to_arrays(data)

            # Generate right-anchored tiered windows and render images
            logging.info("🖼️  Rendering images with backend: %s", self.backend)
            manifest_entries = self._render_tiered_windows(data_arrays)

            # Save manifest
            logging.info("📋 Saving manifest to: %s", self.manifest_path)
            self._save_manifest(manifest_entries)

            results = {
                "output_dir": self.output_dir,
                "manifest_path": self.manifest_path,
                "total_images": len(manifest_entries),
                "window_sizes": self.window_sizes,
                "backend": self.backend,
                "image_format": self.image_format,
            }

            logging.info(
                "✅ Pipeline completed successfully. Generated %d images.",
                len(manifest_entries),
            )
            return results

        except Exception as e:
            logging.error("❌ Pipeline failed: %s", str(e))
            raise CustomException("RenderImagePipeline failed: %s" % str(e)) from e

    def _validate_inputs(self):
        """Validate required inputs and parameters."""
        if not self.input_path:
            raise ValueError("Input path is required")

        if not os.path.exists(self.input_path):
            raise FileNotFoundError("Input file not found: %s" % self.input_path)

    def _setup_output_directory(self):
        """Create output directory if it doesn't exist."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        logging.info("📁 Output directory ready: %s", self.output_dir)

    def _initialize_renderer(self):
        """Initialize the appropriate chart renderer based on backend."""
        self.renderer = ChartRenderService.get_renderer(
            backend=self.backend,
            image_size=self.image_size,
            image_format=self.image_format,
            include_close=self.include_close,
            line_width=self.line_width,
        )
        logging.info(
            "🔧 Initialized %s renderer (include_close=%s)",
            self.backend,
            self.include_close,
        )

    def _load_indicator_data(self) -> pd.DataFrame:
        """
        Load indicator data from input source.

        Returns:
            DataFrame containing indicator data
        """
        try:
            if self.input_path.endswith(".csv"):
                data = pd.read_csv(self.input_path, encoding="utf-8")
            elif self.input_path.endswith(".json"):
                data = pd.read_json(self.input_path, encoding="utf-8")
            else:
                # Try CSV as default
                data = pd.read_csv(self.input_path, encoding="utf-8")

            logging.info("📊 Loaded %d rows of indicator data", len(data))
            return data

        except Exception as e:
            raise CustomException(f"Failed to load indicator data: {e}") from e

    def _dataframe_to_arrays(
        self, df: pd.DataFrame, required_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Convert DataFrame to dict of numpy arrays for efficient processing.

        Args:
            df: Input DataFrame
            required_columns: List of required columns (defaults to standard indicators)

        Returns:
            Dict mapping column names to numpy arrays
        """
        if required_columns is None:
            required_columns = [
                "Close",
                "AHMA",
                "Leavitt_Projection",
                "Leavitt_Convolution",
            ]

        # Validate required columns exist
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError("Missing required columns: %s" % missing_cols)

        return {col: df[col].values for col in required_columns}

    def _render_tiered_windows(
        self, data_arrays: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate right-anchored tiered windows and render images for each window size.

        Args:
            data_arrays: Dict mapping column names to numpy arrays

        Returns:
            List of manifest entries for generated images
        """
        manifest_entries = []
        total_length = len(next(iter(data_arrays.values())))

        for window_size in self.window_sizes:
            logging.info("🔄 Processing window size: %d", window_size)

            # Skip if window size is larger than available data
            if window_size > total_length:
                logging.warning(
                    "⚠️  Skipping window size %d: larger than data length %d",
                    window_size,
                    total_length,
                )
                continue

            try:
                # Generate single right-anchored window (last N bars)
                window_data = {k: v[-window_size:] for k, v in data_arrays.items()}

                # Render the chart image (pass include_close explicitly)
                image = self.renderer.render(
                    window_data,
                    num_days=window_size,
                    include_close=self.include_close,
                    line_width=self.line_width,
                )

                # Generate filename (no index needed since only one window per size)
                filename = "chart_w%d.%s" % (window_size, self.image_format)
                image_path = os.path.join(self.output_dir, filename)

                # Save image
                self._save_image(image, image_path)

                # Create manifest entry
                manifest_entry = {
                    "image_path": image_path,
                    "filename": filename,
                    "window_size": window_size,
                    "window_index": 0,  # Always 0 since only one window per size
                    "data_start_index": total_length - window_size,
                    "data_end_index": total_length - 1,
                    "backend": self.backend,
                    "image_format": self.image_format,
                    "image_size": f"{self.image_size[0]}x{self.image_size[1]}",
                    "include_close": self.include_close,
                    "line_width": self.line_width,
                }

                manifest_entries.append(manifest_entry)
                logging.info("✅ Rendered window size %d", window_size)

            except Exception as e:
                logging.warning(
                    "⚠️  Failed to render window size %d: %s", window_size, str(e)
                )
                continue

        return manifest_entries

    def _save_image(self, image, image_path: str):
        """
        Save rendered image to disk using the renderer's save method.

        Args:
            image: Rendered image object
            image_path: Path where to save the image
        """
        self.renderer.save(image, image_path)

    def _save_manifest(self, manifest_entries: List[Dict[str, Any]]):
        """
        Save manifest file mapping images to metadata.

        Args:
            manifest_entries: List of manifest entries
        """
        try:
            if self.manifest_path.endswith(".json"):
                with open(self.manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest_entries, f, indent=2)
            else:
                # Default to CSV
                if manifest_entries:
                    df = pd.DataFrame(manifest_entries)
                    df.to_csv(self.manifest_path, index=False, encoding="utf-8")
                else:
                    # Create empty CSV with headers
                    with open(
                        self.manifest_path, "w", newline="", encoding="utf-8"
                    ) as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [
                                "image_path",
                                "filename",
                                "window_size",
                                "window_index",
                                "data_start_index",
                                "data_end_index",
                                "backend",
                                "image_format",
                                "image_size",
                                "include_close",  # Add header for include_close
                            ]
                        )

            logging.info("📋 Manifest saved with %d entries", len(manifest_entries))

        except Exception as e:
            raise CustomException("Failed to save manifest: %s" % str(e)) from e
