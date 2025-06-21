#!/usr/bin/env python3
"""
AUD_JPY Chart Pattern Data Processor - BARS VERSION
Generates image rendering calls with only essential data: Date, Close, AHMA, LC, LP
Uses BARS (rows) instead of days for window and stride calculations
"""

import json
import logging
import os
from datetime import datetime

import pandas as pd

# Configure logging with lazy evaluation (Rule 1)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ChartPatternDataProcessor:
    def __init__(self, input_file, window_bars=30, stride_bars=5, min_bars=None):
        """
        Initialize the processor

        Args:
            input_file (str): Path to the AUD_JPY processed signals CSV
            window_bars (int): Number of bars (rows) to include in each data window
            stride_bars (int): Number of bars (rows) to step back for each iteration
            min_bars (int): Minimum number of bars required for processing (default: window_bars)
        """
        self.input_file = input_file
        self.window_bars = window_bars
        self.stride_bars = stride_bars
        self.min_bars = min_bars or window_bars
        self.instrument = "AUD_JPY"

        # Load and prepare data
        self.df = self._load_data()

    def _load_data(self):
        """Load and prepare the CSV data"""
        try:
            # Rule 2: UTF-8 encoding requirement
            df = pd.read_csv(self.input_file, encoding="utf-8")
            # Fix timezone issue - ensure all datetimes are tz-naive
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            df = df.sort_values("Date").reset_index(drop=True)
            logging.info("Loaded %d bars from %s", len(df), self.input_file)
            return df
        except Exception as e:
            logging.error("Failed to load data: %s", str(e))
            raise

    def _extract_essential_data(self, window_df):
        """
        Extract only the essential data for rendering: Date, Close, AHMA, LC, LP

        Args:
            window_df: DataFrame slice for the current window

        Returns:
            dict: Essential data for rendering
        """
        essential_data = {
            "Date": window_df["Date"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(),
            "Close": window_df["Close"].tolist(),
            "AHMA": window_df["AHMA"].tolist(),
            "AHMA": window_df["AHMA"].tolist(),
            "Leavitt_Convolution": window_df["Leavitt_Convolution"].tolist(),
            "Leavitt_Projection": window_df["Leavitt_Projection"].tolist(),
        }

        return essential_data

    def _create_output_directory(self, date_str):
        """Create output directory structure"""
        output_dir = os.path.join("artifacts/data_windows/", self.instrument, date_str)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _save_data_window(self, window_data, output_dir, start_date, end_date):
        """
        Save the data window as JSON and create manifest

        Args:
            window_data: Essential data
            output_dir: Output directory path
            start_date: Window start date
            end_date: Window end date
        """
        # Create data file
        data_filename = (
            f"data_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
        )
        data_filepath = os.path.join(output_dir, data_filename)

        # Add metadata
        complete_data = {
            "data": window_data,
            "metadata": {
                "instrument": self.instrument,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "window_bars": self.window_bars,
                "total_records": len(window_data["Date"]),
                "fields": [
                    "Date",
                    "Close",
                    "AHMA",
                    "Leavitt_Convolution",
                    "Leavitt_Projection",
                ],
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

        # Rule 2: UTF-8 encoding requirement
        with open(data_filepath, "w", encoding="utf-8") as f:
            json.dump(complete_data, f, indent=2, default=str)

        # Create manifest file
        manifest_data = {
            "instrument": self.instrument,
            "date_range": {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
            },
            "files": {
                "data_file": data_filename,
                "images": [],  # Will be populated by render process
            },
            "processing_info": {
                "window_bars": self.window_bars,
                "stride_bars": self.stride_bars,
                "total_records": len(window_data["Date"]),
                "data_fields": [
                    "Date",
                    "Close",
                    "AHMA",
                    "Leavitt_Convolution",
                    "Leavitt_Projection",
                ],
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "render_status": "pending",
        }

        manifest_filepath = os.path.join(output_dir, "manifest.json")
        # Rule 2: UTF-8 encoding requirement
        with open(manifest_filepath, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        logging.info("Created data window: %s", output_dir)
        return data_filepath, manifest_filepath

    def process_data_windows(self):
        """
        Main processing function to create data windows and render calls
        Uses BARS (rows) instead of days for calculations
        """
        if len(self.df) == 0:
            logging.error("No data available for processing")
            return []

        total_bars = len(self.df)

        if total_bars < self.min_bars:
            logging.error(
                "Insufficient data: %d bars available, %d required",
                total_bars,
                self.min_bars,
            )
            return []

        render_calls = []
        window_count = 0

        # Start from the end (most recent bars) and work backwards
        start_index = total_bars - self.window_bars

        logging.info(
            "Starting processing from bar index %d (total bars: %d)",
            start_index,
            total_bars,
        )
        logging.info(
            "Window size: %d bars, Stride: %d bars", self.window_bars, self.stride_bars
        )

        while start_index >= 0:
            end_index = start_index + self.window_bars

            # Get the window data
            window_df = self.df.iloc[start_index:end_index].copy()

            if len(window_df) < self.window_bars:
                logging.warning(
                    "Insufficient bars for window starting at index %d, skipping",
                    start_index,
                )
                start_index -= self.stride_bars
                continue

            # Extract essential data only
            window_data = self._extract_essential_data(window_df)

            # Get date range for this window
            start_date = window_df["Date"].iloc[0]
            end_date = window_df["Date"].iloc[-1]
            date_str = end_date.strftime("%Y-%m-%d")

            # Create output directory
            output_dir = self._create_output_directory(date_str)

            # Save data window
            data_file, manifest_file = self._save_data_window(
                window_data, output_dir, start_date, end_date
            )

            # Create render call information
            render_call = {
                "window_id": window_count,
                "instrument": self.instrument,
                "date": date_str,
                "output_directory": output_dir,
                "data_file": data_file,
                "manifest_file": manifest_file,
                "window_start": start_date.strftime("%Y-%m-%d"),
                "window_end": end_date.strftime("%Y-%m-%d"),
                "record_count": len(window_df),
                "start_bar_index": start_index,
                "end_bar_index": end_index - 1,
            }

            render_calls.append(render_call)
            window_count += 1

            # Move back by stride_bars
            start_index -= self.stride_bars

            if window_count % 10 == 0:
                logging.info("Processed %d windows", window_count)

        logging.info("Processing complete. Created %d data windows", len(render_calls))
        return render_calls

    def generate_render_script(self, render_calls):
        """
        Generate a script to call the patterncli render-images process for each data window

        Args:
            render_calls: List of render call information
        """
        script_lines = [
            "#!/bin/bash",
            "# Auto-generated render script for pattern recognition",
            "# Data fields: Date, Close, AHMA, LC (Leavitt_Convolution), LP (Leavitt_Projection)",
            "# Window/Stride calculated in BARS (rows), not days",
            "# Generated at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "",
            "set -e  # Exit on any error",
            "",
            "echo 'Starting batch render process...'",
            "echo 'Total windows to process: " + str(len(render_calls)) + "'",
            "echo 'Data fields: Date, Close, AHMA, LC, LP'",
            "echo 'Window size: " + str(self.window_bars) + " bars'",
            "echo 'Stride: " + str(self.stride_bars) + " bars'",
            "",
        ]

        for i, call in enumerate(render_calls):
            instrument = call["instrument"]  # e.g., "AUD_JPY"
            date = call["date"]  # e.g., "2025-06-10"
            output_dir = f"./data/rendered/{instrument}/{date}/"
            manifest_csv = f"{output_dir}manifest.csv"

            script_lines.extend(
                [
                    f"echo 'Processing window {i+1}/{len(render_calls)}: {date}'",
                    f"# Window: {call['window_start']} to {call['window_end']} ({call['record_count']} bars)",
                    f"# Bar indices: {call['start_bar_index']} to {call['end_bar_index']}",
                    "poetry run patterncli render-images \\",
                    "  --config configs/render_config.yaml \\",
                    f"  --input '{call['data_file']}' \\",
                    f"  --output-dir '{output_dir}' \\",
                    f"  --manifest '{manifest_csv}' \\",
                    "  --backend pil \\",
                    "  --no-include-close \\",
                    "",
                    "if [ $? -eq 0 ]; then",
                    f"    echo 'Successfully processed window {date}'",
                    "else",
                    f"    echo 'Error processing window {date}'",
                    "    exit 1",
                    "fi",
                    "",
                ]
            )

        script_lines.extend(
            [
                "echo 'Batch render process completed successfully!'",
                "echo 'All images and manifests have been generated.'",
            ]
        )

        # Rule 2: UTF-8 encoding requirement
        with open("render_batch.sh", "w", encoding="utf-8") as f:
            f.write("\n".join(script_lines))

        os.chmod("render_batch.sh", 0o755)

        logging.info("Generated render script: render_batch.sh")
        return "render_batch.sh"


def main():
    """Main execution function"""
    # Configuration - NOW IN BARS, NOT DAYS
    INPUT_FILE = "./examples/AUD_JPY_processed_signals.csv"
    WINDOW_BARS = 30  # Number of bars (rows) for each window
    STRIDE_BARS = 5  # Number of bars (rows) to step back for each iteration
    MIN_BARS = 10  # Minimum bars required for processing

    try:
        # Initialize processor
        processor = ChartPatternDataProcessor(
            input_file=INPUT_FILE,
            window_bars=WINDOW_BARS,
            stride_bars=STRIDE_BARS,
            min_bars=MIN_BARS,
        )

        # Process data windows
        render_calls = processor.process_data_windows()

        if render_calls:
            # Generate render script
            script_file = processor.generate_render_script(render_calls)

            # Save render calls summary
            summary_data = {
                "total_windows": len(render_calls),
                "data_fields": [
                    "Date",
                    "Close",
                    "AHMA",
                    "Leavitt_Convolution",
                    "Leavitt_Projection",
                ],
                "configuration": {
                    "window_bars": WINDOW_BARS,
                    "stride_bars": STRIDE_BARS,
                    "min_bars": MIN_BARS,
                    "input_file": INPUT_FILE,
                },
                "render_calls": render_calls,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Rule 2: UTF-8 encoding requirement
            with open("render_calls_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, default=str)

            print(f"\n=== Processing Summary ===")
            print(f"Data fields: Date, Close, AHMA, LC, LP")
            print(f"Total data windows created: {len(render_calls)}")
            print(f"Window size: {WINDOW_BARS} bars (not days)")
            print(f"Stride: {STRIDE_BARS} bars (not days)")
            print(f"Output directories: {processor.instrument}/YYYY-MM-DD/")
            print(f"Render script: {script_file}")
            print(f"Summary file: render_calls_summary.json")
            print(f"\nTo execute rendering:")
            print(f"  chmod +x {script_file}")
            print(f"  ./{script_file}")

        else:
            logging.warning(
                "No render calls generated. Check your data and configuration."
            )

    except Exception as e:
        logging.error("Processing failed: %s", str(e))
        raise


if __name__ == "__main__":
    main()
