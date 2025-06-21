#!/usr/bin/env python3
"""
Enhanced Chart Pattern Data Processor - BARS VERSION
Generates image rendering calls with only essential data: Date, Close, AHMA, LC, LP
Uses BARS (rows) instead of days for window and stride calculations
Now supports command line arguments for flexible execution
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Configure logging with lazy evaluation (Rule 1)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ChartPatternDataProcessor:

    def __init__(
        self,
        input_file,
        window_bars=30,
        stride_bars=5,
        min_bars=None,
        output_base_dir="artifacts/data_windows",
        instrument=None,
        start_date=None,
        end_date=None,
    ):
        """
        Initialize the processor

        Args:
            input_file (str): Path to the processed signals CSV
            window_bars (int): Number of bars (rows) to include in each data window
            stride_bars (int): Number of bars (rows) to step back for each iteration
            min_bars (int): Minimum number of bars required for processing (default: window_bars)
            output_base_dir (str): Base directory for output files
            instrument (str): Instrument name (auto-detected from filename if not provided)
        """
        self.input_file = input_file
        self.window_bars = window_bars
        self.stride_bars = stride_bars
        self.min_bars = min_bars or window_bars
        self.output_base_dir = output_base_dir

        self.start_date = start_date
        self.end_date = end_date

        # Auto-detect instrument from filename if not provided
        if instrument:
            self.instrument = instrument
        else:
            filename = Path(input_file).stem
            # Extract instrument from filename (e.g., "AUD_JPY_processed_signals" -> "AUD_JPY")
            parts = filename.split("_")
            if len(parts) >= 2 and parts[1] in [
                "JPY",
                "USD",
                "EUR",
                "GBP",
                "CHF",
                "CAD",
                "AUD",
                "NZD",
                "XAU",
                "BTC",
            ]:
                self.instrument = f"{parts[0]}_{parts[1]}"
            else:
                self.instrument = parts[0] if parts else "UNKNOWN"

        # Load and prepare data
        self.df = self._load_data()

    def _load_data(self):
        """Load and prepare the CSV data"""
        try:
            df = pd.read_csv(self.input_file, encoding="utf-8")
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            df = df.sort_values("Date").reset_index(drop=True)
            # Filter by start and end date if provided
            if hasattr(self, "start_date") and self.start_date:
                df = df[df["Date"] >= pd.to_datetime(self.start_date)]
            if hasattr(self, "end_date") and self.end_date:
                df = df[df["Date"] <= pd.to_datetime(self.end_date)]
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
            "Leavitt_Convolution": window_df["Leavitt_Convolution"].tolist(),
            "Leavitt_Projection": window_df["Leavitt_Projection"].tolist(),
        }

        return essential_data

    def _create_output_directory(self, date_str):
        """Create output directory structure"""
        output_dir = os.path.join(self.output_base_dir, self.instrument, date_str)
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

    def generate_render_script(self, render_calls, script_name="render_batch.sh"):
        """
        Generate a script to call the patterncli render-images process for each data window

        Args:
            render_calls: List of render call information
            script_name: Name of the output script file
        """
        script_lines = [
            "#!/bin/bash",
            "# Auto-generated render script for pattern recognition",
            "# Data fields: Date, Close, AHMA, LC (Leavitt_Convolution), LP (Leavitt_Projection)",
            "# Window/Stride calculated in BARS (rows), not days",
            "# Generated at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"# Instrument: {self.instrument}",
            f"# Input file: {self.input_file}",
            "",
            "set -e  # Exit on any error",
            "",
            "echo 'Starting batch render process...'",
            "echo 'Total windows to process: " + str(len(render_calls)) + "'",
            "echo 'Data fields: Date, Close, AHMA, LC, LP'",
            "echo 'Window size: " + str(self.window_bars) + " bars'",
            "echo 'Stride: " + str(self.stride_bars) + " bars'",
            f"echo 'Instrument: {self.instrument}'",
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
        with open(script_name, "w", encoding="utf-8") as f:
            f.write("\n".join(script_lines))

        os.chmod(script_name, 0o755)

        logging.info("Generated render script: %s", script_name)
        return script_name


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Chart Pattern Data Processor - Enhanced with command line support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with single file
  %(prog)s examples/AUD_JPY_processed_signals.csv

  # Custom window and stride settings
  %(prog)s examples/EUR_USD_processed_signals.csv --window-bars 50 --stride-bars 10

  # Process multiple files with custom output directory
  %(prog)s examples/*.csv --output-dir ./custom_output --instrument-override FOREX

  # Batch processing with custom script name
  %(prog)s examples/GBP_USD_processed_signals.csv --script-name gbp_usd_render.sh

  # Verbose logging
  %(prog)s examples/AUD_JPY_processed_signals.csv --verbose
        """,
    )

    parser.add_argument(
        "input_files",
        nargs="+",
        help="Input CSV file(s) containing processed signals data",
    )

    parser.add_argument(
        "--window-bars",
        type=int,
        default=30,
        help="Number of bars (rows) to include in each data window (default: 30)",
    )

    parser.add_argument(
        "--stride-bars",
        type=int,
        default=5,
        help="Number of bars (rows) to step back for each iteration (default: 5)",
    )

    parser.add_argument(
        "--min-bars",
        type=int,
        help="Minimum number of bars required for processing (default: same as window-bars)",
    )

    parser.add_argument(
        "--output-dir",
        default="artifacts/data_windows",
        help="Base directory for output files (default: artifacts/data_windows)",
    )

    parser.add_argument(
        "--instrument-override",
        help="Override instrument name (auto-detected from filename if not provided)",
    )

    parser.add_argument(
        "--script-name",
        default="render_batch.sh",
        help="Name of the generated render script (default: render_batch.sh)",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        help="Only process data from and including this date (format: YYYY-MM-DD or YYYYMMDD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="Only process data up to and including this date (format: YYYY-MM-DD or YYYYMMDD)",
    )

    parser.add_argument(
        "--summary-file",
        default="render_calls_summary.json",
        help="Name of the summary JSON file (default: render_calls_summary.json)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing",
    )

    return parser.parse_args()


def process_single_file(input_file, args):
    """Process a single input file"""
    try:
        logging.info("Processing file: %s", input_file)

        if args.dry_run:
            # Expanded list of codes for forex, metals, and crypto
            CURRENCY_CODES = [
                "JPY",
                "USD",
                "EUR",
                "GBP",
                "CHF",
                "CAD",
                "AUD",
                "NZD",
                "XAU",
                "XAG",
                "BTC",
                "ETH",
                "LTC",
                "USDT",
                "USDC",
            ]
            # Determine instrument (override or auto-detect from filename)
            if args.instrument_override:
                instrument = args.instrument_override
            else:
                filename = Path(input_file).stem
                parts = filename.split("_")
                if len(parts) >= 2 and parts[1] in CURRENCY_CODES:
                    instrument = f"{parts[0]}_{parts[1]}"
                else:
                    instrument = parts[0] if parts else "UNKNOWN"

            # Build date filter info for dry run
            date_filter_info = ""
            if hasattr(args, "start_date") and args.start_date:
                date_filter_info += f", start_date={args.start_date}"
            if hasattr(args, "end_date") and args.end_date:
                date_filter_info += f", end_date={args.end_date}"

            logging.info(
                "DRY RUN: Would process %s (instrument: %s) with window_bars=%d, stride_bars=%d%s",
                input_file,
                instrument,
                args.window_bars,
                args.stride_bars,
                date_filter_info,
            )
            return []

        # Initialize processor
        processor = ChartPatternDataProcessor(
            input_file=input_file,
            window_bars=args.window_bars,
            stride_bars=args.stride_bars,
            min_bars=args.min_bars,
            output_base_dir=args.output_dir,
            instrument=args.instrument_override,
            start_date=args.start_date,
            end_date=args.end_date,
        )

        # Process data windows
        render_calls = processor.process_data_windows()

        if render_calls:
            logging.info(
                "Successfully processed %s: %d windows created",
                input_file,
                len(render_calls),
            )
        else:
            logging.warning("No render calls generated for %s", input_file)

        return render_calls, processor

    except Exception as e:
        logging.error("Failed to process %s: %s", input_file, str(e))
        return [], None


def main():
    """Main execution function"""
    args = parse_arguments()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Verbose logging enabled")

    # Validate input files
    valid_files = []
    for input_file in args.input_files:
        if os.path.exists(input_file):
            valid_files.append(input_file)
        else:
            logging.error("Input file not found: %s", input_file)

    if not valid_files:
        logging.error("No valid input files found")
        sys.exit(1)

    logging.info("Processing %d file(s)", len(valid_files))

    all_render_calls = []
    processed_files = []

    # Process each file
    for input_file in valid_files:
        result = process_single_file(input_file, args)
        if len(result) == 2:
            render_calls, processor = result
            if render_calls:
                all_render_calls.extend(render_calls)
                processed_files.append(
                    {
                        "file": input_file,
                        "instrument": processor.instrument,
                        "windows": len(render_calls),
                    }
                )

    if args.dry_run:
        logging.info("DRY RUN completed. %d files would be processed", len(valid_files))
        return

    if not all_render_calls:
        logging.warning("No render calls generated from any input files")
        return

    # Generate combined render script
    if len(valid_files) == 1:
        # Single file - use the processor from that file
        _, processor = process_single_file(valid_files[0], args)
        script_file = processor.generate_render_script(
            all_render_calls, args.script_name
        )
    else:
        # Multiple files - create a generic script
        script_lines = [
            "#!/bin/bash",
            "# Auto-generated render script for multiple instruments",
            "# Generated at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f"# Processed files: {len(processed_files)}",
            "",
            "set -e  # Exit on any error",
            "",
            "echo 'Starting batch render process for multiple instruments...'",
            f"echo 'Total windows to process: {len(all_render_calls)}'",
            f"echo 'Files processed: {len(processed_files)}'",
            "",
        ]

        for i, call in enumerate(all_render_calls):
            instrument = call["instrument"]
            date = call["date"]
            output_dir = f"./data/rendered/{instrument}/{date}/"
            manifest_csv = f"{output_dir}manifest.csv"

            script_lines.extend(
                [
                    f"echo 'Processing window {i+1}/{len(all_render_calls)}: {instrument} - {date}'",
                    "poetry run patterncli render-images \\",
                    "  --config configs/render_config.yaml \\",
                    f"  --input '{call['data_file']}' \\",
                    f"  --output-dir '{output_dir}' \\",
                    f"  --manifest '{manifest_csv}' \\",
                    "  --backend pil \\",
                    "  --no-include-close \\",
                    "",
                    "if [ $? -eq 0 ]; then",
                    f"    echo 'Successfully processed {instrument} - {date}'",
                    "else",
                    f"    echo 'Error processing {instrument} - {date}'",
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
        with open(args.script_name, "w", encoding="utf-8") as f:
            f.write("\n".join(script_lines))

        os.chmod(args.script_name, 0o755)
        script_file = args.script_name

    # Save comprehensive summary
    summary_data = {
        "processing_summary": {
            "total_files_processed": len(processed_files),
            "total_windows_created": len(all_render_calls),
            "processed_files": processed_files,
        },
        "configuration": {
            "window_bars": args.window_bars,
            "stride_bars": args.stride_bars,
            "min_bars": args.min_bars,
            "output_dir": args.output_dir,
        },
        "data_fields": [
            "Date",
            "Close",
            "AHMA",
            "Leavitt_Convolution",
            "Leavitt_Projection",
        ],
        "render_calls": all_render_calls,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Rule 2: UTF-8 encoding requirement
    with open(args.summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, default=str)

    # Print summary
    print(f"\n=== Processing Summary ===")
    print(f"Files processed: {len(processed_files)}")
    for pf in processed_files:
        print(f"  - {pf['file']} ({pf['instrument']}): {pf['windows']} windows")
    print(f"Total data windows created: {len(all_render_calls)}")
    print(f"Window size: {args.window_bars} bars")
    print(f"Stride: {args.stride_bars} bars")
    print(f"Output directory: {args.output_dir}")
    print(f"Render script: {script_file}")
    print(f"Summary file: {args.summary_file}")
    print(f"\nTo execute rendering:")
    print(f"  chmod +x {script_file}")
    print(f"  ./{script_file}")


if __name__ == "__main__":
    main()
