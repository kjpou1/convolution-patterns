#!/usr/bin/env python3
"""
Stage Classified Images Script

Processes manifest files from rendered chart data and stages
classified images into organized directories for training/validation.

Usage:
    python scripts/stage_classified_images.py
"""

import csv
import logging
import os
import shutil
from collections import defaultdict

# Configure logging with lazy evaluation
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def stage_classified_images(rendered_data_path, staging_base_path, instrument):
    """
    Stage classified images from rendered data into classification directories.

    Args:
        rendered_data_path (str): Path to rendered data directory
        staging_base_path (str): Base path for staging area
        instrument (str): Instrument name (e.g., 'AUD_CAD')
    """
    instrument_path = os.path.join(rendered_data_path, instrument)
    staging_instrument_path = os.path.join(staging_base_path, instrument)

    blank_labels = {"", "none", "null", "nan", "NaN"}
    blank_type_counts = defaultdict(int)

    summary = defaultdict(int)
    skipped = 0
    total = 0

    # Walk through all date subdirectories
    for date_dir in os.listdir(instrument_path):
        date_path = os.path.join(instrument_path, date_dir)
        if not os.path.isdir(date_path):
            continue

        manifest_path = os.path.join(date_path, "manifest.csv")
        if not os.path.exists(manifest_path):
            logger.warning("Manifest not found: %s", manifest_path)
            continue

        logger.info("Processing manifest: %s", manifest_path)

        try:
            with open(manifest_path, "r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                has_classification = (
                    reader.fieldnames and "classification" in reader.fieldnames
                )
                if not has_classification:
                    logger.warning(
                        "Manifest missing 'classification' column: %s. Treating all as 'nan'. Columns found: %s",
                        manifest_path,
                        reader.fieldnames,
                    )

                for row in reader:
                    total += 1
                    classification = (
                        row["classification"].strip() if has_classification else "nan"
                    )
                    image_path = row["image_path"].strip()
                    filename = row["filename"].strip()

                    # Skip blank classifications
                    label = classification.strip().lower()
                    if not label or label in blank_labels:
                        blank_type_counts[label] += 1
                        skipped += 1
                        logger.debug(
                            "Skipping blank classification (%s) for: %s",
                            label,
                            filename,
                        )
                        continue

                    # Prepare destination directory and filename
                    classification_dir = os.path.join(
                        staging_instrument_path, classification
                    )
                    os.makedirs(classification_dir, exist_ok=True)
                    summary[classification] += 1
                    new_filename = "%s_%03d.png" % (instrument, summary[classification])
                    dest_path = os.path.join(classification_dir, new_filename)

                    # Copy image
                    try:
                        if os.path.exists(image_path):
                            shutil.copy2(image_path, dest_path)
                            logger.info("Copied: %s -> %s", image_path, dest_path)
                        else:
                            skipped += 1
                            logger.warning("Source file not found: %s", image_path)
                    except Exception as e:
                        skipped += 1
                        logger.error("Error copying file %s: %s", image_path, str(e))
        except Exception as e:
            logger.error("Error reading manifest file: %s", str(e))
            continue

    # Print and log summary
    logger.info("--- Staging Summary for %s ---", instrument)
    print("\n--- Staging Summary for %s ---" % instrument)
    print("Total images processed: %d" % total)
    for classification, count in summary.items():
        print("  %s: %d" % (classification, count))
    print("Images skipped: %d" % skipped)
    if skipped:
        print("  Breakdown by blank type:")
        for blank_type, count in blank_type_counts.items():
            print("    %r: %d" % (blank_type, count))


# -*- coding: utf-8 -*-


def main():
    """Main function to stage classified images."""
    rendered_data_path = "artifacts/rendered"
    staging_base_path = "artifacts/staging"
    instrument = "AUD_CAD"  # Change this to process a different instrument

    logger.info("Starting image staging process for instrument: %s", instrument)
    # os.makedirs(staging_base_path, exist_ok=True)
    # stage_classified_images(rendered_data_path, staging_base_path, instrument)
    # logger.info("Image staging process completed")

    # To process all instruments in the future, replace the above with:
    for instrument in os.listdir(rendered_data_path):
        if os.path.isdir(os.path.join(rendered_data_path, instrument)):
            stage_classified_images(rendered_data_path, staging_base_path, instrument)


if __name__ == "__main__":
    main()
