import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image

from convolution_patterns.config.config import Config
from convolution_patterns.exception import CustomException
from convolution_patterns.logger_manager import LoggerManager
from convolution_patterns.services.augmentation_service import AugmentationService
from convolution_patterns.services.splitter_service import SplitterService

logging = LoggerManager.get_logger(__name__)


class IngestionService:
    """
    Handles image ingestion for chart pattern classification.
    Performs optional raw copy, stratified split, and metadata generation.
    """

    def __init__(self):
        self.config = Config()

        if not self.config.staging_dir:
            raise ValueError("Missing staging_dir in configuration.")

        self.staging_dir = self.config.staging_dir
        self.raw_data_dir = self.config.RAW_DATA_DIR
        self.label_mode = (
            self.config.label_mode
        )  # 'pattern_only' or 'instrument_specific'

    def copy_raw_images(self):
        """
        Copies images from staging to raw, applying label collapsing if enabled.
        """
        try:
            collapse = self.config.collapse_labels
            remap = self.config.label_remap

            staging_root = Path(self.staging_dir)
            raw_root = Path(self.raw_data_dir)

            if not staging_root.exists():
                raise FileNotFoundError(
                    f"Staging directory not found: {self.staging_dir}"
                )

            if raw_root.exists():
                if not self.config.preserve_raw:
                    shutil.rmtree(raw_root)
                else:
                    logging.info("Preserve raw enabled. Skipping copy.")
                    return

            for instrument_dir in staging_root.iterdir():
                if not instrument_dir.is_dir():
                    continue
                for pattern_dir in instrument_dir.iterdir():
                    if not pattern_dir.is_dir():
                        continue

                    original_label = pattern_dir.name
                    collapsed_label = (
                        remap.get(original_label, original_label)
                        if collapse
                        else original_label
                    )
                    dest_dir = raw_root / instrument_dir.name / collapsed_label
                    dest_dir.mkdir(parents=True, exist_ok=True)

                    for img_file in pattern_dir.glob("*.png"):
                        shutil.copy2(img_file, dest_dir / img_file.name)

            logging.info(
                f"✅ Copied raw images to: {self.raw_data_dir} (collapse_labels={collapse})"
            )

        except Exception as e:
            raise CustomException(e, sys) from e

    def augment_training_data(
        self, train_df: pd.DataFrame, target_minimum: int = 50
    ) -> pd.DataFrame:
        """
        Applies vertical flip + random augmentations to training data.
        Returns augmented DataFrame to append to train split.
        """
        class_names = sorted(train_df["label"].unique())
        augmenter = AugmentationService(class_names, image_size=self.config.image_size)

        logging.info("🔄 Applying vertical flips (label-safe).")
        vflip_df = augmenter.generate_augmented_images(
            train_df,
            mode="vflip",
            output_base_dir=self.raw_data_dir,  # Always target raw/
        )

        logging.info("🎨 Applying random augmentations to underrepresented classes.")
        rand_df = augmenter.generate_augmented_images(
            train_df,
            mode="random",
            target_minimum=target_minimum,
            output_base_dir=self.raw_data_dir,
        )

        augmented_df = pd.concat([vflip_df, rand_df], ignore_index=True)
        logging.info(f"✅ Augmentation complete. {len(augmented_df)} new images.")
        return augmented_df

    def split_dataset(self):
        """
        Loads image paths and performs stratified split into train/val/test using SplitterService.

        Returns:
            dict: Keys 'train', 'val', 'test' each mapped to a list of image record dicts.
        """
        try:
            if not os.path.exists(self.raw_data_dir):
                raise FileNotFoundError(
                    f"Raw data directory not found: {self.raw_data_dir}. "
                    "Did you forget to run copy_raw_images()?"
                )

            records = []
            for instrument in os.listdir(self.raw_data_dir):
                instrument_path = os.path.join(self.raw_data_dir, instrument)
                if not os.path.isdir(instrument_path):
                    continue
                for pattern_type in os.listdir(instrument_path):
                    pattern_path = os.path.join(instrument_path, pattern_type)
                    if not os.path.isdir(pattern_path):
                        continue
                    for filename in os.listdir(pattern_path):
                        if filename.endswith(".png"):
                            label = (
                                pattern_type
                                if self.label_mode == "pattern_only"
                                else f"{instrument}__{pattern_type}"
                            )
                            collapsed_label = (
                                self.config.label_remap.get(label, label)
                                if self.config.collapse_labels
                                else label
                            )
                            records.append(
                                {
                                    "instrument": instrument,
                                    "pattern_type": pattern_type,
                                    "original_label": label,  # Save original label for metadata
                                    "label": collapsed_label,  # This is what gets used in folder path
                                    "filename": filename,
                                    "source_path": os.path.join(pattern_path, filename),
                                }
                            )

            if not records:
                raise ValueError(
                    "No image records found for ingestion. Ensure the staging/raw directory is populated."
                )

            splitter = SplitterService(
                split_ratios=self.config.split_ratios, seed=self.config.random_seed
            )

            splits = splitter.split(records)

            # Convert splits to DataFrames for easier augmentation handling
            splits_df = {
                k: pd.DataFrame(v) if not isinstance(v, pd.DataFrame) else v
                for k, v in splits.items()
            }
            return splits_df

            # return splitter.split(records)

        except Exception as e:
            raise CustomException(e, sys) from e

    def write_processed_dataset(self, split_result):
        """
        Copies split images to processed/{train,val,test}/{label}/filename.png
        """
        try:
            for split_name, df in split_result.items():
                for record in df.to_dict(orient="records"):
                    dst_dir = os.path.join(
                        self.config.PROCESSED_DATA_DIR, split_name, record["label"]
                    )
                    os.makedirs(dst_dir, exist_ok=True)
                    dst_path = os.path.join(dst_dir, record["filename"])
                    shutil.copy2(record["source_path"], dst_path)
        except Exception as e:
            raise CustomException(e, sys) from e

    def write_metadata(self, split_result):
        """
        Saves metadata as a CSV file containing image info and split label.
        """
        try:
            all_records = []
            for split_name, df in split_result.items():
                for rec in df.to_dict(orient="records"):
                    rec["split"] = split_name
                    all_records.append(rec)

            df = pd.DataFrame(all_records)
            columns = list(df.columns)
            for col in ["augmentation", "label_changed"]:
                if col not in columns:
                    df[col] = None

            metadata_path = os.path.join(
                self.config.METADATA_DIR, "pattern_metadata.csv"
            )
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            df.to_csv(metadata_path, index=False)
            return metadata_path
        except Exception as e:
            raise CustomException(e, sys) from e
