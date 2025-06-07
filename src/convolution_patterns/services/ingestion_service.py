import os
import sys
import shutil
import random

import pandas as pd

from convolution_patterns.config.config import Config
from convolution_patterns.exception import CustomException
from convolution_patterns.logger_manager import LoggerManager
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


    def copy_raw_images(self):
        """
        Copies raw images from staging_dir to RAW_DATA_DIR as a snapshot.
        """
        try:
            if not os.path.exists(self.staging_dir):
                raise FileNotFoundError(f"Staging directory not found or not set: {self.config.staging_dir}")

            if os.path.exists(self.raw_data_dir):
                shutil.rmtree(self.raw_data_dir)
            shutil.copytree(self.staging_dir, self.raw_data_dir)
            logging.info(f"Copied raw images to: {self.raw_data_dir}")
        except Exception as e:
            raise CustomException(e, sys) from e

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
                            records.append({
                                "instrument": instrument,
                                "pattern_type": pattern_type,
                                "filename": filename,
                                "source_path": os.path.join(pattern_path, filename),
                            })

            if not records:
                raise ValueError("No image records found for ingestion. Ensure the staging/raw directory is populated.")

            splitter = SplitterService(
                split_ratios=self.config.split_ratios,
                seed=self.config.random_seed
            )
            return splitter.split(records)

        except Exception as e:
            raise CustomException(e, sys) from e

    def write_processed_dataset(self, split_result):
        """
        Copies split images to processed/{train,val,test}/instrument/pattern_type folders.
        """
        try:
            for split_name, df in split_result.items():
                for record in df.to_dict(orient="records"):
                    dst_dir = os.path.join(
                        self.config.PROCESSED_DATA_DIR,
                        split_name,
                        record["instrument"],
                        record["pattern_type"]
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
                # Convert each row to a dict and add 'split' key
                for rec in df.to_dict(orient="records"):
                    rec["split"] = split_name
                    all_records.append(rec)

            df = pd.DataFrame(all_records)
            metadata_path = os.path.join(self.config.BASE_DIR, "data", "metadata", "pattern_metadata.csv")
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            df.to_csv(metadata_path, index=False)
            return metadata_path
        except Exception as e:
            raise CustomException(e, sys) from e

