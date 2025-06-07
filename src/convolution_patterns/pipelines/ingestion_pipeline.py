import os
import sys

from convolution_patterns.config.config import Config
from convolution_patterns.exception import CustomException
from convolution_patterns.logger_manager import LoggerManager
from convolution_patterns.services.ingestion_service import IngestionService

logging = LoggerManager.get_logger(__name__)


class IngestionPipeline:
    """
    Ingestion pipeline for Convolution CV chart pattern dataset.
    Handles optional raw snapshot, dataset organization, and metadata generation.
    """

    def __init__(self):
        self.config = Config()
        self.ingestion_service = IngestionService()

        logging.info("📦 Initialized IngestionPipeline with config:")
        logging.info(f"    staging_dir: {self.config.staging_dir}")
        logging.info(f"    preserve_raw: {self.config.preserve_raw}")
        logging.info(f"    label_mode: {self.config.label_mode}")
        logging.info(f"    split_ratios: {self.config.split_ratios}")
        logging.info(f"    random_seed: {self.config.random_seed}")

    def run_pipeline(self):
        """
        Executes the full ingestion pipeline:
        1. Optional raw snapshot
        2. Stratified train/val/test split
        3. Processed directory structure generation
        4. Metadata CSV creation
        """
        try:
            logging.info("🚀 Starting ingestion pipeline.")

            # Step 1: Optional raw copy
            if self.config.preserve_raw:
                logging.info("📂 Copying raw images from staging → raw snapshot")
                self.ingestion_service.copy_raw_images()
                logging.info("✅ Raw snapshot complete.")

            # Step 2: Load, parse, and split
            logging.info("🔄 Parsing directory and splitting dataset.")
            split_result = self.ingestion_service.split_dataset()
            logging.info("✅ Dataset split into train/val/test.")

            # Step 3: Copy to processed layout
            logging.info("📁 Writing processed dataset to artifacts/data/processed/")
            self.ingestion_service.write_processed_dataset(split_result)
            logging.info("✅ Processed dataset saved.")

            # Step 4: Save metadata
            metadata_path = self.ingestion_service.write_metadata(split_result)
            logging.info(f"🧾 Metadata CSV saved: {metadata_path}")

            return {
                "train_count": len(split_result["train"]),
                "val_count": len(split_result["val"]),
                "test_count": len(split_result["test"]),
                "metadata_path": metadata_path,
            }

        except Exception as e:
            logging.error(f"❌ Error in ingestion pipeline: {e}")
            raise CustomException(e, sys) from e
