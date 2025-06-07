import os
import sys

from convolution_patterns.config.config import Config
from convolution_patterns.exception import CustomException
from convolution_patterns.logger_manager import LoggerManager
from convolution_patterns.services.ingestion_service import IngestionService

from src.convolution_patterns.utils.path_utils import is_raw_snapshot_empty


logging = LoggerManager.get_logger(__name__)


class IngestionPipeline:
    """
    Ingestion pipeline for Convolution CV chart pattern dataset.
    Handles optional raw snapshot, dataset organization, and metadata generation.
    """

    def __init__(self):
        self.config = Config()
        self.raw_data_dir = self.config.RAW_DATA_DIR
        self.ingestion_service = IngestionService()

        logging.info("📦 Initialized IngestionPipeline with config:")
        logging.info(f"    staging_dir: {self.config.staging_dir}")
        logging.info(f"    preserve_raw: {self.config.preserve_raw}")
        logging.info(f"    label_mode: {self.config.label_mode}")
        logging.info(f"    split_ratios: {self.config.split_ratios}")
        logging.info(f"    random_seed: {self.config.random_seed}")

    def run_pipeline(self):
        try:
            logging.info("🚀 Starting ingestion pipeline.")

            
            raw_empty = is_raw_snapshot_empty(self.raw_data_dir)

            if not self.config.preserve_raw or raw_empty:
                if self.config.preserve_raw and raw_empty:
                    logging.warning("⚠️ Raw snapshot exists but is empty. Recopying from staging...")

                logging.info("📂 Copying raw images from staging → raw snapshot")
                self.ingestion_service.copy_raw_images()
                logging.info("✅ Raw snapshot complete.")
            else:
                logging.info("🛑 Skipping raw copy. Using existing raw snapshot.")

            logging.info("🔄 Parsing directory and splitting dataset.")
            split_result = self.ingestion_service.split_dataset()
            logging.info("✅ Dataset split into train/val/test.")

            logging.info("📁 Writing processed dataset to artifacts/data/processed/")
            self.ingestion_service.write_processed_dataset(split_result)
            logging.info("✅ Processed dataset saved.")

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
