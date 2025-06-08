import tensorflow as tf
from pathlib import Path
from typing import Tuple

from convolution_patterns.config.config import Config
from convolution_patterns.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)

class ImageDatasetService:
    def __init__(self):
        """
        Service to load image datasets for training, validation, or testing.

        Args:
            data_dir (Path): Root directory containing train/val/test splits.
            image_size (Tuple[int, int]): Target size for resizing images.
            batch_size (int): Number of images per batch.
        """
        self.config = Config()
        self.data_dir = Path(self.config.PROCESSED_DATA_DIR)
        self.image_size = self.config.image_size
        self.batch_size = self.config.batch_size

    def _print_dataset_stats(self, dataset, class_names: list[str], name: str):
        lines = [f"\n📊 Dataset: {name}"]
        total_images = 0
        num_classes = None
        for batch_idx, (images, labels) in enumerate(dataset):
            if batch_idx == 0:
                lines.append(f"  ➤ Image shape: {images.shape}, dtype: {images.dtype}")
                lines.append(f"  ➤ Label shape: {labels.shape}, dtype: {labels.dtype}")
                num_classes = labels.shape[-1]
            total_images += images.shape[0]

        lines.append(f"  ➤ Total images: {total_images}")
        lines.append(f"  ➤ Number of batches: {batch_idx + 1}")
        if num_classes is not None:
            lines.append(f"  ➤ Number of classes (inferred): {num_classes}")
            lines.append(f"  ➤ Class Names: {class_names}")

        logging.info("\n" + "\n".join(lines))

    def get_dataset(self, split: str = "train", print_stats: bool = False) -> Tuple["tf.data.Dataset", list[str]]:

        """
        Load a dataset split without applying transforms or shuffling.

        Args:
            split (str): One of 'train', 'val', or 'test'.

        Returns:
            tf.data.Dataset: Batched and prefetched dataset.
        """
        split_path = self.data_dir / split
        if not split_path.exists():
            raise FileNotFoundError(f"Dataset split path not found: {split_path}")

        raw_dataset = tf.keras.utils.image_dataset_from_directory(
            directory=split_path,
            labels="inferred",
            label_mode="categorical",
            image_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=False
        )

        class_names = raw_dataset.class_names
        dataset = raw_dataset.prefetch(tf.data.AUTOTUNE)

        if print_stats:
            self._print_dataset_stats(raw_dataset, class_names,  split)

        return dataset, class_names
