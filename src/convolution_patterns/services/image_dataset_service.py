from pathlib import Path
from typing import Tuple

import tensorflow as tf

from convolution_patterns.config.config import Config
from convolution_patterns.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)


class ImageDatasetService:
    def __init__(self):
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

    def get_dataset(
        self,
        split: str = "train",
        prefetch: bool = False,
        print_stats: bool = False,
    ) -> Tuple[tf.data.Dataset, list[str]]:
        split_path = self.data_dir / split
        if not split_path.exists():
            raise FileNotFoundError(f"Dataset split path not found: {split_path}")

        # Discover all class folders
        all_class_names = sorted(
            [entry.name for entry in split_path.iterdir() if entry.is_dir()]
        )

        # Start from "all classes"
        final_class_names = all_class_names

        # === Handle INCLUDE ===
        include = self.config.include_classes
        if include:
            # Validate
            invalid = set(include) - set(all_class_names)
            if invalid:
                logging.warning(f"⚠️ Invalid include_classes ignored: {sorted(invalid)}")
            final_class_names = [c for c in include if c in all_class_names]
            if not final_class_names:
                raise ValueError(
                    "No valid classes remain after applying include_classes."
                )

        # === Handle EXCLUDE ===
        exclude = set(self.config.exclude_classes or [])
        invalid_exclude = exclude - set(all_class_names)
        if invalid_exclude:
            logging.warning(
                f"⚠️ Invalid exclude_classes ignored: {sorted(invalid_exclude)}"
            )
        exclude = exclude & set(all_class_names)
        final_class_names = [c for c in final_class_names if c not in exclude]
        if not final_class_names:
            raise ValueError("No valid classes remain after applying exclude_classes.")

        logging.info(f"✅ Using class names: {final_class_names}")

        # Load dataset with the final class_names (this reindexes labels automatically)
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory=split_path,
            labels="inferred",
            label_mode="categorical",
            class_names=final_class_names,
            image_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=False,
        )

        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        if print_stats:
            self._print_dataset_stats(dataset, final_class_names, split)

        return dataset, final_class_names
