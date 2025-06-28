import os
import random
from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
import tensorflow as tf
from tensorflow import keras

from convolution_patterns.config.config import Config
from convolution_patterns.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)


class BackgroundMode(Enum):
    WHITE = "white"
    BLACK = "black"
    TOP_LEFT = "top_left"
    MOST_COMMON = "most_common"


class AugmentationService:

    vertical_flip_label_map = {
        "CT_Uptrend": "CT_Downtrend",
        "CT_Downtrend": "CT_Uptrend",
        "PB_Uptrend": "PB_Downtrend",
        "PB_Downtrend": "PB_Uptrend",
        "Uptrend_Convergence": "Downtrend_Convergence",
        "Downtrend_Convergence": "Uptrend_Convergence",
        "Uptrend_No_Convergence": "Downtrend_No_Convergence",
        "Downtrend_No_Convergence": "Uptrend_No_Convergence",
        "Trend_Change_Bull": "Trend_Change_Bear",
        "Trend_Change_Bear": "Trend_Change_Bull",
        # "No_Pattern": "No_Pattern",  # no change
    }

    def __init__(
        self, class_names: List[str], image_size: tuple[int, int] = (224, 224)
    ):
        self.config = Config()
        self.label_mode = (
            self.config.label_mode
        )  # 'pattern_only' or 'instrument_specific'

        self.class_names = class_names
        self.num_classes = len(class_names)
        self.image_size = image_size
        self.name_to_index = {name: idx for idx, name in enumerate(class_names)}
        self.index_to_name = {idx: name for idx, name in enumerate(class_names)}

    # === Augmentations ===

    def vertical_flip(self, image: tf.Tensor, label: str) -> tuple[tf.Tensor, str]:
        new_label = self.vertical_flip_label_map.get(label, label)
        image = tf.image.flip_up_down(image)
        return image, new_label

    def add_gaussian_noise(self, image: tf.Tensor) -> tf.Tensor:
        noise = tf.random.normal(
            shape=tf.shape(image), mean=0.0, stddev=0.02, dtype=tf.float32
        )
        noisy = tf.convert_to_tensor(image + noise)
        return tf.clip_by_value(noisy, 0.0, 1.0)

    def brightness_contrast(self, image: tf.Tensor) -> tf.Tensor:
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        return tf.clip_by_value(image, 0.0, 1.0)

    def random_zoom(
        self, image: tf.Tensor, background_mode: BackgroundMode = BackgroundMode.WHITE
    ) -> tf.Tensor:
        """
        Random zoom augmentation with configurable background fill.

        Args:
            image (tf.Tensor): Input image.
            background_mode (BackgroundMode): Enum controlling fill color.

        Returns:
            tf.Tensor: Augmented image.
        """
        zoom = tf.random.uniform([], 0.95, 1.05)
        new_size = tf.cast(tf.cast(self.image_size, tf.float32) * zoom, tf.int32)

        # Resize image
        image_resized = tf.image.resize(image, new_size)

        # Determine padding amounts
        pad_height = self.image_size[0] - new_size[0]
        pad_width = self.image_size[1] - new_size[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # If zoom >1.0 (cropping), center crop to target size
        if pad_height < 0 or pad_width < 0:
            return tf.image.resize_with_crop_or_pad(
                image_resized, self.image_size[0], self.image_size[1]
            )

        # Convert pad values to Python ints
        pad_top = int(pad_top.numpy())
        pad_bottom = int(pad_bottom.numpy())
        pad_left = int(pad_left.numpy())
        pad_right = int(pad_right.numpy())

        # Select background color
        if background_mode == BackgroundMode.WHITE:
            bg_color = tf.constant([1.0, 1.0, 1.0], dtype=tf.float32)
        elif background_mode == BackgroundMode.BLACK:
            bg_color = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        elif background_mode == BackgroundMode.TOP_LEFT:
            bg_color = image[0, 0, :]
        elif background_mode == BackgroundMode.MOST_COMMON:
            # Convert to uint8 for stable counting
            img_uint8 = tf.cast(image * 255, tf.uint8).numpy()
            h, w, c = img_uint8.shape
            flat = img_uint8.reshape(-1, 3)
            import numpy as np

            colors, counts = np.unique(flat, axis=0, return_counts=True)
            dominant_color = colors[np.argmax(counts)]
            bg_color = tf.cast(dominant_color, tf.float32) / 255.0
        else:
            raise ValueError(f"Invalid background_mode: {background_mode}")

        # Pad with zeros (we'll add bg_color after)
        paddings = tf.constant([[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])

        image_padded = tf.pad(
            image_resized, paddings, mode="CONSTANT", constant_values=0.0
        )

        # Add background color
        image_padded = image_padded + tf.reshape(bg_color, [1, 1, 3])

        return image_padded

    def vertical_shift(self, image: tf.Tensor, allow_zero: bool = False) -> tf.Tensor:
        """
        Vertically shifts the image up or down with optional allowance for zero shift.

        Args:
            image (tf.Tensor): Input image tensor.
            allow_zero (bool): If True, allows zero shift (no change).
                            If False, ensures shift is always non-zero.

        Returns:
            tf.Tensor: Vertically shifted image.
        """
        height = tf.shape(image)[0]
        width = tf.shape(image)[1]
        shift_pixels = tf.cast(0.05 * tf.cast(height, tf.float32), tf.int32)
        shift_pixels = tf.maximum(1, shift_pixels)

        if allow_zero:
            shift = tf.random.uniform(
                [], -shift_pixels, shift_pixels + 1, dtype=tf.int32
            )
        else:
            possible_shifts = tf.concat(
                [tf.range(-shift_pixels, 0), tf.range(1, shift_pixels + 1)], axis=0
            )
            shift = tf.random.shuffle(possible_shifts)[0]

        # Pad the image with white on top and bottom (max shift)
        pad_amt = shift_pixels
        pad_top = tf.ones([pad_amt, width, 3], dtype=tf.float32)
        pad_bottom = tf.ones([pad_amt, width, 3], dtype=tf.float32)
        padded = tf.concat([pad_top, image, pad_bottom], axis=0)

        # Compute start row based on shift
        start_row = pad_amt - shift
        shifted = tf.slice(padded, [start_row, 0, 0], [height, width, 3])

        return shifted

    def _load_image(self, path: str) -> tf.Tensor:
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(img, self.image_size)

    def _save_image(self, image: tf.Tensor, out_path: Path):
        img_uint8 = tf.image.convert_image_dtype(image, tf.uint8)
        encoded = tf.io.encode_png(img_uint8)
        tf.io.write_file(str(out_path), encoded)

    # === Main Driver ===
    def generate_augmented_images(
        self,
        df: pd.DataFrame,
        mode: Literal["vflip", "random"],
        target_minimum: Optional[int] = None,
        output_base_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Augments images from df and saves them to disk.

        Args:
            df (pd.DataFrame): Original dataset
            mode (str): "vflip" or "random"
            target_minimum (int): Required if mode='random'
            output_base_dir (str): Optional override path for writing images

        Returns:
            pd.DataFrame of new records with 'label_changed' and 'augmentation' metadata
        """
        new_records = []

        if mode == "vflip":
            # ✅ Filter out any already vflipped images
            df = df[~df["filename"].str.contains("_vflip")].copy()

            for _, rec in df.iterrows():
                label = rec["pattern_type"]
                if label not in self.vertical_flip_label_map:
                    continue

                image = self._load_image(rec["source_path"])
                image, new_label = self.vertical_flip(image, label)

                base, ext = os.path.splitext(rec["filename"])
                new_fname = f"{base}_vflip{ext}"

                out_dir = Path(self.config.AUGMENTED_IMAGES_DIR) / label
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / new_fname

                # ✅ If file already exists, skip to avoid duplicate work
                if out_path.exists():
                    continue

                self._save_image(image, out_path)

                new_record = rec.copy()
                new_record["filename"] = new_fname
                new_record["source_path"] = str(out_path)
                new_record["pattern_type"] = new_label
                if self.label_mode == "pattern_only":
                    new_record["label"] = new_label
                else:
                    new_record["label"] = f"{rec['instrument']}__{new_label}"

                new_record["label_changed"] = label != new_label
                new_records.append(new_record)

        elif mode == "random":
            if target_minimum is None:
                raise ValueError("target_minimum is required for mode='random'")

            # 🚫 Filter out any already augmented images
            df = df[~df["filename"].str.contains("_aug_")].copy()

            counts = df["label"].value_counts().to_dict()
            for label, count in counts.items():
                if count >= target_minimum:
                    continue

                samples = df[df["label"] == label].to_dict("records")
                needed = target_minimum - count
                logging.info(f"🔁 Augmenting '{label}' with {needed} random samples.")

                for i in range(needed):
                    rec = random.choice(samples)
                    image = self._load_image(rec["source_path"])
                    pattern_type = rec["pattern_type"]

                    # op = random.choice(["noise", "contrast", "zoom", "shift", "vflip"])
                    op = random.choice(["noise", "contrast", "zoom", "shift"])
                    if op == "noise":
                        image = self.add_gaussian_noise(image)
                    elif op == "contrast":
                        image = self.brightness_contrast(image)
                    elif op == "zoom":
                        image = self.random_zoom(image, BackgroundMode.MOST_COMMON)
                    elif op == "shift":
                        image = self.vertical_shift(image)
                    elif op == "vflip":
                        image, pattern_type = self.vertical_flip(image, pattern_type)

                    base, ext = os.path.splitext(rec["filename"])
                    new_fname = f"{base}_aug_{op}_{i:03d}{ext}"

                    out_dir = Path(self.config.AUGMENTED_IMAGES_DIR) / pattern_type
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / new_fname

                    label_changed = pattern_type != rec["pattern_type"]

                    new_record = rec.copy()
                    new_record["filename"] = new_fname
                    new_record["source_path"] = str(out_path)
                    new_record["pattern_type"] = pattern_type
                    if self.label_mode == "pattern_only":
                        new_record["label"] = pattern_type
                    else:
                        new_record["label"] = f"{rec['instrument']}__{pattern_type}"

                    new_record["augmentation"] = op
                    new_record["label_changed"] = label_changed
                    new_records.append(new_record)

                    self._save_image(image, out_path)

            # 🧪 Check: only vflip changed labels
            for r in new_records:
                if r["label_changed"] and r.get("augmentation") != "vflip":
                    raise ValueError(
                        f"❌ Unexpected label change in non-vflip op: {r['augmentation']} / {r['filename']}"
                    )

        return pd.DataFrame(new_records)
