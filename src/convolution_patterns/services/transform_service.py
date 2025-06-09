from typing import Optional
import yaml
import tensorflow as tf
from tensorflow.keras.applications import (
    efficientnet,
    resnet,
    mobilenet,
    inception_v3,
    xception
)

PREPROCESS_FN_MAP = {
    "EfficientNetB0": efficientnet.preprocess_input,
    "EfficientNetB1": efficientnet.preprocess_input,
    "ResNet50": resnet.preprocess_input,
    "MobileNetV2": mobilenet.preprocess_input,
    "InceptionV3": inception_v3.preprocess_input,
    "Xception": xception.preprocess_input,
}

from convolution_patterns.config.transform_config import TransformConfig


class TransformService:
    PREPROCESS_FN_MAP = {
        "EfficientNetB0": efficientnet.preprocess_input,
        "EfficientNetB1": efficientnet.preprocess_input,
        "ResNet50": resnet.preprocess_input,
        "MobileNetV2": mobilenet.preprocess_input,
        "InceptionV3": inception_v3.preprocess_input,
        "Xception": xception.preprocess_input,
    }
    def __init__(self, transform_config: TransformConfig, model_name: Optional[str] = None):
        self.config = transform_config
        self.model_name = model_name

    @classmethod
    def from_yaml(cls, path: Optional[str], model_name: Optional[str] = None):
        if path is None:
            raise ValueError("Transform config path must be provided (got None).")
        return cls(TransformConfig.from_yaml(path), model_name=model_name)

    def get_pipeline(self, mode: str = "train") -> tf.keras.Sequential:
        layers = [
            tf.keras.layers.Resizing(*self.config.image_size)
        ]

        # Optionally use built-in preprocessing for pretrained models
        if self.model_name and self.model_name in PREPROCESS_FN_MAP:
            preprocess_fn = PREPROCESS_FN_MAP[self.model_name]
            layers.append(tf.keras.layers.Lambda(preprocess_fn))
        elif self.config.rescale != 1.0:
            layers.append(tf.keras.layers.Rescaling(self.config.rescale))

        # Augmentations (only for training)
        if mode == "train":
            aug = self.config.train_augmentation
            if aug.get("horizontal_flip"):
                layers.append(tf.keras.layers.RandomFlip("horizontal"))
            if aug.get("rotation", 0) > 0:
                layers.append(tf.keras.layers.RandomRotation(aug["rotation"]))
            zoom = self.config.zoom_range
            if isinstance(zoom, tuple):
                layers.append(tf.keras.layers.RandomZoom(height_factor=zoom, width_factor=zoom))
            elif isinstance(zoom, float) and zoom > 0:
                layers.append(tf.keras.layers.RandomZoom(zoom))
            if aug.get("width_shift", 0) > 0 or aug.get("height_shift", 0) > 0:
                layers.append(tf.keras.layers.RandomTranslation(
                    height_factor=aug.get("height_shift", 0),
                    width_factor=aug.get("width_shift", 0)
                ))
            if aug.get("brightness", 0) > 0:
                layers.append(tf.keras.layers.RandomBrightness(aug["brightness"]))
            if aug.get("contrast", 0) > 0:
                layers.append(tf.keras.layers.RandomContrast(aug["contrast"]))

        return tf.keras.Sequential(layers, name=f"{mode}_transform")


    def save_config(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config.to_dict(), f)