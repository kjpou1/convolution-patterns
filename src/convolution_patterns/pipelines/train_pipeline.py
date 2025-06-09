import tensorflow as tf
from convolution_patterns.config.config import Config
from convolution_patterns.logger_manager import LoggerManager
from convolution_patterns.services.image_dataset_service import ImageDatasetService
from convolution_patterns.services.transform_service import TransformService

logging = LoggerManager.get_logger(__name__)

class TrainPipeline:
    def __init__(self, split: str = "train"):
        self.config = Config()
        self.split = split
        self.print_stats = True


    def run(self):
        # Load transform config
        path = self.config.transform_config_path
        if path is None:
            raise ValueError("Transform config path must not be None.")

        transform_service = TransformService.from_yaml(path)
        train_pipeline = transform_service.get_pipeline(mode="train")
        val_pipeline = transform_service.get_pipeline(mode="val")

        # Load dataset
        dataset_service = ImageDatasetService()
        train_ds, train_class_names = dataset_service.get_dataset("train", print_stats=self.print_stats, prefetch=False)
        val_ds, val_class_names = dataset_service.get_dataset("val", print_stats=self.print_stats, prefetch=False)

        # Apply transform pipeline
        train_ds = train_ds.map(lambda x, y: (train_pipeline(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(lambda x, y: (val_pipeline(x), y), num_parallel_calls=tf.data.AUTOTUNE)

        # prefetch but no batch as the imagedatasetservice already applies batch
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds, train_class_names
