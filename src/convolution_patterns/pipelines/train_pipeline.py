import numpy as np
import tensorflow as tf

from convolution_patterns.config.config import Config
from convolution_patterns.logger_manager import LoggerManager
from convolution_patterns.services.callbacks_service import CallbacksService
from convolution_patterns.services.image_dataset_service import ImageDatasetService
from convolution_patterns.services.model_builder_service import ModelBuilderService
from convolution_patterns.services.training_logger_service import TrainingLoggerService
from convolution_patterns.services.transform_service import TransformService

logging = LoggerManager.get_logger(__name__)


class TrainPipeline:
    def __init__(self, split: str = "train"):
        self.config = Config()
        self.split = split
        self.print_stats = True

        # Initialize CallbacksService with path from config (adjust path as needed)
        callbacks_config_path = self.config.config_path
        if callbacks_config_path is None:
            raise ValueError("Callbacks config path must not be None.")
        self.callbacks_service = CallbacksService(callbacks_config_path)

    def _collect_predictions_and_labels(self, model, dataset):
        y_true = []
        y_pred = []

        for batch_x, batch_y in dataset:
            preds = model.predict(batch_x, verbose=0)
            y_pred.extend(np.argmax(preds, axis=1))
            y_true.extend(
                np.argmax(batch_y.numpy(), axis=1)
            )  # assuming one-hot encoded

        return np.array(y_true), np.array(y_pred)

    def run(self):
        # Load transform config
        path = self.config.transform_config_path
        if path is None:
            raise ValueError("Transform config path must not be None.")

        transform_service = TransformService.from_yaml(path)
        train_pipeline = transform_service.get_pipeline(mode="train")
        val_pipeline = transform_service.get_pipeline(mode="val")

        # Load datasets
        dataset_service = ImageDatasetService()
        train_ds, train_class_names = dataset_service.get_dataset(
            "train", print_stats=self.print_stats, prefetch=False
        )
        val_ds, val_class_names = dataset_service.get_dataset(
            "val", print_stats=self.print_stats, prefetch=False
        )

        # Apply transformations
        train_ds = train_ds.map(
            lambda x, y: (train_pipeline(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
        val_ds = val_ds.map(
            lambda x, y: (val_pipeline(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )

        # Optional caching
        if self.config.cache:
            logging.info("[TrainPipeline] Caching datasets in memory...")
            train_ds = train_ds.cache()
            val_ds = val_ds.cache()

        # Prefetch
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

        # Build model
        num_classes = len(train_class_names)
        model_builder_service = ModelBuilderService()
        model = model_builder_service.build(num_classes=num_classes)

        # Get callbacks from CallbacksService
        callbacks = self.callbacks_service.get_callbacks()

        # Train model
        logging.info("[TrainPipeline] Starting training loop...")
        model_name = model_builder_service.model_name
        logger = TrainingLoggerService(model_name)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.epochs,
            callbacks=callbacks,
            verbose=self.config.verbose if hasattr(self.config, "verbose") else 1,
        )

        # Save training artifacts
        logger.save_history(history)
        logger.save_plots(history)
        logger.save_model(model)
        logger.save_metrics_summary(history)

        # Evaluate model and log artifacts
        y_true, y_pred = self._collect_predictions_and_labels(model, val_ds)
        logger.save_evaluation_artifacts(y_true, y_pred, train_class_names)

        return model, train_ds, val_ds, train_class_names
