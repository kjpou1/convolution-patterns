import os
from typing import Any, Dict, List

import tensorflow as tf
import yaml

from convolution_patterns.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)


class CallbacksService:
    """
    Service to manage TensorFlow training callbacks based on a YAML configuration file.

    This service loads callback configurations, instantiates TensorFlow callbacks dynamically,
    and provides them to the training pipeline. It supports enabling/disabling callbacks and
    tuning their parameters without code changes.

    Key Features:
    - Supports EarlyStopping, ReduceLROnPlateau, ModelCheckpoint out of the box
    - Easily extensible to add more callbacks
    - Uses structured YAML config with 'enabled' flags for each callback
    - Detailed logging of callback creation and errors

    Attributes:
        config_path (str): Path to the YAML configuration file for callbacks.
        config (Dict[str, Any]): Parsed YAML configuration dictionary.
        callbacks (List[tf.keras.callbacks.Callback]): List of instantiated callbacks.

    Example:
        >>> service = CallbacksService(config_path="configs/default_training_profile.yaml")
        >>> callbacks = service.get_callbacks()
        >>> model.fit(..., callbacks=callbacks)
    """

    def __init__(self, config_path: str):
        """
        Initialize the CallbacksService with the path to the YAML config.

        Args:
            config_path (str): Path to the YAML config file.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If the config file is invalid YAML.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Callbacks config file not found: {config_path}")

        self.config_path = config_path
        self.config = self._load_config()
        self.callbacks: List[tf.keras.callbacks.Callback] = []

    def _load_config(self) -> Dict[str, Any]:
        """
        Load and parse the YAML configuration file.

        Returns:
            Dict[str, Any]: Parsed YAML config dictionary.

        Raises:
            yaml.YAMLError: If YAML parsing fails.
        """
        with open(self.config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logging.info("Loaded callbacks config from %s", self.config_path)
        return config

    def get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        """
        Instantiate and return the list of TensorFlow callbacks based on the config.

        Returns:
            List[tf.keras.callbacks.Callback]: List of instantiated callbacks.

        Notes:
            - Only callbacks with 'enabled: true' in config are instantiated.
            - Maps config keys to TensorFlow callback classes.
            - Handles parameter name mapping (e.g., 'filename' -> 'filepath' for ModelCheckpoint).
            - Logs errors but continues processing other callbacks.
        """
        self.callbacks.clear()

        # Mapping from config keys to TensorFlow callback classes
        callback_map = {
            "early_stopping": tf.keras.callbacks.EarlyStopping,
            "lr_scheduler": tf.keras.callbacks.ReduceLROnPlateau,
            "model_checkpoint": tf.keras.callbacks.ModelCheckpoint,
            # Add more callbacks here as needed
        }

        for cb_key, cb_class in callback_map.items():
            cb_cfg = self.config.get(cb_key, {})
            if not cb_cfg.get("enabled", False):
                logging.debug("Callback '%s' is disabled or missing in config", cb_key)
                continue

            # Prepare parameters, excluding 'enabled'
            cb_params = {k: v for k, v in cb_cfg.items() if k != "enabled"}

            # Special parameter mapping for ModelCheckpoint
            if cb_key == "model_checkpoint" and "filename" in cb_params:
                cb_params["filepath"] = cb_params.pop("filename")

            try:
                callback_instance = cb_class(**cb_params)
                self.callbacks.append(callback_instance)
                logging.info(
                    "Instantiated callback '%s' with params: %s", cb_key, cb_params
                )
            except Exception as e:
                logging.error(
                    "Failed to instantiate callback '%s' with params %s: %s",
                    cb_key,
                    cb_params,
                    e,
                )

        return self.callbacks
