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

    class DebugCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            for metric_name, metric_value in logs.items():
                logging.info(
                    "DebugCallback - Epoch %d metric '%s' type: %s value: %s",
                    epoch,
                    metric_name,
                    type(metric_value),
                    metric_value,
                )

    def _parse_early_stopping_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        parsed = {}
        parsed["patience"] = self._parse_int(params.get("patience"), 10)
        parsed["min_delta"] = self._parse_float(params.get("min_delta"), 0.0)
        parsed["verbose"] = self._parse_int(params.get("verbose"), 0)
        # Pass through other params as is
        for key in params:
            if key not in parsed:
                parsed[key] = params[key]
        return parsed

    def _parse_lr_scheduler_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        parsed = {}
        parsed["factor"] = self._parse_float(params.get("factor"), 0.5)
        parsed["patience"] = self._parse_int(params.get("patience"), 5)
        parsed["min_lr"] = self._parse_float(params.get("min_lr"), 1e-6)
        parsed["verbose"] = self._parse_int(params.get("verbose"), 0)
        # Pass through other params as is
        for key in params:
            if key not in parsed:
                parsed[key] = params[key]
        return parsed

    def _parse_model_checkpoint_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # No numeric parsing needed here, but you can add if needed
        # Also handle filename->filepath mapping if you want
        if "filename" in params:
            params["filepath"] = params.pop("filename")
        return params

    def get_callbacks(self) -> List[tf.keras.callbacks.Callback]:
        self.callbacks.clear()

        callback_map = {
            "early_stopping": (
                tf.keras.callbacks.EarlyStopping,
                self._parse_early_stopping_params,
            ),
            "lr_scheduler": (
                tf.keras.callbacks.ReduceLROnPlateau,
                self._parse_lr_scheduler_params,
            ),
            "model_checkpoint": (
                tf.keras.callbacks.ModelCheckpoint,
                self._parse_model_checkpoint_params,
            ),
        }

        for cb_key, (cb_class, parse_fn) in callback_map.items():
            cb_cfg = self.config.get(cb_key, {})
            if not cb_cfg.get("enabled", False):
                logging.debug("Callback '%s' is disabled or missing in config", cb_key)
                continue

            cb_params = {k: v for k, v in cb_cfg.items() if k != "enabled"}
            cb_params = parse_fn(cb_params)

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

        # self.callbacks.append(self.DebugCallback())
        # logging.info("Added DebugCallback to callbacks list")

        return self.callbacks

    def _parse_float(self, value, default):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _parse_int(self, value, default):
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
