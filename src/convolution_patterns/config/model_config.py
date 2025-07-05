from typing import Any, Dict, Tuple, Union

import yaml


class ModelConfig:
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._validate()

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(config)

    def _validate(self):
        required_model_keys = ["backbone", "input_shape", "custom_head"]
        required_head_keys = ["layers"]
        required_training_keys = ["optimizer", "learning_rate", "loss"]

        model = self._config.get("model")
        if not isinstance(model, dict):
            raise ValueError("'model' block must be a dictionary")

        freeze_backbone = model.get("freeze_backbone")
        if freeze_backbone is not None and not isinstance(freeze_backbone, bool):
            raise ValueError("'freeze_backbone' must be a boolean if specified")

        unfreeze_from = model.get("unfreeze_from_layer")
        if unfreeze_from is not None and not isinstance(unfreeze_from, int):
            raise ValueError("'unfreeze_from_layer' must be an integer if specified")

        for key in required_model_keys:
            if key not in model:
                raise ValueError(f"Missing required key in 'model': '{key}'")

        custom_head = model["custom_head"]
        if not isinstance(custom_head, dict):
            raise ValueError("'custom_head' must be a dictionary")

        for key in required_head_keys:
            if key not in custom_head:
                raise ValueError(f"Missing required key in 'custom_head': '{key}'")

        if not isinstance(custom_head["layers"], list):
            raise ValueError("'custom_head.layers' must be a list")

        for i, layer in enumerate(custom_head["layers"]):
            if not isinstance(layer, dict):
                raise ValueError(f"Layer at index {i} must be a dictionary")
            if "type" not in layer:
                raise ValueError(f"Layer at index {i} missing required 'type' field")

        training = self._config.get("training")
        if not isinstance(training, dict):
            raise ValueError("'training' block must be a dictionary")

        for key in required_training_keys:
            if key not in training:
                raise ValueError(f"Missing required key in 'training': '{key}'")

        input_shape = model.get("input_shape")
        if not (isinstance(input_shape, (list, tuple)) and len(input_shape) == 3):
            raise ValueError("'input_shape' must be a list or tuple of 3 integers")

    @property
    def model(self) -> Dict[str, Any]:
        if "model" not in self._config:
            raise KeyError("Missing 'model' block in config")
        return self._config["model"]

    @property
    def training(self) -> Dict[str, Any]:
        if "training" not in self._config:
            raise KeyError("Missing 'training' block in config")
        return self._config["training"]

    def to_dict(self) -> dict:
        return self._config

    def summary(self) -> str:
        return yaml.dump(self._config, sort_keys=False, default_flow_style=False)
