import yaml
from typing import Any, Dict, Tuple, Union

class TransformConfig:
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._validate()

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(config)

    def _validate(self):
        if "image_size" not in self._config:
            raise ValueError("Missing required field: 'image_size'")
        if "rescale" not in self._config:
            raise ValueError("Missing required field: 'rescale'")
        aug = self._config.get("train_augmentation", {})
        if not isinstance(aug, dict):
            raise ValueError("'train_augmentation' must be a dictionary")

        zoom = aug.get("zoom")
        if zoom is not None:
            if isinstance(zoom, (list, tuple)):
                if len(zoom) != 2 or not all(isinstance(z, (float, int)) for z in zoom):
                    raise ValueError("'zoom' must be a float or a [min, max] list of two numbers")

    @property
    def image_size(self) -> Tuple[int, int]:
        return tuple(self._config["image_size"])

    @property
    def rescale(self) -> float:
        return float(self._config["rescale"])

    @property
    def train_augmentation(self) -> dict:
        return self._config.get("train_augmentation", {})

    @property
    def zoom_range(self) -> Union[Tuple[float, float], float, None]:
        zoom = self.train_augmentation.get("zoom")
        if zoom is None:
            return None
        if isinstance(zoom, (list, tuple)) and len(zoom) == 2:
            return float(zoom[0]), float(zoom[1])
        elif isinstance(zoom, (float, int)):
            return float(zoom)
        return None

    def to_dict(self) -> dict:
        return self._config
