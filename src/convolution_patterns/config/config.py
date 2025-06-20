import os
from typing import Optional, cast

import yaml
from dotenv import load_dotenv

from convolution_patterns.models import SingletonMeta
from convolution_patterns.utils.path_utils import (
    ensure_all_dirs_exist,
    get_project_root,
)


def _was_explicit(args, field: str) -> bool:
    return hasattr(args, "_explicit_args") and field in args._explicit_args


class Config(metaclass=SingletonMeta):
    _is_initialized = False

    def __init__(self):
        if Config._is_initialized:
            return

        load_dotenv()

        self._debug = os.getenv("DEBUG", False)
        self._config_path = os.getenv("CONFIG_PATH", "config/default.yaml")
        self._model_type = None
        self._best_of_all = False
        self._save_best = False

        self.PROJECT_ROOT = get_project_root()
        # Resolve BASE_DIR intelligently
        base_dir_env = os.getenv("BASE_DIR", "artifacts")
        if os.path.isabs(base_dir_env):
            self.BASE_DIR = base_dir_env
        else:
            # Resolve relative to project root (assumed to be two levels up from this file)
            self.BASE_DIR = os.path.join(self.PROJECT_ROOT, base_dir_env)

        self.RAW_DATA_DIR = os.path.join(self.BASE_DIR, "data", "raw")
        self.MODEL_DIR = os.path.join(self.BASE_DIR, "models")
        self.MODEL_FILE_PATH = os.path.join(self.MODEL_DIR, "model.pkl")
        self.PREPROCESSOR_FILE_PATH = os.path.join(self.BASE_DIR, "preprocessor.pkl")
        self.LOG_DIR = os.path.join(self.BASE_DIR, "logs")
        self.HISTORY_DIR = os.path.join(self.BASE_DIR, "history")
        self.HISTORY_FILE_PATH = os.path.join(self.HISTORY_DIR, "training_history.json")
        self.REPORTS_DIR = os.path.join(self.BASE_DIR, "reports")
        self.PROCESSED_DATA_DIR = os.path.join(self.BASE_DIR, "data", "processed")
        self.METADATA_DIR = os.path.join(self.BASE_DIR, "data", "metadata")
        self.RENDERED_IMAGES_DIR = os.path.join(self.BASE_DIR, "data", "rendered")

        self._staging_dir = os.getenv("STAGING_DIR", None)
        self._preserve_raw = str(os.getenv("PRESERVE_RAW", "true")).strip().lower() in [
            "1",
            "true",
            "yes",
        ]
        self._label_mode = os.getenv("LABEL_MODE", "pattern_only")
        self._split_ratios = [
            int(x) for x in os.getenv("SPLIT_RATIOS", "70,15,15").split(",")
        ]
        self._deterministic = os.getenv("DETERMINISTIC", "true").lower() == "true"
        self._random_seed = int(os.getenv("RANDOM_SEED", "42"))

        img_vals = os.getenv("IMAGE_SIZE", "224,224").split(",")
        if len(img_vals) != 2:
            raise ValueError(
                "IMAGE_SIZE must contain exactly two comma-separated values"
            )
        self._image_size: tuple[int, int] = (int(img_vals[0]), int(img_vals[1]))

        # Add check to enforce it's exactly 2
        if len(self._image_size) != 2:
            raise ValueError("IMAGE_SIZE must contain exactly two integers")

        self._batch_size = int(os.getenv("BATCH_SIZE", "32"))
        self._epochs = int(os.getenv("EPOCHS", "10"))

        self._transform_config_path = self._resolve_path(
            os.getenv("TRANSFORM_CONFIG_PATH", "configs/transforms_used.yaml")
        )
        self._cache = str(os.getenv("CACHE", "false")).strip().lower() in [
            "1",
            "true",
            "yes",
        ]
        self._model_config_path = self._resolve_path(
            os.getenv("MODEL_CONFIG_PATH", "configs/backbones/EfficientNetB0.yaml")
        )

        # Inference-specific config
        self._inference_model_path = os.getenv(
            "INFERENCE_MODEL_PATH", os.path.join(self.MODEL_DIR, "best_model.h5")
        )
        self._inference_line_thickness = int(os.getenv("INFERENCE_LINE_THICKNESS", "2"))
        self._inference_window_sizes = [
            int(x)
            for x in os.getenv("INFERENCE_WINDOW_SIZES", "21,19,17,15,13,11").split(",")
        ]
        self._inference_confidence_threshold = float(
            os.getenv("INFERENCE_CONFIDENCE_THRESHOLD", "0.8")
        )

        # === Render-Images Config ===
        self._render_input_path = os.getenv("RENDER_INPUT_PATH", None)
        self._render_output_dir = os.getenv(
            "RENDER_OUTPUT_DIR", self.RENDERED_IMAGES_DIR
        )
        self._render_manifest_path = os.getenv("RENDER_MANIFEST_PATH", "manifest.csv")
        self._render_window_sizes = [
            int(x)
            for x in os.getenv("RENDER_WINDOW_SIZES", "21,19,17,15,13,11").split(",")
        ]
        self._render_backend = os.getenv("RENDER_BACKEND", "matplotlib")
        self._render_image_format = os.getenv("RENDER_IMAGE_FORMAT", "png")

        self._include_close = str(
            os.getenv("INCLUDE_CLOSE", "true")
        ).strip().lower() in ["1", "true", "yes"]

        self._render_line_width = float(os.getenv("RENDER_LINE_WIDTH", "1.5"))

        self._ensure_directories_exist()
        Config._is_initialized = True

    def _ensure_directories_exist(self):
        ensure_all_dirs_exist(
            [
                self.RAW_DATA_DIR,
                self.PROCESSED_DATA_DIR,
                self.METADATA_DIR,
                self.MODEL_DIR,
                self.LOG_DIR,
                self.REPORTS_DIR,
                self.HISTORY_DIR,
                self.RENDERED_IMAGES_DIR,
            ]
        )

    def apply_cli_overrides(self, args):
        if _was_explicit(args, "staging_dir"):
            print(f"[Config] Overriding 'staging_dir' from CLI: {args.staging_dir}")
            self.staging_dir = args.staging_dir

        if _was_explicit(args, "preserve_raw"):
            print(
                f"[Config] Overriding 'preserve_raw' from CLI: {self._preserve_raw} → {args.preserve_raw}"
            )
            self.preserve_raw = args.preserve_raw

        if _was_explicit(args, "label_mode"):
            print(
                f"[Config] Overriding 'label_mode' from CLI: {self._label_mode} → {args.label_mode}"
            )
            self.label_mode = args.label_mode

        if _was_explicit(args, "split_ratios"):
            print(
                f"[Config] Overriding 'split_ratios' from CLI: {self._split_ratios} → {args.split_ratios}"
            )
            self.split_ratios = args.split_ratios

        if _was_explicit(args, "random_seed"):
            print(
                f"[Config] Overriding 'random_seed' from CLI: {self._random_seed} → {args.random_seed}"
            )
            self.random_seed = args.random_seed

        if _was_explicit(args, "image_size"):
            print(f"[Config] Overriding 'image_size' from CLI: {args.image_size}")
            self.image_size = args.image_size

        if _was_explicit(args, "batch_size"):
            print(f"[Config] Overriding 'batch_size' from CLI: {args.batch_size}")
            self.batch_size = args.batch_size

        if _was_explicit(args, "cache"):
            print(f"[Config] Overriding 'cache' from CLI: {self._cache} → {args.cache}")
            self.cache = args.cache

        # === Render-Images CLI Overrides ===
        if _was_explicit(args, "input"):
            print(
                f"[Config] Overriding 'render_input_path' from CLI: {args.input_path}"
            )
            self.render_input_path = args.input_path

        if _was_explicit(args, "output_dir"):
            print(
                f"[Config] Overriding 'render_output_dir' from CLI: {args.output_dir}"
            )
            self.render_output_dir = args.output_dir

        if _was_explicit(args, "manifest"):
            print(
                f"[Config] Overriding 'render_manifest_path' from CLI: {args.manifest_path}"
            )
            self.render_manifest_path = args.manifest_path

        if _was_explicit(args, "window_sizes"):
            print(
                f"[Config] Overriding 'render_window_sizes' from CLI: {args.window_sizes}"
            )
            self.render_window_sizes = args.window_sizes

        if _was_explicit(args, "backend"):
            print(f"[Config] Overriding 'render_backend' from CLI: {args.backend}")
            self.render_backend = args.backend

        if _was_explicit(args, "image_format"):
            print(
                f"[Config] Overriding 'render_image_format' from CLI: {args.image_format}"
            )
            self.render_image_format = args.image_format

        if _was_explicit(args, "include_close"):
            print(
                f"[Config] Overriding 'include_close' from CLI: {self._include_close} → {args.include_close}"
            )
            self.include_close = args.include_close

        if _was_explicit(args, "line_width"):
            print(
                f"[Config] Overriding 'render_line_width' from CLI: {args.line_width}"
            )
            self.render_line_width = args.line_width

    def load_from_yaml(self, path: str):
        """
        Override config values from a YAML config file.
        Logs changes to config values.
        """
        if not os.path.exists(path):
            print(f"[Config] YAML config file not found: {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        print(f"[Config] Loaded YAML config: {path}")

        if "debug" in data:
            print(f"[Config] Overriding 'debug': {self._debug} → {data['debug']}")
            self.debug = data["debug"]

        if "staging_dir" in data:
            print(
                f"[Config] Overriding 'staging_dir': {self._staging_dir} → {data['staging_dir']}"
            )
            self.staging_dir = data["staging_dir"]

        if "preserve_raw" in data:
            print(
                f"[Config] Overriding 'preserve_raw': {self._preserve_raw} → {data['preserve_raw']}"
            )
            self.preserve_raw = data["preserve_raw"]

        if "label_mode" in data:
            print(
                f"[Config] Overriding 'label_mode': {self._label_mode} → {data['label_mode']}"
            )
            self.label_mode = data["label_mode"]

        if "split_ratios" in data:
            print(
                f"[Config] Overriding 'split_ratios': {self._split_ratios} → {data['split_ratios']}"
            )
            self.split_ratios = data["split_ratios"]

        if "random_seed" in data:
            print(
                f"[Config] Overriding 'random_seed': {self._random_seed} → {data['random_seed']}"
            )
            self.random_seed = data["random_seed"]

        if "image_size" in data:
            val = data["image_size"]
            if (
                not isinstance(val, (list, tuple))
                or len(val) != 2
                or not all(isinstance(x, int) for x in val)
            ):
                raise ValueError(
                    "image_size from YAML must be a list or tuple of two integers."
                )
            print(f"[Config] Overriding 'image_size': {self._image_size} → {val}")
            self.image_size = val

        if "batch_size" in data:
            val = data["batch_size"]
            if not isinstance(val, int):
                raise ValueError("batch_size from YAML must be an integer.")
            print(f"[Config] Overriding 'batch_size': {self._batch_size} → {val}")
            self.batch_size = val

        if "epochs" in data:
            val = data["epochs"]
            if not isinstance(val, int):
                raise ValueError("epochs from YAML must be an integer.")
            print(f"[Config] Overriding 'epochs': {self._epochs} → {val}")
            self.epochs = val

        if "cache" in data:
            print(f"[Config] Overriding 'cache': {self._cache} → {data['cache']}")
            self.cache = data["cache"]

        # === Render-Images YAML Config ===
        render_config = data.get("render_images", {})

        if "input_path" in render_config:
            print(
                f"[Config] Overriding 'render_input_path': {self._render_input_path} → {render_config['input_path']}"
            )
            self.render_input_path = render_config["input_path"]

        if "output_dir" in render_config:
            print(
                f"[Config] Overriding 'render_output_dir': {self._render_output_dir} → {render_config['output_dir']}"
            )
            self.render_output_dir = render_config["output_dir"]

        if "manifest_path" in render_config:
            print(
                f"[Config] Overriding 'render_manifest_path': {self._render_manifest_path} → {render_config['manifest_path']}"
            )
            self.render_manifest_path = render_config["manifest_path"]

        if "window_sizes" in render_config:
            val = render_config["window_sizes"]
            if not (isinstance(val, list) and all(isinstance(x, int) for x in val)):
                raise ValueError(
                    "render_images.window_sizes from YAML must be a list of integers."
                )
            print(
                f"[Config] Overriding 'render_window_sizes': {self._render_window_sizes} → {val}"
            )
            self.render_window_sizes = val

        if "backend" in render_config:
            val = render_config["backend"]
            if val not in ["matplotlib", "pil"]:
                raise ValueError(
                    "render_images.backend from YAML must be 'matplotlib' or 'pil'."
                )
            print(
                f"[Config] Overriding 'render_backend': {self._render_backend} → {val}"
            )
            self.render_backend = val

        if "image_format" in render_config:
            val = render_config["image_format"]
            if val not in ["png", "jpg", "numpy"]:
                raise ValueError(
                    "render_images.image_format from YAML must be 'png', 'jpg', or 'numpy'."
                )
            print(
                f"[Config] Overriding 'render_image_format': {self._render_image_format} → {val}"
            )
            self.render_image_format = val

        if "include_close" in render_config:
            val = render_config["include_close"]
            if not isinstance(val, bool):
                raise ValueError(
                    "render_images.include_close from YAML must be a boolean."
                )
            print(f"[Config] Overriding 'include_close': {self._include_close} → {val}")
            self.include_close = val

        if "line_width" in render_config:
            val = render_config["line_width"]
            if not isinstance(val, (float, int)):
                raise ValueError(
                    "render_images.line_width from YAML must be a float or int."
                )
            print(
                f"[Config] Overriding 'render_line_width': {self._render_line_width} → {val}"
            )
            self.render_line_width = val

    # === Existing Properties (unchanged) ===
    @property
    def config_path(self):
        return self._config_path

    @config_path.setter
    def config_path(self, value):
        if not isinstance(value, str):
            raise ValueError("config_path must be a string.")
        self._config_path = value

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, value):
        if not isinstance(value, bool):
            raise ValueError("debug must be a boolean.")
        self._debug = value

    @property
    def staging_dir(self):
        return self._resolve_path(self._staging_dir)

    @staging_dir.setter
    def staging_dir(self, value):
        if not isinstance(value, str):
            raise ValueError("staging_dir must be a string.")
        self._staging_dir = value

    @property
    def preserve_raw(self):
        return self._preserve_raw

    @preserve_raw.setter
    def preserve_raw(self, value):
        if not isinstance(value, bool):
            raise ValueError("preserve_raw must be a boolean.")
        self._preserve_raw = value

    @property
    def label_mode(self):
        return self._label_mode

    @label_mode.setter
    def label_mode(self, value):
        if value not in ["pattern_only", "instrument_specific"]:
            raise ValueError(
                "label_mode must be 'pattern_only' or 'instrument_specific'."
            )
        self._label_mode = value

    @property
    def split_ratios(self):
        return self._split_ratios

    @split_ratios.setter
    def split_ratios(self, value):
        if (
            not isinstance(value, list)
            or len(value) != 3
            or not all(isinstance(x, int) for x in value)
        ):
            raise ValueError("split_ratios must be a list of three integers.")
        self._split_ratios = value

    @property
    def deterministic(self) -> bool:
        return self._deterministic

    @deterministic.setter
    def deterministic(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("deterministic must be a boolean.")
        self._deterministic = value

    @property
    def random_seed(self):
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value):
        if not isinstance(value, int):
            raise ValueError("random_seed must be an integer.")
        self._random_seed = value

    @property
    def image_size(self) -> tuple[int, int]:
        return cast(tuple[int, int], self._image_size)

    @image_size.setter
    def image_size(self, value):
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or not all(isinstance(x, int) for x in value)
        ):
            raise ValueError("image_size must be a tuple of exactly two integers.")
        self._image_size = (int(value[0]), int(value[1]))

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if not isinstance(value, int):
            raise ValueError("batch_size must be an integer.")
        self._batch_size = value

    @property
    def epochs(self) -> int:
        return self._epochs

    @epochs.setter
    def epochs(self, value):
        if not isinstance(value, int):
            raise ValueError("epochs must be an integer.")
        self._epochs = value

    @property
    def transform_config_path(self) -> Optional[str]:
        return self._transform_config_path

    @transform_config_path.setter
    def transform_config_path(self, value: str):
        if not isinstance(value, str):
            raise ValueError("transform_config_path must be a string.")
        print(
            f"[Config] Overriding 'transform_config_path': {self._transform_config_path} → {value}"
        )
        self._transform_config_path = self._resolve_path(value)

    @property
    def cache(self) -> bool:
        return self._cache

    @cache.setter
    def cache(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("cache must be a boolean.")
        self._cache = value

    @property
    def model_config_path(self) -> Optional[str]:
        return self._model_config_path

    @model_config_path.setter
    def model_config_path(self, value: str):
        if not isinstance(value, str):
            raise ValueError("model_config_path must be a string.")
        print(
            f"[Config] Overriding 'model_config_path': {self._model_config_path} → {value}"
        )
        self._model_config_path = self._resolve_path(value)

    @property
    def inference_model_path(self) -> str:
        return self._inference_model_path

    @inference_model_path.setter
    def inference_model_path(self, value: str):
        if not isinstance(value, str):
            raise ValueError("inference_model_path must be a string.")
        self._inference_model_path = value

    @property
    def line_thickness(self) -> int:
        return self._inference_line_thickness

    @line_thickness.setter
    def line_thickness(self, value: int):
        if not isinstance(value, int):
            raise ValueError("line_thickness must be an integer.")
        self._inference_line_thickness = value

    @property
    def window_sizes(self) -> list:
        return self._inference_window_sizes

    @window_sizes.setter
    def window_sizes(self, value: list):
        if not (isinstance(value, list) and all(isinstance(x, int) for x in value)):
            raise ValueError("window_sizes must be a list of integers.")
        self._inference_window_sizes = value

    @property
    def confidence_threshold(self) -> float:
        return self._inference_confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value: float):
        if not isinstance(value, (float, int)):
            raise ValueError("confidence_threshold must be a float.")
        self._inference_confidence_threshold = float(value)

    # === Render-Images Properties ===
    @property
    def render_input_path(self) -> Optional[str]:
        return self._render_input_path

    @render_input_path.setter
    def render_input_path(self, value: Optional[str]):
        if value is not None and not isinstance(value, str):
            raise ValueError("render_input_path must be a string or None.")
        self._render_input_path = self._resolve_path(value)

    @property
    def render_output_dir(self) -> str:
        return self._render_output_dir

    @render_output_dir.setter
    def render_output_dir(self, value: str):
        if not isinstance(value, str):
            raise ValueError("render_output_dir must be a string.")
        self._render_output_dir = self._resolve_path(value)

    @property
    def render_manifest_path(self) -> str:
        return self._render_manifest_path

    @render_manifest_path.setter
    def render_manifest_path(self, value: str):
        if not isinstance(value, str):
            raise ValueError("render_manifest_path must be a string.")
        self._render_manifest_path = self._resolve_path(value)

    @property
    def render_window_sizes(self) -> list[int]:
        return self._render_window_sizes

    @render_window_sizes.setter
    def render_window_sizes(self, value: list[int]):
        if not (isinstance(value, list) and all(isinstance(x, int) for x in value)):
            raise ValueError("render_window_sizes must be a list of integers.")
        self._render_window_sizes = value

    @property
    def render_backend(self) -> str:
        return self._render_backend

    @render_backend.setter
    def render_backend(self, value: str):
        if value not in ["matplotlib", "pil"]:
            raise ValueError("render_backend must be 'matplotlib' or 'pil'.")
        self._render_backend = value

    @property
    def render_image_format(self) -> str:
        return self._render_image_format

    @render_image_format.setter
    def render_image_format(self, value: str):
        if value not in ["png", "jpg", "numpy"]:
            raise ValueError("render_image_format must be 'png', 'jpg', or 'numpy'.")
        self._render_image_format = value

    @property
    def include_close(self) -> bool:
        return self._include_close

    @include_close.setter
    def include_close(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("include_close must be a boolean.")
        self._include_close = value

    @property
    def render_line_width(self) -> float:
        return self._render_line_width

    @render_line_width.setter
    def render_line_width(self, value: float):
        if not isinstance(value, (float, int)):
            raise ValueError("render_line_width must be a float or int.")
        self._render_line_width = float(value)

    def _resolve_path(self, val: Optional[str]) -> Optional[str]:
        if not val:
            return None
        if os.path.isabs(val):
            return val
        # Tier 1: try resolving relative to BASE_DIR
        base_resolved = os.path.join(self.BASE_DIR, val)
        if os.path.exists(base_resolved):
            print(f"[Config] Resolved (BASE_DIR): {val} → {base_resolved}")
            return base_resolved
        # Tier 2: try resolving relative to PROJECT_ROOT
        root_resolved = os.path.join(self.PROJECT_ROOT, val)
        if os.path.exists(root_resolved):
            print(f"[Config] Resolved (PROJECT_ROOT): {val} → {root_resolved}")
            return root_resolved
        # Fallback: assume BASE_DIR anyway
        fallback = base_resolved
        print(f"[Config] Resolved (fallback to BASE_DIR): {val} → {fallback}")
        return fallback

    @classmethod
    def initialize(cls):
        if not cls._is_initialized:
            cls()

    @classmethod
    def is_initialized(cls):
        return cls._is_initialized

    @classmethod
    def reset(cls):
        cls._is_initialized = False
        cls._instances = {}
