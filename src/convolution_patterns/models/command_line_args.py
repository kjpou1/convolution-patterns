from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple


@dataclass
class CommandLineArgs:
    """
    Structured command-line arguments for the convolution_patterns pipeline.

    Supports:
    - Ingest pipeline
    - Training pipeline
    - Render-images pipeline
    """

    # === Core CLI ===
    command: str  # Subcommand: 'ingest', 'train', 'render-images', etc.
    config: Optional[str] = None  # Optional path to YAML config file
    debug: bool = False  # Enable verbose logging
    _explicit_args: Set[str] = field(default_factory=set)

    # === Ingest Pipeline ===
    staging_dir: Optional[str] = None  # Path to staging dir (required for ingest)
    preserve_raw: bool = True  # Copy original images to artifacts/data/raw/
    label_mode: str = (
        "pattern_only"  # Label format: 'pattern_only' or 'instrument_specific'
    )
    split_ratios: List[int] = field(
        default_factory=lambda: [70, 15, 15]
    )  # Train/val/test split
    random_seed: int = 42  # Reproducible split shuffling

    # === Training Pipeline ===
    data_dir: Optional[str] = None
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    epochs: int = 10
    transform_config_path: Optional[str] = (
        None  # Optional path to YAML transform config file
    )
    model_config_path: Optional[str] = (
        None  # Optional path to YAML transform config file
    )
    cache: bool = False  # Enable dataset caching during training

    # === Render-Images Pipeline ===
    input_path: Optional[str] = None  # Path to input indicator data (CSV, etc.)
    output_dir: Optional[str] = None  # Directory to save rendered images
    manifest_path: str = "manifest.csv"  # Path to output manifest file
    window_sizes: List[int] = field(
        default_factory=lambda: [21, 19, 17, 15, 13, 11]
    )  # Sliding window sizes
    backend: str = "matplotlib"  # Rendering backend: 'matplotlib' or 'pil'
    image_format: str = "png"  # Output image format: 'png', 'jpg', or 'numpy'
    include_close: bool = True  # Include Close price series in rendered charts
    line_width: Optional[float] = None
    image_margin: int = 0  # Image margin/padding in pixels
