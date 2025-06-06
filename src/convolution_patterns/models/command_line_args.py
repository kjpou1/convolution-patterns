from dataclasses import dataclass, field
from typing import List, Optional, Set


@dataclass
class CommandLineArgs:
    """
    Structured command-line arguments for the regimetry pipeline.

    Arguments are grouped into:
    - CLI command selection
    """

    # === Core CLI ===
    command: str  # Subcommand: 'ingest', 'embed', or 'cluster'
    config: Optional[str]  # Optional path to YAML config file
    debug: bool  # Enable verbose logging
    _explicit_args: Set[str] = field(default_factory=set)

    # === Ingest Command ===
    staging_dir: Optional[str] = None  # Path to staging dir (required for ingest)
    preserve_raw: bool = True  # Copy original images to artifacts/data/raw/
    label_mode: str = (
        "pattern_only"  # Label format: 'pattern_only' or 'instrument_specific'
    )
    split_ratios: List[int] = field(
        default_factory=lambda: [70, 15, 15]
    )  # Train/val/test split
    preserve_raw: bool = True  # Copy original images to artifacts/data/raw/
    random_seed: int = 42  # Reproducible split shuffling
