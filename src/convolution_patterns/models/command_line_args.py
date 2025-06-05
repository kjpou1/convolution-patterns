from dataclasses import dataclass
from typing import Optional


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
