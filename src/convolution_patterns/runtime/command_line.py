import sys

from convolution_patterns.models.command_line_args import CommandLineArgs
from convolution_patterns.utils.argparse_utils import parse_image_size

from .logging_argument_parser import LoggingArgumentParser


class CommandLine:
    @staticmethod
    def parse_arguments() -> CommandLineArgs:
        """
        Parse command-line arguments and return a CommandLineArgs object.

        Supports subcommands like 'ingest', 'train', 'inference', and 'render-images'.
        """
        parser = LoggingArgumentParser(
            description="Convolution CV CLI for chart pattern ingestion, model training, image inference, and image rendering workflows."
        )

        # Create subparsers for subcommands
        subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

        # === INGEST Subcommand ===
        ingest_parser = subparsers.add_parser(
            "ingest", help="Run the ingestion pipeline"
        )
        ingest_parser.add_argument(
            "--staging-dir", type=str, help="Path to staging directory"
        )
        ingest_parser.add_argument(
            "--no-preserve-raw",
            dest="preserve_raw",
            action="store_false",
            help="Disable copying raw images to artifacts/data/raw/",
        )
        ingest_parser.set_defaults(preserve_raw=True)

        ingest_parser.add_argument(
            "--label-mode",
            type=str,
            default="pattern_only",
            choices=["pattern_only", "instrument_specific"],
            help="Labeling mode to use",
        )
        ingest_parser.add_argument(
            "--split-ratios",
            type=int,
            nargs=3,
            default=[70, 15, 15],
            help="Train/val/test split ratios (default: 70 15 15)",
        )
        ingest_parser.add_argument(
            "--random-seed",
            type=int,
            default=42,
            help="Random seed for reproducibility",
        )
        ingest_parser.add_argument(
            "--config", type=str, help="Path to YAML config file"
        )
        ingest_parser.add_argument(
            "--debug", action="store_true", help="Enable debug logging"
        )

        # === TRAIN Subcommand ===
        train_parser = subparsers.add_parser(
            "train", help="Run the model training pipeline"
        )

        train_parser.add_argument(
            "--image-size",
            nargs=2,
            metavar=("HEIGHT", "WIDTH"),
            default=[224, 224],
            help="Target image size as HEIGHT WIDTH (e.g. 224 224)",
        )

        train_parser.add_argument(
            "--batch-size",
            type=int,
            default=32,
            help="Batch size to use for training and validation",
        )

        train_parser.add_argument(
            "--epochs", type=int, default=10, help="Number of training epochs"
        )

        train_parser.add_argument(
            "--transform-config",
            type=str,
            help="Path to transform config YAML (default: configs/transforms_used.yaml)",
        )

        train_parser.add_argument(
            "--model-config",
            type=str,
            help="Path to model config YAML (defines backbone, head, loss, etc.)",
        )

        train_parser.add_argument(
            "--cache",
            action="store_true",
            help="Enable dataset caching to speed up training (in-memory only)",
        )

        train_parser.add_argument(
            "--debug", action="store_true", help="Enable debug logging"
        )

        train_parser.add_argument(
            "--config", type=str, help="Optional path to training config file (YAML)"
        )

        # === RENDER-IMAGES Subcommand ===
        render_parser = subparsers.add_parser(
            "render-images", help="Render chart images from indicator data"
        )

        render_parser.add_argument(
            "--input",
            type=str,
            required=True,
            help="Path to input indicator data (CSV, database, etc.)",
        )

        render_parser.add_argument(
            "--output-dir",
            type=str,
            required=True,
            help="Directory to save rendered images",
        )

        render_parser.add_argument(
            "--manifest",
            type=str,
            default="manifest.csv",
            help="Path to output manifest file (default: manifest.csv)",
        )

        render_parser.add_argument(
            "--window-sizes",
            type=int,
            nargs="+",
            default=[21, 19, 17, 15, 13, 11],
            help="Sliding window sizes to render (default: 21 19 17 15 13 11)",
        )

        render_parser.add_argument(
            "--backend",
            type=str,
            choices=["matplotlib", "pil"],
            default="matplotlib",
            help="Rendering backend to use (default: matplotlib)",
        )

        render_parser.add_argument(
            "--image-format",
            type=str,
            choices=["png", "jpg", "numpy"],
            default="png",
            help="Output image format (default: png)",
        )

        render_parser.add_argument(
            "--image-size",
            nargs=2,
            metavar=("HEIGHT", "WIDTH"),
            type=int,
            default=[224, 224],
            help="Output image size as HEIGHT WIDTH (default: 224 224)",
        )

        render_parser.add_argument(
            "--image-margin",
            type=int,
            default=0,
            help="Image margin/padding in pixels (default: 0)",
        )

        render_parser.add_argument(
            "--include-close",
            action="store_true",
            default=True,
            help="Include Close price series in rendered charts (default: True)",
        )

        render_parser.add_argument(
            "--no-include-close",
            dest="include_close",
            action="store_false",
            help="Exclude Close price series from rendered charts",
        )

        render_parser.add_argument(
            "--line-width",
            type=float,
            default=1.5,
            help="Line width for rendered charts (default: 1.5)",
        )

        render_parser.add_argument(
            "--config", type=str, help="Path to rendering config YAML file"
        )

        render_parser.add_argument(
            "--debug", action="store_true", help="Enable debug logging"
        )

        # Parse the arguments
        args = parser.parse_args()

        # Check if a subcommand was provided
        if args.command is None:
            parser.print_help()
            exit(1)

        # Determine which subparser was used
        command = args.command

        # Get subparser object based on command
        subparser = {
            "ingest": ingest_parser,
            "train": train_parser,
            "render-images": render_parser,
        }.get(command)

        # Track which args were explicitly passed on CLI
        args._explicit_args = set()
        if subparser:
            for action in subparser._actions:
                for opt in action.option_strings:
                    if opt in sys.argv:
                        args._explicit_args.add(action.dest)

        # If config is NOT specified, validate that required CLI arguments are present
        if args.command == "ingest":
            CommandLine._validate_ingest_args(args, parser)
        if args.command == "train":
            CommandLine._validate_train_args(args, parser)
        if args.command == "render-images":
            CommandLine._validate_render_args(args, parser)

        # Return a CommandLineArgs object with parsed values
        return CommandLineArgs(
            command=args.command,
            _explicit_args=getattr(args, "_explicit_args", set()),
            config=getattr(args, "config", None),
            debug=getattr(args, "debug", False),
            # Ingest args
            staging_dir=getattr(args, "staging_dir", None),
            preserve_raw=getattr(args, "preserve_raw", True),
            label_mode=getattr(args, "label_mode", "pattern_only"),
            split_ratios=getattr(args, "split_ratios", [70, 15, 15]),
            random_seed=getattr(args, "random_seed", 42),
            # Train args
            data_dir=getattr(args, "data_dir", None),
            image_size=parse_image_size(getattr(args, "image_size", [224, 224])),
            batch_size=getattr(args, "batch_size", 32),
            epochs=getattr(args, "epochs", 10),
            transform_config_path=getattr(args, "transform_config", None),
            model_config_path=getattr(args, "model_config", None),
            cache=getattr(args, "cache", False),
            # Render-images args
            input_path=getattr(args, "input", None),
            output_dir=getattr(args, "output_dir", None),
            manifest_path=getattr(args, "manifest", "manifest.csv"),
            window_sizes=getattr(args, "window_sizes", [21, 19, 17, 15, 13, 11]),
            backend=getattr(args, "backend", "matplotlib"),
            image_format=getattr(args, "image_format", "png"),
            include_close=getattr(args, "include_close", True),
            line_width=getattr(args, "line_width", 1.5),
            image_margin=getattr(args, "image_margin", 0),
        )

    @staticmethod
    def _validate_ingest_args(args, parser):
        """
        Validate required ingest arguments if no config is provided.
        """
        if args.config is None:
            missing = []
            if not args.staging_dir:
                missing.append("--staging-dir")
            if missing:
                parser.error(
                    f"Missing required arguments: {', '.join(missing)} (required if no --config is given)"
                )

    @staticmethod
    def _validate_train_args(args, parser):
        """
        Validate required train arguments if no config file is used.
        """
        if args.config is None:
            missing = []
            if not args.data_dir:
                missing.append("--data-dir")
            if not args.image_size or len(args.image_size) != 2:
                missing.append("--image-size")
            if missing:
                parser.error(
                    f"Missing required arguments for training: {', '.join(missing)} (required if no --config is given)"
                )

    @staticmethod
    def _validate_render_args(args, parser):
        """
        Validate required render-images arguments if no config file is used.
        """
        if args.config is None:
            missing = []
            if not args.input:
                missing.append("--input")
            if not args.output_dir:
                missing.append("--output-dir")
            if missing:
                parser.error(
                    f"Missing required arguments for render-images: {', '.join(missing)} (required if no --config is given)"
                )
