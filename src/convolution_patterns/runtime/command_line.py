import argparse

from convolution_patterns.models.command_line_args import CommandLineArgs

from .logging_argument_parser import LoggingArgumentParser


class CommandLine:
    @staticmethod
    def parse_arguments() -> CommandLineArgs:
        """
        Parse command-line arguments and return a CommandLineArgs object.

        Supports subcommands like 'ingest', 'embed', and 'cluster'.
        """
        parser = LoggingArgumentParser(description="convolution_patterns CLI")

        # Create subparsers for subcommands
        subparsers = parser.add_subparsers(dest="command", help="Subcommands")

        # Parse the arguments
        args = parser.parse_args()

        # Check if a subcommand was provided
        if args.command is None:
            parser.print_help()
            exit(1)

        # Return a CommandLineArgs object with parsed values
        return CommandLineArgs(
            command=args.command,
            config=getattr(args, "config", None),
            debug=getattr(args, "debug", False),
        )
