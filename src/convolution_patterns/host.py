import asyncio
import os

from convolution_patterns.config.config import Config
from convolution_patterns.exception import CustomException
from convolution_patterns.logger_manager import LoggerManager
from convolution_patterns.models.command_line_args import CommandLineArgs

logging = LoggerManager.get_logger(__name__)


class Host:
    """
    Host class to manage the execution of the main application.

    This class handles initialization with command-line arguments and
    configuration, and runs the main asynchronous functionality.
    """

    def __init__(self, args: CommandLineArgs):
        """
        Initialize the Host class with command-line arguments and configuration.

        Parameters:
        args (CommandLineArgs): Command-line arguments passed to the script.
        """
        self.args = args
        self.config = Config()
        if args.config:
            self.config.config_path = args.config
            self.config.load_from_yaml(args.config)

        self.config.apply_cli_overrides(args)

    def run(self):
        """
        Synchronously run the asynchronous run_async method.

        This is a blocking call that wraps the asynchronous method.
        """
        return asyncio.run(self.run_async())

    async def run_async(self):
        """
        Main asynchronous method to execute the host functionality.

        Determines the action based on the provided subcommand.
        """
        try:
            logging.info("Starting host operations.")

            if self.args.command == "ingest":
                logging.info("Executing data ingestion workflow.")
                await self.run_ingestion()

            else:
                logging.error("No valid subcommand provided.")
                raise ValueError(
                    "Please specify a valid subcommand: 'ingest', 'embed', or 'cluster'."
                )

        except CustomException as e:
            logging.error("A custom error occurred during host operations: %s", e)
            raise
        except Exception as e:
            logging.error("An unexpected error occurred: %s", e)
            raise  # CustomException("An unexpected error occured", sys) from e
        finally:
            logging.info("Shutting down host gracefully.")

    async def run_ingestion(self):
        """
        Run the supervised ingestion pipeline from CSV input to processed train/val/test outputs.
        """
        logging.info("Initializing ingestion pipeline...")
        pass
