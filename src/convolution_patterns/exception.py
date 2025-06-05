import sys
import types
from typing import Optional

from convolution_patterns.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)


def error_message_detail(
    error: Exception, error_detail: Optional[types.ModuleType] = None
) -> str:
    """
    Captures details about the error, including the file name, line number, and error message.
    """
    if error_detail is None:
        return f"{error}"
    exc_info = error_detail.exc_info()
    if exc_info is None or exc_info[2] is None:
        return f"{error}"

    _, _, exc_tb = exc_info
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = (
        f"Error occurred in Python script name [{file_name}] at line number "
        f"[{exc_tb.tb_lineno}] with error message [{str(error)}]"
    )
    return error_message


class CustomException(Exception):
    """
    A custom exception class that provides detailed error messages,
    including the script name, line number, and error details.
    """

    def __init__(
        self, error_message: Exception, error_detail: Optional[types.ModuleType] = None
    ):
        self.original_exception = error_message
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )
        super().__init__(self.error_message)

    def __str__(self) -> str:
        return self.error_message


# Test
if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.error("Divide by Zero")
        raise CustomException(e, sys) from e
