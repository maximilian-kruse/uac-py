"""Logging utilities for the construction pipeline.

Classes:
    LoggerSettings: Settings for the output channels of a logger.
    BaseLogger: Logger writing messages to the console and/or a log file.

Functions:
    format_key: Render a key sequence for logging.
    format_range: Render the extent of an array for logging.
    resolve_logger: Turn an optional logger argument into a logger.
    format_table_row: Pad one row of a log table to fixed column widths.
"""

import itertools
import logging
import sys
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Literal, Self, override

import numpy as np

from uacpy.common.dict_utils import KeySequence

_instance_counter = itertools.count()

_SECTION_RULE_WIDTH = 80
_SECTION_RULE = "=" * _SECTION_RULE_WIDTH


@dataclass
class LoggerSettings:
    """Settings for the output channels of a logger.

    Attributes:
        print_to_console: Whether to print messages to standard output. Defaults to `True`.
        logfile_path: File to write messages to. Missing parent directories are created. If `None`,
            no log file is written. Defaults to `None`.
        write_mode: Whether to overwrite (`"w"`) or append to (`"a"`) an existing log file.
            Defaults to `"w"`.
    """

    print_to_console: bool = True
    logfile_path: Path | None = None
    write_mode: Literal["w", "a"] = "w"


class _LevelAwareFormatter(logging.Formatter):
    """Formatter that writes info messages as they are and tags all other levels."""

    def __init__(self) -> None:
        super().__init__()
        self._info_formatter = logging.Formatter("%(message)s")
        self._level_formatter = logging.Formatter("[%(levelname)s] %(message)s")

    @override
    def format(self, record: logging.LogRecord) -> str:
        """Format a record as `message`, or `[LEVEL] message` for non-info levels.

        Args:
            record: Record to format.

        Returns:
            The formatted message.
        """
        formatter = (
            self._info_formatter if record.levelno == logging.INFO else self._level_formatter
        )
        return formatter.format(record)


class BaseLogger:
    """Logger writing messages to the console and/or a log file.

    Every instance owns a separate, non-propagating Python logger, so that instances with different
    output channels do not interfere with each other or with the root logger. The log file, if any,
    receives all levels including debug messages; the console only receives info level and above.

    The logger holds an open log file until it is closed, either explicitly via `close`, by using
    it as a context manager, or at the latest when it is garbage collected or the interpreter
    exits. Logging to a closed logger raises an error.

    Attributes:
        closed: Whether the logger has been closed.
    """

    def __init__(self, logger_settings: LoggerSettings) -> None:
        """Initialize the logger and its output channels.

        Args:
            logger_settings: Output channel settings.
        """
        pylogger_name = f"{__name__}.{next(_instance_counter)}"
        self._pylogger = logging.getLogger(pylogger_name)
        self._pylogger.setLevel(logging.DEBUG)
        self._pylogger.propagate = False
        formatter = _LevelAwareFormatter()

        if logger_settings.print_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self._pylogger.addHandler(console_handler)

        if logger_settings.logfile_path is not None:
            logger_settings.logfile_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                logger_settings.logfile_path, mode=logger_settings.write_mode
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self._pylogger.addHandler(file_handler)

        # The finalizer must not reference self, otherwise the logger is never garbage collected.
        self._finalizer = weakref.finalize(self, _remove_handlers, self._pylogger)

    @property
    def closed(self) -> bool:
        """Whether the logger has been closed."""
        return not self._finalizer.alive

    def close(self) -> None:
        """Flush and release all output channels.

        The log file is closed, the console stream is left open. Closing an already closed logger
        has no effect.
        """
        self._finalizer()

    def __enter__(self) -> Self:
        """Return the logger for use in a `with` statement, which closes it on exit."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the logger when leaving the `with` statement."""
        self.close()

    def info(self, message: str) -> None:
        """Log an info message.

        Args:
            message: Message to log.

        Raises:
            ValueError: If the logger has been closed.
        """
        self._check_open()
        self._pylogger.info(message)

    def debug(self, message: str) -> None:
        """Log a debug message.

        Args:
            message: Message to log.

        Raises:
            ValueError: If the logger has been closed.
        """
        self._check_open()
        self._pylogger.debug(message)

    def warning(self, message: str) -> None:
        """Log a warning.

        Args:
            message: Message to log.

        Raises:
            ValueError: If the logger has been closed.
        """
        self._check_open()
        self._pylogger.warning(message)

    def exception(self, message: str) -> None:
        """Log an error message together with the traceback of the current exception.

        Only call this from within an exception handler.

        Args:
            message: Message to log.

        Raises:
            ValueError: If the logger has been closed.
        """
        self._check_open()
        self._pylogger.exception(message)

    def error(self, message: str) -> None:
        """Log an error message.

        Args:
            message: Message to log.

        Raises:
            ValueError: If the logger has been closed.
        """
        self._check_open()
        self._pylogger.error(message)

    def section(self, title: str) -> None:
        """Log a section header between two horizontal rules, preceded by a blank line.

        Args:
            title: Name of the pipeline section that follows.

        Raises:
            ValueError: If the logger has been closed.
        """
        self.info("")
        self.info(_SECTION_RULE)
        self.info(title.upper())
        self.info(_SECTION_RULE)

    def _check_open(self) -> None:
        """Raise if the logger has been closed.

        Without handlers, Python would silently route warnings and errors to its last-resort
        handler on stderr instead.
        """
        if self.closed:
            raise ValueError("Cannot log to a closed logger.")


def _remove_handlers(pylogger: logging.Logger) -> None:
    """Detach, flush and close all handlers of a Python logger."""
    for handler in list(pylogger.handlers):
        pylogger.removeHandler(handler)
        handler.close()


def format_key(key_sequence: KeySequence) -> str:
    """Render a key sequence for logging as e.g. `LIPV.inner.mitral`, not its tuple-of-enum repr.

    Args:
        key_sequence: Keys addressing an entry, outermost first.

    Returns:
        The keys joined with dots.
    """
    return ".".join(str(key) for key in key_sequence)


def format_range(values: np.ndarray) -> str:
    """Render the extent of an array for logging as `[min, max]`.

    Args:
        values: Non-empty array of any shape.

    Returns:
        The minimum and maximum, to four significant digits.
    """
    return f"[{values.min():.4g}, {values.max():.4g}]"


def resolve_logger(logger: BaseLogger | None, print_to_console: bool = False) -> BaseLogger:
    """Return the given logger, or a new one without a log file if the caller passed none.

    The single place where an omitted `logger` argument is turned into a logger.

    Args:
        logger: The logger the caller passed, if any.
        print_to_console: Whether the fallback logger prints to standard output. Silent by default,
            for helpers whose caller did not ask for output; pipelines and checks whose result is
            the output itself pass `True`.

    Returns:
        `logger` if given, otherwise a new logger.
    """
    if logger is not None:
        return logger
    return BaseLogger(LoggerSettings(print_to_console=print_to_console))


def format_table_row(cells: Sequence[str], widths: Sequence[int], alignments: str) -> str:
    """Pad the cells of one log table row to fixed column widths.

    Args:
        cells: Text of each column.
        widths: Width of each column in characters.
        alignments: One `str.format` alignment character per column: `<` (left) or `>` (right).

    Returns:
        The columns joined by two spaces, without trailing whitespace.
    """
    padded = (
        f"{cell:{alignment}{width}}"
        for cell, width, alignment in zip(cells, widths, alignments, strict=True)
    )
    return "  ".join(padded).rstrip()
