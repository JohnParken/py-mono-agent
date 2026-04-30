"""Small logging compatibility layer used by the current codebase."""

from __future__ import annotations

import logging
import os
from typing import Any


class _PiLogger:
    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(_format_message(message, kwargs), *args)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.info(_format_message(message, kwargs), *args)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(_format_message(message, kwargs), *args)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.error(_format_message(message, kwargs), *args)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.exception(_format_message(message, kwargs), *args)


def _format_message(message: str, extra: dict[str, Any]) -> str:
    if not extra:
        return message
    context = " ".join(f"{key}={value}" for key, value in extra.items())
    return f"{message} [{context}]"


def _resolve_log_level() -> int:
    level_name = os.environ.get("PI_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def configure_logging(level: int | None = None) -> None:
    resolved_level = _resolve_log_level() if level is None else level
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=resolved_level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    else:
        root_logger.setLevel(resolved_level)

    # Keep default INFO logs focused on application events rather than
    # subprocess transport noise from the stdlib event loop.
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> _PiLogger:
    configure_logging()
    logger = logging.getLogger(name)
    return _PiLogger(logger)
