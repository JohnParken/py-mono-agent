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


def get_logger(name: str) -> _PiLogger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=_resolve_log_level(),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    return _PiLogger(logger)
