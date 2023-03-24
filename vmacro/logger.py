import sys

from loguru import logger as _logger


try:
    _logger.remove(0)
except ValueError:
    ...

_logger.add(sys.stdout, level="DEBUG")
_logger.add("runs/logs/log_{time}.log", level="DEBUG")

logger = _logger
