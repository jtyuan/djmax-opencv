import sys

from loguru import logger as _logger

logger = _logger

try:
    _logger.remove(0)
except ValueError:
    ...

_logger.add(sys.stdout, level="INFO")


def init_logger():
    _logger.add("runs/logs/log_{time}.log", level="DEBUG")
