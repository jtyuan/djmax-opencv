import sys

from loguru import logger as _logger

try:
    _logger.remove(0)
except ValueError:
    ...


def init_logger(log_key, name='main'):
    fmt = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | ' \
          '<cyan>{extra[worker]}</cyan>:<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - ' \
          '<level>{message}</level>'
    _logger.add(
        sys.stdout, format=fmt, level="INFO",
        filter=lambda record: record["extra"].get("worker") == name
    )
    _logger.add(
        f"runs/logs/{log_key}/{name}.log", format=fmt, level="DEBUG",
        filter=lambda record: record["extra"].get("worker") == name
    )
    return _logger.bind(worker=name)
