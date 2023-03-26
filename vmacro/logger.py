import sys

from loguru import logger as _logger


def init_logger(log_key, name='main'):
    try:
        _logger.remove(0)
    except ValueError:
        ...
    fmt = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | ' \
          '<cyan>{extra[worker]}</cyan>:<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - ' \
          '<level>{message}</level>'
    _logger.add(sys.stdout, format=fmt, level="INFO")
    _logger.add(f"runs/logs/{log_key}/{name}.log", format=fmt, level="DEBUG")
    return _logger.bind(worker=name)
