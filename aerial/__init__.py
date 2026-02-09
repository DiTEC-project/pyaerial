import logging
import sys
from . import discretization, rule_quality, model
from aerial.rule_extraction import generate_rules, generate_frequent_itemsets

__all__ = [discretization, rule_quality, model, generate_rules, generate_frequent_itemsets]


class ColoredFormatter(logging.Formatter):
    """Formatter with colored output for different log levels."""
    COLORS = {
        'DEBUG': '\033[35m',    # Magenta
        'INFO': '\033[94m',     # Light blue
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Light red
        'CRITICAL': '\033[91;1m', # Bold red
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        message = super().format(record)
        return f"{color}{message}{self.RESET}" if color else message


# Create a package-wide logger
logger = logging.getLogger("aerial")
logger.propagate = True
logger.addHandler(logging.NullHandler())


def setup_logging(level=logging.INFO, propagate=False, colors=True):
    """Configure package logging.

    :param level: logging level (default=logging.INFO)
    :param propagate: whether to propagate to root logger (default=False)
    :param colors: whether to use colored output (default=True, disabled if not a tty)
    """
    logger.setLevel(level)
    logger.propagate = propagate

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if level != logging.NOTSET:
        handler = logging.StreamHandler(sys.stderr)
        if colors:
            formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        else:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)


setup_logging()

try:
    from ._version import version as __version__
except Exception:  # pragma: no cover
    __version__ = "0.0.0"
