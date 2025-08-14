import logging
from . import discretization, rule_quality, model
from aerial.rule_extraction import generate_rules, generate_frequent_itemsets

__all__ = [discretization, rule_quality, model, generate_rules, generate_frequent_itemsets]

# Create a package-wide logger
logger = logging.getLogger("aerial")
logger.propagate = True
logger.addHandler(logging.NullHandler())


def setup_logging(level=logging.INFO, propagate=False):  # default to False
    """Configure package logging"""
    logger.setLevel(level)
    logger.propagate = propagate  # disable root logger propagation

    # Remove all existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add new console handler if level is not NOTSET
    if level != logging.NOTSET:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)


setup_logging()

try:
    from ._version import version as __version__
except Exception:  # pragma: no cover
    __version__ = "0.0.0"
