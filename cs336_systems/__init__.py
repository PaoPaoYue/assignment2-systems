import importlib.metadata

__version__ = importlib.metadata.version("cs336-systems")


import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
