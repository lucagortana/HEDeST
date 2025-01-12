from __future__ import annotations

import sys

from loguru import logger

try:
    from tqdm import tqdm

    is_terminal = sys.stdout.isatty()

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=is_terminal)
except ModuleNotFoundError:
    pass
