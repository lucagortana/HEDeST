from __future__ import annotations

import io
import sys

from loguru import logger
from tqdm import tqdm


class TqdmToLogger(io.StringIO):
    """
    Output stream for TQDM which will output to logger module instead of
    the StdOut.
    """

    logger = None
    level = None
    buf = ""

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or "INFO"

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.log(self.level, self.buf)


try:
    is_terminal = sys.stdout.isatty()

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=is_terminal)
    logger.add(sys.stdout, format="{time} - {level} - {message}", filter="sub.module")
except ModuleNotFoundError:
    pass
