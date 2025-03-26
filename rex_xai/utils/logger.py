#!/usr/bin/env python
"""simple wrapper around python logging module"""

import logging
import sys
import os


def set_log_level(i: int, rex_logger):
    """sets the logging level for rex"""
    if i == 0:
        rex_logger.setLevel(logging.CRITICAL)
    elif i == 1:
        rex_logger.setLevel(logging.WARNING)
    elif i == 2:
        rex_logger.setLevel(logging.INFO)
    else:
        rex_logger.setLevel(logging.DEBUG)


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logger = logging.getLogger("ReX")
logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
