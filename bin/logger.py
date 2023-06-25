import os
import logging
import logging.handlers

from loguru import logger

class Logger:

    def __init__(self, logPath, logLevel=None):
        self.logPath = logPath
        self.logLevel = logLevel if logLevel is not None else "info"

        logger.add(logPath, level=self.logLevel)

    def info(self, message):
        logger.info(message)

    def debug(self, message):
        logger.debug(message)
