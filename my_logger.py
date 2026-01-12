import sys

from loguru import logger

logger.remove()  # Remove default logger
logger.add("my_log.log", level="WARNING")
logger.info("Logger initialized")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.debug("This is a debug message")
logger.critical("This is a critical message")
