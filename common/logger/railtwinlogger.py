import logging
import sys
from datetime import datetime
from pathlib import Path

_FORMAT = "[%(asctime)s] [RT-Logger] [%(levelname)s] %(message)s"


class RailTwinLogger:
    """
    Shared "RT-Logger" (Python logging). create() returns the logging.Logger; every module logs to the same one.
    Output goes to stderr, register_file_logger() adds a log file.
    """
    name = "RT-Logger"

    @staticmethod
    def create():
        logger = logging.getLogger(RailTwinLogger.name)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter(_FORMAT))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
        return logger

    @staticmethod
    def register_file_logger(path=None):
        """Also log into <path>/<date>-RailTwin.log (default: the log folder next to this package)."""
        folder = Path(path) if path else Path(__file__).parent.parent.resolve() / "log"
        folder.mkdir(parents=True, exist_ok=True)
        log_file = folder / f"{datetime.today().strftime('%Y-%m-%d')}-RailTwin.log"
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger = RailTwinLogger.create()
        logger.addHandler(handler)
        return log_file
