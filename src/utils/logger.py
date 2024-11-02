import logging
from typing import Optional
import os
from datetime import datetime
import pytz

import os
from datetime import datetime
import pytz

class LoggerManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self.log_dir = "logs"
            os.makedirs(self.log_dir, exist_ok=True)
            self.current_log_file = self._generate_log_filename()
            self._setup_file_logger()

    def _generate_log_filename(self) -> str:
        """Generate timestamp-based log filename."""
        timestamp = datetime.now(pytz.UTC).strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.log_dir, f"qa_pipeline_{timestamp}.log")

    def _setup_file_logger(self):
        """Setup file-based logging with rotation."""
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # File handler
        file_handler = logging.FileHandler(self.current_log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)

        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplication
        root_logger.handlers = []
        
        # Add handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """Get a logger instance."""
        return logging.getLogger(name if name else __name__)