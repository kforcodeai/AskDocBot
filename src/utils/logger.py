import logging
from typing import Optional
from src.utils.filemanager import FileManager

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
            self.file_manager = FileManager()
            self._setup_logger()

    def _setup_logger(self):
        """Setup file-based logging with both file and console output."""
        formatter = logging.Formatter(
            f"%(asctime)s - [%(name)s] - [Run: {self.file_manager.run_id}] - %(levelname)s - %(message)s"
        )

        # File handler
        file_handler = logging.FileHandler(self.file_manager.current_log_file)
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
        logger = logging.getLogger(name if name else __name__)
        logger.info(f"Logger initialized for run: {self.file_manager.run_id}")
        return logger
