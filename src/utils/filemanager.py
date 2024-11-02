import os
from datetime import datetime
import pytz
from typing import NamedTuple

class FileInfo(NamedTuple):
    """Store file paths and related information for a processing run."""
    timestamp: str
    log_path: str
    result_path: str
    run_id: str

class FileManager:
    def __init__(self, base_dir: str = "artifacts", log_dir: str = "logs"):
        self.base_dir = base_dir
        self.log_dir = log_dir
        self._ensure_directories()
        self.current_file_info = self._generate_file_info()

    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def _generate_file_info(self) -> FileInfo:
        """Generate synchronized file paths for logs and results."""
        timestamp = datetime.now(pytz.UTC).strftime("%Y%m%d_%H%M%S")
        run_id = f"qa_run_{timestamp}"
        
        return FileInfo(
            timestamp=timestamp,
            log_path=os.path.join(self.log_dir, f"{run_id}.log"),
            result_path=os.path.join(self.base_dir, f"{run_id}.json"),
            run_id=run_id
        )

    @property
    def current_log_file(self) -> str:
        """Get current log file path."""
        return self.current_file_info.log_path

    @property
    def current_result_file(self) -> str:
        """Get current result file path."""
        return self.current_file_info.result_path

    @property
    def run_id(self) -> str:
        """Get current run ID."""
        return self.current_file_info.run_id