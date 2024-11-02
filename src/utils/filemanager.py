import os
from datetime import datetime
import pytz

class FileManager:
    def __init__(self, base_dir: str = "artifacts"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def generate_timestamped_filename(self, prefix: str, extension: str) -> str:
        """Generate a timestamp-based filename."""
        timestamp = datetime.now(pytz.UTC).strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.base_dir, f"{prefix}_{timestamp}.{extension}")