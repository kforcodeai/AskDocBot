import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np
from json import JSONEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@dataclass
class QAResponse:
    """Data class to structure the QA response with relevant metadata."""
    question: str
    answer: str
    confidence_score: float
    relevant_chunks: List[Dict[str, Any]]
    source_pages: List[int]

class QAEncoder(JSONEncoder):
    """Custom JSON encoder to handle QAResponse objects."""
    def default(self, obj):
        if isinstance(obj, QAResponse):
            return asdict(obj)
        return super().default(obj)

@dataclass
class TextChunk:
    """Represents a text chunk with metadata and optional embedding for vector storage."""
    text: str
    page_num: int
    section_title: str = ""
    subsection_title: str = ""
    embedding: Optional[np.ndarray] = None

    def to_dict(self):
        """Convert TextChunk to dictionary, excluding embeddings for JSON compatibility."""
        try:
            return {
                'text': self.text,
                'page_num': self.page_num,
                'section_title': self.section_title,
                'subsection_title': self.subsection_title
            }
        except Exception as e:
            logger.error("Failed to convert TextChunk to dictionary: %s", e)
            raise
