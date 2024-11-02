from typing import Dict, List
import torch
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from src.core.data_models import TextChunk
import sys


class EmbeddingGenerator:
    def __init__(self, config: Dict, logger):
        """Initialize the embedding model, tokenizer"""
        try:
            self.logger = logger
            self.embedding_model_name = config.get("DEV", "embedding_model_name")
            self.embedding_size = int(config.get("DEV", "embedding_size"))
            self.max_chunk_size = int(config.get("DEV", "max_chunk_size"))
            self.min_chunk_size = int(config.get("DEV", "min_chunk_size"))
            self.max_length = int(config.get("DEV", "max_length"))

            # Initialize the model and tokenizer
            self.model = AutoModel.from_pretrained(self.embedding_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.logger.info("Initialized Vectorizer successfully.")
        except Exception as e:
            self.logger.error("Error initializing vectorizer: %s", str(e))
            sys.exit(1)
    
    def compute_embedding(self, text: str) -> np.ndarray:
        """Generate text embeddings using the pre-trained model."""
        try:
            inputs = self.tokenizer(
                text, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu().numpy()[0]
        except Exception as e:
            self.logger.error("Error computing embedding: %s", str(e))
            return np.zeros(self.embedding_size)
    
    def __call__(self, chunks:List[TextChunk]):
        try:
            for chunk in chunks:
                chunk.embedding = self.compute_embedding(chunk.text)
            return chunks
        except Exception as e:
            self.logger.error("Error creating embedding from cunks: %s", str(e))
            return []

        