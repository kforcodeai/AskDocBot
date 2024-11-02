import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass
from src.data_models import TextChunk
import logging

class SemanticVectorStore:
    """A class for managing semantic vector storage and search using FAISS."""

    def __init__(self, dimension: int = 768, logger=None):
        self.logger = logger
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
        self.chunks = []
        self.logger.info("Initialized SemanticVectorStore with dimension %d.", dimension)

    def add_chunks(self, chunks: List[TextChunk]):
        """Add text chunks and their embeddings to the FAISS index."""
        if not chunks:
            self.logger.warning("No chunks provided for addition.")
            return

        embeddings = [chunk.embedding for chunk in chunks if chunk.embedding is not None]
        if not embeddings:
            self.logger.error("Chunks have no embeddings to add to the vector store.")
            raise ValueError("Chunks have no embeddings to add to the vector store.")
        
        # Check for consistency in embedding dimensions
        embedding_dim = embeddings[0].shape[0]
        if embedding_dim != self.dimension:
            self.logger.error("Mismatch in embedding dimensions. Expected: %d, Got: %d", self.dimension, embedding_dim)
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dimension}, got {embedding_dim}")

        embeddings_array = np.array(embeddings, dtype='float32')
        
        try:
            self.index.add(embeddings_array)
            self.chunks.extend(chunks)
            self.logger.info("Added %d chunks to the vector store.", len(chunks))
        except Exception as e:
            self.logger.exception("Failed to add chunks to the vector store: %s", e)
            raise

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[TextChunk, float]]:
        """Search the vector store for similar chunks using a query embedding."""
        if query_embedding is None or query_embedding.shape[0] != self.dimension:
            self.logger.error("Query embedding has an invalid shape: %s", query_embedding.shape)
            raise ValueError("Query embedding has an invalid shape.")
        
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        try:
            distances, indices = self.index.search(query_embedding, k)
            results = [
                (self.chunks[idx], float(dist)) 
                for idx, dist in zip(indices[0], distances[0]) 
                if idx != -1
            ]
            self.logger.info("Search completed, returning %d results.", len(results))
            return results
        except Exception as e:
            self.logger.exception("Failed to perform search in the vector store: %s", e)
            raise

    def save(self, directory: str):
        """Persist vector store and metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        
        try:
            faiss.write_index(self.index, os.path.join(directory, "vector_store.faiss"))
            with open(os.path.join(directory, "chunks.pkl"), "wb") as f:
                pickle.dump([chunk.to_dict() for chunk in self.chunks], f)
            self.logger.info("Vector store and chunks metadata saved successfully in %s.", directory)
        except Exception as e:
            self.logger.exception("Failed to save the vector store: %s", e)
            raise

    @classmethod
    def load(cls, directory: str, nlp, logger=None) -> 'SemanticVectorStore':
        """Load vector store and reconstruct chunks with embeddings from a directory."""
        logger = logger or logging.getLogger(__name__)
        
        try:
            # Load the FAISS index
            index = faiss.read_index(os.path.join(directory, "vector_store.faiss"))
            dimension = index.d  # Get the embedding dimension from the index
            logger.info("Loaded FAISS index with dimension %d.", dimension)

            # Initialize the vector store with the correct dimension
            vector_store = cls(dimension=dimension, logger=logger)
            vector_store.index = index
            
            # Load the chunk data and recreate embeddings
            with open(os.path.join(directory, "chunks.pkl"), "rb") as f:
                chunks_data = pickle.load(f)
            
            vector_store.chunks = [
                TextChunk(**chunk_data, embedding=nlp(chunk_data['text']).vector) 
                for chunk_data in chunks_data
            ]
            logger.info("Loaded vector store and %d chunks from %s.", len(vector_store.chunks), directory)
            return vector_store
        except FileNotFoundError as e:
            logger.error("File not found: %s", e)
            raise
        except Exception as e:
            logger.exception("Failed to load the vector store: %s", e)
            raise
