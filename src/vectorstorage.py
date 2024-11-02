import faiss
import numpy as np
from typing import List, Tuple, Dict, Any
from src.data_models import TextChunk
import logging
import pickle
import sys
import os

class SemanticVectorStore:
    def __init__(self, config:Dict, logger=logging.getLogger(__name__)):
        """Initialize the vector storage."""
        dimension = int(config.get("DEV", "embedding_size"))
        self.logger = logger
        self.dimension = dimension
        self.nn = int(config.get("DEV", "num_nearest_neighbours"))
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []
        self.logger.info("Initialized SemanticVectorStore with dimension %d.", dimension)

    def add_chunks(self, chunks: List[TextChunk]):
        """Add chunks with precomputed embeddings to the vector store."""
        embeddings = [chunk.embedding for chunk in chunks if chunk.embedding is not None]
        embeddings_array = np.array(embeddings, dtype='float32')
        self.index.add(embeddings_array)
        self.chunks.extend(chunks)
        self.logger.info("Added %d chunks to the vector store.", len(chunks))

    def search(self, query_embedding: np.ndarray) -> List[Tuple[TextChunk, float]]:
        """Search for similar chunks based on the query embedding."""
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_embedding, k=self.nn)
        results = [(self.chunks[idx], float(dist)) for idx, dist in zip(indices[0], distances[0]) if idx != -1]
        self.logger.info("Search completed, returning %d results.", len(results))
        return results
    
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

    def __call__(self, chunks:List[TextChunk])->Any:
        try:
            self.add_chunks(chunks)
            self.logger.info("chunks stored to vector store successfully.")
            return self
        except Exception as e:
            self.logger.error("Error processing PDF to vectors: %s", str(e))
            sys.exit(1)