import fitz  # PyMuPDF for PDF handling
import spacy
from typing import List, Dict, Tuple
import numpy as np
import re
from src.core.data_models import TextChunk
import sys


class SemanticPDFChunker:
    def __init__(self, config: Dict, logger):
        """
        Initialize the chunker with NLP processing tools.

        Parameters:
        - config (dict): Configuration dictionary to set up model and parameters.
        - logger
        """
        try:
            self.logger = logger
            # Load model and tokenizer using parameters from the configuration
            self.embedding_model_name = config.get("DEV", "embedding_model_name")
            self.embedding_size = int(config.get("DEV", "embedding_size"))
            self.max_chunk_size = int(config.get("DEV", "max_chunk_size"))
            self.min_chunk_size = int(config.get("DEV", "min_chunk_size"))
            self.max_length = int(config.get("DEV", "max_length"))
            # Initialize spaCy for text processing without word vectors
            self.nlp = spacy.load("en_core_web_sm")

            # Define common section header patterns
            self.section_patterns = [
                r"^(?:CHAPTER|Section)\s+\d+[\.:]\s*(.+)$",
                r"^\d+[\.:]\d*\s+(.+)$",
                r"^[A-Z][A-Z\s]+(?:\:|$)",
            ]
            self.logger.info("Initialized SemanticPDFChunker successfully.")
        except Exception as e:
            self.logger.error("Error initializing SemanticPDFChunker: %s", str(e))
            sys.exit(1)

    def extract_structured_text(self, pdf_path: str) -> List[Dict]:
        """Extract structured text with formatting information from a PDF."""
        try:
            doc = fitz.open(pdf_path)
            pages = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]

                page_content = []
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            if "spans" in line:
                                for span in line["spans"]:
                                    page_content.append(
                                        {
                                            "text": span["text"],
                                            "font_size": span["size"],
                                            "font_name": span["font"],
                                            "is_bold": span.get("flags", 0) & 2 > 0,
                                            "bbox": span["bbox"],
                                        }
                                    )
                pages.append(page_content)
            self.logger.info("Extracted structured text from PDF successfully.")
            return pages
        except Exception as e:
            self.logger.error("Error extracting text from PDF: %s", str(e))
            return []

    def identify_section_headers(self, text_block: Dict) -> bool:
        """Check if a text block is likely to be a section header."""
        text = text_block["text"].strip()
        for pattern in self.section_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        return (
            text_block["font_size"] > 11
            and len(text.split()) < 10
            and text_block["is_bold"]
            and not text.endswith(".")
        )

    def get_semantic_boundaries(self, text: str) -> List[int]:
        """Find semantic boundaries within a block of text using NLP processing."""
        try:
            doc = self.nlp(text)
            boundaries = [sent.end_char for sent in doc.sents]
            paragraph_breaks = [m.end() for m in re.finditer(r"\n\s*\n", text)]
            boundaries.extend(paragraph_breaks)

            for ent in doc.ents:
                if ent.label_ in ["ORG", "GPE", "DATE"]:
                    boundaries.append(ent.end_char)

            return sorted(set(boundaries))
        except Exception as e:
            self.logger.error("Error finding semantic boundaries: %s", str(e))
            return []

    def combine_small_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Combine smaller text chunks to ensure each meets minimum length requirements."""
        combined_chunks = []
        current_chunk = None

        for chunk in chunks:
            if current_chunk is None:
                current_chunk = chunk
            elif len(current_chunk.text) + len(chunk.text) < self.min_chunk_size:
                current_chunk.text += " " + chunk.text
            else:
                combined_chunks.append(current_chunk)
                current_chunk = chunk

        if current_chunk:
            combined_chunks.append(current_chunk)
        return combined_chunks

    def process_pdf(self, pdf_path: str) -> List[TextChunk]:
        """Extract and create semantic chunks from a PDF document."""
        try:
            pages = self.extract_structured_text(pdf_path)
            return self.create_semantic_chunks(pages)
        except Exception as e:
            self.logger.error("Error processing PDF: %s", str(e))
            return []

    def create_semantic_chunks(self, pages: List[Dict]) -> List[TextChunk]:
        """Generate semantic chunks from the structured text in a PDF document."""
        chunks = []
        current_section = ""
        current_subsection = ""
        current_text = ""
        current_page = 0

        try:
            for page_num, page_content in enumerate(pages):
                for block in page_content:
                    if self.identify_section_headers(block):
                        if current_text:
                            chunks.append(
                                TextChunk(
                                    text=current_text.strip(),
                                    page_num=current_page,
                                    section_title=current_section,
                                    subsection_title=current_subsection,
                                )
                            )
                            current_text = ""

                        if block["font_size"] > 14:
                            current_section = block["text"].strip()
                            current_subsection = ""
                        else:
                            current_subsection = block["text"].strip()
                    else:
                        current_text += block["text"] + " "
                        current_page = page_num

                        if len(current_text) >= self.max_chunk_size:
                            boundaries = self.get_semantic_boundaries(current_text)
                            if boundaries:
                                middle = len(current_text) // 2
                                best_boundary = min(boundaries, key=lambda x: abs(x - middle))

                                chunks.append(
                                    TextChunk(
                                        text=current_text[:best_boundary].strip(),
                                        page_num=current_page,
                                        section_title=current_section,
                                        subsection_title=current_subsection,
                                    )
                                )
                                current_text = current_text[best_boundary:]

            if current_text:
                chunks.append(
                    TextChunk(
                        text=current_text.strip(),
                        page_num=current_page,
                        section_title=current_section,
                        subsection_title=current_subsection,
                    )
                )
            return chunks
        except Exception as e:
            self.logger.error("Error creating semantic chunks: %s", str(e))
            return []
    
    def __call__(self, pdf_path: str) -> List:
        """Process a PDF file, extract text, compute embeddings, and store in vector store."""
        try:
            chunks = self.process_pdf(pdf_path)
            combined_chunks = self.combine_small_chunks(chunks)
            return combined_chunks
        except Exception as e:
            self.logger.error("Error processing PDF to vectors: %s", str(e))
            return []
