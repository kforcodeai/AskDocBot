import json
import os
from datetime import datetime
from typing import List, Dict, Any
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.data_models import QAEncoder, QAResponse
from src.chunking import SemanticPDFChunker


class OpenAIPDFQuestionAnswering:
    """System to handle PDF-based question answering using OpenAI API."""
    def __init__(self, config, api_key: str, logger):
        """Initialize with configuration and API key, and set up required components."""
        try:
            self.logger = logger
            self.embedding_model_name = config.get("DEV", "embedding_model_name", fallback="infgrad/stella-base-en-v2")
            self.model_name = config.get("DEV", "model_name", fallback="gpt-4o-mini")
            self.client = openai.OpenAI(api_key=api_key)
            self.chunker = SemanticPDFChunker(config=config, logger=self.logger)
            self.config = config
            self.logger.info("Initialized OpenAIPDFQuestionAnswering instance.")
        except Exception as e:
            self.logger.error("Failed to initialize OpenAIPDFQuestionAnswering instance.", exc_info=e)
            raise

    def format_context(self, relevant_chunks: List[Dict]) -> str:
        """Combine relevant PDF chunks into a formatted context string."""
        context_parts = []
        for chunk in relevant_chunks:
            context_parts.append(
                f"[Page {chunk['page']}, Section: {chunk['section']}]\n{chunk['text']}"
            )
        return "\n\n".join(context_parts)

    def get_answer_from_openai(self, question: str, context: str) -> Dict[str, Any]:
        """Send a formatted question and context to OpenAI API and parse the response."""
        try:
            prompt = f"""
            Based on the following context, please answer the question. If the answer cannot be determined from the context, respond with "NA".
            
            Context:
            {context}
            
            Question: {question}
            
            Please provide your response in the following JSON format:
            {{
                "answer": "your answer here",
                "confidence_score": float between 0 and 1
            }}
            """
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            result = json.loads(response.choices[0].message.content)
            self.logger.info("Received response from OpenAI API.")
            return result
        except json.JSONDecodeError as e:
            self.logger.error("Failed to parse OpenAI API response.", exc_info=e)
            return {"answer": "NA", "confidence_score": 0.0}
        except Exception as e:
            self.logger.error("Error querying OpenAI API.", exc_info=e)
            return {"answer": "NA", "confidence_score": 0.0}

    def process_single_question(self, pdf_path: str, question: str) -> QAResponse:
        """Process a single question and extract the answer."""
        try:
            chunk_result = self.chunker.answer_question(question)
            context = self.format_context(chunk_result["relevant_chunks"])
            openai_response = self.get_answer_from_openai(question, context)

            response = QAResponse(
                question=question,
                answer=openai_response["answer"],
                confidence_score=openai_response["confidence_score"],
                relevant_chunks=chunk_result["relevant_chunks"],
                source_pages=list(set(chunk["page"] for chunk in chunk_result["relevant_chunks"])),
            )
            self.logger.info(f"Processed question: {question}")
            return response
        except Exception as e:
            self.logger.error(f"Error processing question '{question}'", exc_info=e)
            return QAResponse(question=question, answer="Error processing question", confidence_score=0.0)

    def process_questions(self, pdf_path: str, questions: List[str]) -> Dict[str, List[QAResponse]]:
        """Process multiple questions concurrently and extract answers with associated metadata."""
        try:
            vector_store = self.chunker.process_pdf_to_vectors(pdf_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            vector_store_path = os.path.join(self.config.get("DEV", "output_folder"), f"vector_store_{timestamp}")
            vector_store.save(vector_store_path)
            self.logger.info(f"Saved vector store at {vector_store_path}")

            results = []
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.process_single_question, pdf_path, question) for question in questions]
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        self.logger.error("Error processing question in thread.", exc_info=e)

            self.logger.info("Completed processing all questions.")
            return {"timestamp": timestamp, "pdf_path": pdf_path, "results": results}
        except Exception as e:
            self.logger.error("Error processing multiple questions.", exc_info=e)
            return {"timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"), "pdf_path": pdf_path, "results": []}