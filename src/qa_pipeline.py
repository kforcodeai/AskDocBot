from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import json
from src.core.chunking import SemanticPDFChunker
from src.core.embedding import EmbeddingGenerator
from src.core.vectorstorage import SemanticVectorStore
from src.core.qa import OpenAIPDFQuestionAnswering
from src.services.slack import SlackService
from src.core.data_models import QAResponse, QAEncoder
from src.utils.logger import LoggerManager
from src.utils.filemanager import FileManager


class QAPipeline:
    def __init__(self, config_manager):
        self.file_manager = FileManager()
        logger = LoggerManager().get_logger(__name__)
        self.config = config_manager.get_config()
        self.api_config = config_manager.get_api_config()
        self._initialize_core_services(logger)

    def _initialize_core_services(self, logger):
        self.chunker = SemanticPDFChunker(config=self.config, logger=logger)
        self.vectorizer = EmbeddingGenerator(config=self.config, logger=logger)
        self.vector_store = SemanticVectorStore(config=self.config, logger=logger)
        self.qa_system = OpenAIPDFQuestionAnswering(
            config=self.config, api_key=self.api_config.openai_api_key, logger=logger
        )
        self.slack_service = SlackService(self.api_config.slack_token)
        self.logger = logger

    def process(
        self, pdf_path: str, questions: List[str], user_command: str, slack_channel: str
    ) -> None:
        try:
            # Process PDF
            chunks = self._process_pdf(pdf_path)

            # Answer questions
            results = self._answer_questions(questions, chunks)

            # Format and save results
            self._handle_results(results, user_command, slack_channel)

        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {str(e)}")
            raise

    def _process_pdf(self, pdf_path: str):
        chunks = self.chunker(pdf_path)
        chunks = self.vectorizer(chunks)
        return self.vector_store(chunks)

    def _answer_questions(self, questions: List[str], chunks) -> List[QAResponse]:
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._process_single_question, question, chunks)
                for question in questions
            ]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    self.logger.error(
                        "Error processing question in thread.", exc_info=e
                    )
        return results

    def _process_single_question(self, question: str, chunks) -> QAResponse:
        chunk_result = self._retrieve_relevant_chunks(question)
        context = self._format_context(chunk_result["relevant_chunks"])
        openai_response = self.qa_system.get_answer(question, context)

        return QAResponse(
            question=question,
            answer=openai_response["answer"],
            confidence_score=openai_response["confidence_score"],
            relevant_chunks=chunk_result["relevant_chunks"],
            source_pages=list(
                set(chunk["page"] for chunk in chunk_result["relevant_chunks"])
            ),
        )

    def _retrieve_relevant_chunks(self, question: str) -> Dict:
        try:
            question_embed = self.vectorizer.compute_embedding(question)
            relevant_chunks = self.vector_store.search(question_embed)

            return {
                "question": question,
                "relevant_chunks": [
                    {
                        "text": chunk.text,
                        "section": chunk.section_title,
                        "page": chunk.page_num + 1,
                        "relevance_score": float(score),
                    }
                    for chunk, score in relevant_chunks
                ],
                "context": self._format_context(
                    [
                        {"text": chunk.text, "section": chunk.section_title}
                        for chunk, _ in relevant_chunks
                    ]
                ),
            }
        except Exception as e:
            self.logger.error(f"Error retrieving chunks: {str(e)}")
            return {"question": question, "relevant_chunks": [], "context": ""}

    def _format_context(self, chunks: List[Dict]) -> str:
        return "\n\n".join(
            f"[Section: {chunk['section']}] {chunk['text']}" for chunk in chunks
        )

    def _handle_results(
        self, results: List[QAResponse], user_command: str, slack_channel: str
    ) -> None:
        formatted_results = self._format_results(results)

        result_file = self.file_manager.current_result_file

        try:
            with open(result_file, "w") as f:
                json.dump(formatted_results, f, cls=QAEncoder, indent=2)
            self.logger.info(f"Results saved to {result_file}")

            # Post to Slack if requested
            if "post results on slack" in user_command.lower():
                slack_message = (
                    f"Results for run {self.file_manager.run_id}:\n"
                    f"{json.dumps(formatted_results, cls=QAEncoder, indent=2)}"
                )
                self.slack_service.post_message(slack_channel, slack_message)
                self.logger.info(f"Results posted to Slack channel: {slack_channel}")

        except Exception as e:
            self.logger.error(f"Error handling results: {str(e)}", exc_info=True)
            raise

    def _format_results(self, qa_responses: List[QAResponse]) -> Dict:
        results = {"results": []}
        for response in qa_responses:
            answer = response.answer
            if response.confidence_score < 0.5:
                answer = "Data Not Available"

            results["results"].append(
                {
                    "question": response.question,
                    "answer": answer,
                    "confidence_score": response.confidence_score,
                    "source_pages": response.source_pages,
                }
            )
        return results
