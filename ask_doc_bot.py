import os
import json
import argparse
import logging
from dotenv import load_dotenv, find_dotenv
from configparser import ConfigParser
from src.data_models import QAEncoder, QAResponse
from src.chunking import SemanticPDFChunker
from src.embedding import EmbeddingGenerator
from src.vectorstorage import SemanticVectorStore
from src.openai_qa import OpenAIPDFQuestionAnswering
from src.services import SlackService
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import datetime

# Configure the logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config():
    """Load environment variables and configuration file."""
    load_dotenv(find_dotenv())
    config = ConfigParser()
    config.read("config.ini")
    return config


def get_api_keys():
    """Retrieve API keys from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    slack_token = os.getenv("SLACK_API_TOKEN")
    if not api_key or not slack_token:
        logger.error("API_KEY or SLACK_API_TOKEN is missing in environment variables.")
        raise EnvironmentError(
            "API_KEY or SLACK_API_TOKEN is missing in environment variables."
        )
    return api_key, slack_token


def format_results(questions, answers) -> dict:
    """Format results using the format from `format_results_json`."""
    results = {"results": []}
    for qa_response in answers:
        qa_response = qa_response.__dict__
        question = qa_response["question"]
        answer = qa_response["answer"]
        confidence = qa_response["confidence_score"]
        source_pages = qa_response["source_pages"]
        if confidence < 0.5:
            answer = "Data Not Available"
        results["results"].append(
            {
                "question": question,
                "answer": answer,
                "confidence_score": confidence,
                "source_pages": source_pages,
            }
        )
    return results


def format_context(relevant_chunks: List[Dict]) -> str:
    """Combine relevant chunks into a context string."""
    context_parts = [
        f"[Page {chunk['page']}] {chunk['text']}" for chunk in relevant_chunks
    ]
    return "\n\n".join(context_parts)


def retreival(vectorizer, question, vector_store):
    """Retreive relevant chunks for a question"""
    try:
        question_embed = vectorizer.compute_embedding(question)
        relevant_chunks = vector_store.search(question_embed)
        context = "\n".join(
            [
                f"[Section: {chunk.section_title}] {chunk.text}"
                for chunk, score in relevant_chunks
            ]
        )
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
            "context": context,
        }
    except Exception as e:
        logger.error("Error answering question: %s", str(e))
        return {"question": question, "relevant_chunks": [], "context": ""}


def answer_one_question(
    vectorizer: EmbeddingGenerator,
    vector_store: SemanticVectorStore,
    qa_system: OpenAIPDFQuestionAnswering,
    question: str,
):
    chunk_result = retreival(vectorizer, question, vector_store)
    context = format_context(chunk_result["relevant_chunks"])
    openai_response = qa_system.get_answer_from_openai(question, context)
    response = QAResponse(
        question=question,
        answer=openai_response["answer"],
        confidence_score=openai_response["confidence_score"],
        relevant_chunks=chunk_result["relevant_chunks"],
        source_pages=list(
            set(chunk["page"] for chunk in chunk_result["relevant_chunks"])
        ),
    )
    return response


def main(pdf_path, questions, user_command, slack_channel):
    config = load_config()
    api_key, slack_token = get_api_keys()

    chunker = SemanticPDFChunker(config=config, logger=logger)
    vectorizer = EmbeddingGenerator(config=config, logger=logger)
    vector_store = SemanticVectorStore(config=config, logger=logger)
    qa_system = OpenAIPDFQuestionAnswering(
        config=config, api_key=api_key, logger=logger
    )
    slack_service = SlackService(slack_token)

    # Chunk the PDF
    chunks = chunker(pdf_path)
    chunks = vectorizer(chunks)
    # store the chunks
    vector_store = vector_store(chunks)

    # Answer Questions
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                answer_one_question, vectorizer, vector_store, qa_system, question
            )
            for question in questions
        ]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error("Error processing question in thread.", exc_info=e)

    logger.info("Completed processing all questions.")

    # Format Results
    formatted_results = format_results(questions, results)

    # Save to file
    output_file = os.path.join(
        config.get("DEV", "output_folder", fallback="artifacts"), "qa_results.json"
    )
    with open(output_file, "w") as f:
        json.dump(formatted_results, f, cls=QAEncoder, indent=2)
    logger.info(f"Results saved to {output_file}")

    if "post results on Slack" in user_command.lower():
        slack_service.post_message(
            slack_channel, json.dumps(results, cls=QAEncoder, indent=2)
        )
        logger.info(f"Results posted to Slack channel: {slack_channel}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI for PDF-based question answering."
    )
    parser.add_argument("pdf_path", type=str, help="Path to the PDF document.")
    parser.add_argument(
        "questions", nargs="+", type=str, help="List of questions to answer."
    )
    parser.add_argument(
        "--user_command",
        type=str,
        default="",
        help="Command with instructions for posting results.",
    )
    parser.add_argument(
        "--slack_channel",
        type=str,
        default="#general",
        help="Optional Slack channel to post results.",
    )
    args = parser.parse_args()
    main(args.pdf_path, args.questions, args.user_command, args.slack_channel)
